"""模型下载：镜像轮换 + 断点续传 + 进度回报。

**镜像速度会变，而且变得很快**（实测同一个镜像几个月内从 2.7 掉到 0.13 MB/s），
所以不认死一个，按 MIRRORS 顺序逐个试、挂了自动换。也因为最快的镜像也只在
0.6 MB/s 量级而 SenseVoice 那个档案有 1GB，**续传是必须的**：断点存在
`<目标>.part` 里，换镜像、换进程都接着用。
"""
from __future__ import annotations

import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from lecture.config import AsrConfig

# 按实测速度排序，第一个通常最快。挂了自动换下一个
MIRRORS = (
    "https://ghfast.top/https://github.com",
    "https://gh.llkk.cc/https://github.com",
    "https://ghproxy.net/https://github.com",
    "https://github.com",
)

RELEASE_PATH = "/k2-fsa/sherpa-onnx/releases/download/asr-models"

# 不少 GitHub 加速镜像会拒绝 Python-urllib 的默认 UA，必须伪装
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

CHUNK = 1 << 18


@dataclass
class Progress:
    """一次进度回报。done/total 是字节，total 为 0 表示服务器没给长度。"""
    stage: str          # download | extract | prune
    label: str
    done: int = 0
    total: int = 0

    @property
    def pct(self) -> int:
        return int(self.done * 100 / self.total) if self.total else 0

    def __str__(self) -> str:
        if self.stage != "download":
            return f"{self.stage} {self.label}"
        if self.total:
            return (f"{self.label} {self.pct}% "
                    f"({self.done / 1e6:.0f}/{self.total / 1e6:.0f} MB)")
        return f"{self.label} {self.done / 1e6:.0f} MB"


Reporter = Callable[[Progress], None]
Logger = Callable[[str], None]


class DownloadError(RuntimeError):
    pass


# ---------------- 模型清单 ----------------
#
# 归档名由模型目录名加后缀得到（上游发布时就是这个规律），所以不用另维护一张表——
# 换模型只要改 config.yaml，这里跟着走。

def _archive_name(model_dir_name: str) -> str:
    return f"{model_dir_name}.tar.bz2"


@dataclass
class Item:
    """要下的一样东西。archive 为 True 时是 tar.bz2，下完要解压。"""
    name: str           # 归档名或文件名
    target: Path        # 下完之后应该存在的目录或文件
    desc: str
    archive: bool


def wanted(cfg: AsrConfig) -> list[Item]:
    return [
        Item(_archive_name(cfg.streaming_model), cfg.streaming_dir,
             "实时字幕（流式 Paraformer）", True),
        Item(_archive_name(cfg.offline_model), cfg.offline_dir,
             "句子定稿（SenseVoice）", True),
        Item(cfg.vad_model, cfg.vad_path, "断句 VAD", False),
    ]


def missing(cfg: AsrConfig) -> list[Item]:
    """还缺哪些。空列表 = 模型齐了。"""
    return [it for it in wanted(cfg) if not it.target.exists()]


def status(cfg: AsrConfig) -> list[tuple[str, Path, bool, int]]:
    """(说明, 路径, 在不在, 字节数)。`models` 子命令和界面都用它。"""
    out = []
    for it in wanted(cfg):
        ok = it.target.exists()
        size = 0
        if ok:
            size = (sum(f.stat().st_size for f in it.target.rglob("*") if f.is_file())
                    if it.target.is_dir() else it.target.stat().st_size)
        out.append((it.desc, it.target, ok, size))
    return out


# ---------------- 下载 ----------------

def _open(url: str, offset: int, timeout: float):
    """发请求。offset > 0 时带 Range 头。返回 (response, 是否续上了)。"""
    headers = {"User-Agent": UA, "Accept": "*/*"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    req = urllib.request.Request(url, headers=headers)
    resp = urllib.request.urlopen(req, timeout=timeout)
    # 206 = 服务器认了 Range；200 = 不认，只能从头写
    return resp, bool(offset) and resp.status == 206


def _fetch_one(url: str, dest: Path, label: str,
               report: Reporter | None, timeout: float) -> None:
    """下一个文件到 dest，支持续传。失败抛 DownloadError。

    先写 `<dest>.part`，完整了才改名——半个文件留在正式位置上会被
    `missing()` 当成「已经有了」，那种错很难查。
    """
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0

    try:
        resp, resumed = _open(url, have, timeout)
    except urllib.error.HTTPError as exc:
        # 416 = 请求的区间超出文件长度，说明 .part 已经是完整的了
        if exc.code == 416 and have:
            part.replace(dest)
            return
        raise DownloadError(f"HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise DownloadError(str(exc)) from exc

    if have and not resumed:
        have = 0        # 服务器不支持 Range，白攒了，从头来

    total = int(resp.headers.get("Content-Length") or 0)
    if resumed:
        total += have   # Range 响应里的长度是「剩下多少」，不是整个文件

    done = have
    try:
        with resp, part.open("ab" if have else "wb") as f:
            while True:
                buf = resp.read(CHUNK)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if report:
                    report(Progress("download", label, done, total))
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        # .part 留着，下一个镜像接着下
        raise DownloadError(f"传到 {done / 1e6:.0f}MB 断了：{exc}") from exc

    if total and done < total:
        raise DownloadError(f"只收到 {done / 1e6:.0f}/{total / 1e6:.0f} MB")
    part.replace(dest)


def fetch(name: str, dest: Path, label: str, *, mirrors: Iterable[str] = MIRRORS,
          report: Reporter | None = None, log: Logger = print,
          timeout: float = 60.0) -> None:
    """按镜像顺序下 name 到 dest。全都失败才抛。"""
    tried = []
    for mirror in mirrors:
        url = mirror.rstrip("/") + RELEASE_PATH + "/" + name
        try:
            _fetch_one(url, dest, label, report, timeout)
            return
        except DownloadError as exc:
            host = mirror.split("//")[-1].split("/")[0]
            log(f"[下载] {host} 不行（{exc}），换下一个")
            tried.append(f"{host}: {exc}")
    raise DownloadError("所有镜像都失败了：\n  " + "\n  ".join(tried))


# ---------------- 解压与瘦身 ----------------

def prune(model_dir: Path) -> int:
    """删掉 fp32 权重和测试音频，只留 int8。推理只用 int8，体积能省一大半。"""
    freed = 0
    for f in list(model_dir.rglob("*.onnx")):
        if f.name.endswith(".int8.onnx"):
            continue
        if f.with_name(f.name.replace(".onnx", ".int8.onnx")).exists():
            freed += f.stat().st_size
            f.unlink()
    for d in list(model_dir.rglob("test_wavs")):
        if d.is_dir():
            freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d)
    return freed


def ensure(cfg: AsrConfig, *, report: Reporter | None = None, log: Logger = print,
           mirrors: Iterable[str] = MIRRORS, keep_fp32: bool = False,
           force: bool = False) -> int:
    """把缺的模型补齐。返回下了几样。模型齐了就什么都不做。

    下载量约 1.5GB，prune 之后落盘约 477MB——差的那一倍是 fp32 权重和
    测试音频，解压后才删得掉，所以省不了流量。
    """
    todo = wanted(cfg) if force else missing(cfg)
    if not todo:
        return 0

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    for item in todo:
        log(f"[下载] {item.desc}：{item.name}")
        if not item.archive:
            fetch(item.name, item.target, item.name,
                  mirrors=mirrors, report=report, log=log)
            continue

        # 归档下到临时目录，解压到 models_dir。临时目录里的 .part 会跟着
        # 目录一起消失，所以中途失败的续传只在同一次调用内有效——归档路径
        # 因此放在 models_dir 下，让续传能跨进程活下来
        tar_path = cfg.models_dir / item.name
        fetch(item.name, tar_path, item.name,
              mirrors=mirrors, report=report, log=log)
        if report:
            report(Progress("extract", item.name))
        log(f"[解压] {item.name}")
        try:
            with tarfile.open(tar_path, "r:bz2") as tf:
                tf.extractall(cfg.models_dir, filter="data")
        except (tarfile.TarError, OSError) as exc:
            tar_path.unlink(missing_ok=True)     # 坏档案留着只会让续传一直失败
            raise DownloadError(f"{item.name} 解压失败（档案可能不完整）：{exc}") from exc
        tar_path.unlink(missing_ok=True)

        if not item.target.exists():
            raise DownloadError(f"解压后没找到 {item.target}")
        if not keep_fp32:
            if report:
                report(Progress("prune", item.name))
            freed = prune(item.target)
            if freed:
                log(f"[瘦身] 清掉 fp32/测试音频，省下 {freed / 1e6:.0f} MB")

    return len(todo)


def total_bytes(cfg: AsrConfig) -> int:
    """还要下多少字节（粗估，用于界面上显示总进度）。

    档案大小是写死的实测值：为了显示一个进度条去发三个 HEAD 请求，
    而镜像本来就慢、还可能不返回 Content-Length，不值得。
    """
    sizes = {
        _archive_name(cfg.streaming_model): 450_000_000,
        _archive_name(cfg.offline_model): 1_048_000_000,
        cfg.vad_model: 643_854,
    }
    return sum(sizes.get(it.name, 0) for it in missing(cfg))
