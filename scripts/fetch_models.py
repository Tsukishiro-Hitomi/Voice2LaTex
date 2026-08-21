"""下载 sherpa-onnx 模型到 models/。

    python scripts/fetch_models.py
    python scripts/fetch_models.py --base-url https://ghfast.top/https://github.com

只是 `lecture/models.py` 的命令行外壳，和 `python -m lecture fetch` 等价。
默认按顺序试几个镜像（见 `lecture.models.MIRRORS`），挂了自动换，中途断了
下次接着下。`--base-url` 只在你想指定某一个镜像时才需要。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lecture import config, models      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="配置文件路径（默认 config.yaml）")
    ap.add_argument("--base-url", help="只用这一个镜像，默认按内置顺序逐个试")
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    ap.add_argument("--keep-fp32", action="store_true",
                    help="保留 fp32 权重和测试音频（默认删掉，只留 int8）")
    args = ap.parse_args()

    cfg = config.load(args.config).asr
    mirrors = (args.base_url,) if args.base_url else models.MIRRORS

    last = ""

    def report(p: models.Progress) -> None:
        # 同一行覆盖刷新，别让进度把日志刷满屏
        nonlocal last
        line = str(p)
        if line != last:
            sys.stderr.write("\r" + line.ljust(len(last)))
            sys.stderr.flush()
            last = line

    def log(msg: str) -> None:
        nonlocal last
        if last:
            sys.stderr.write("\n")
            last = ""
        print(msg)

    try:
        n = models.ensure(cfg, report=report, log=log, mirrors=mirrors,
                          keep_fp32=args.keep_fp32, force=args.force)
    except models.DownloadError as exc:
        log(f"\n下载失败：{exc}")
        return 1
    if last:
        sys.stderr.write("\n")

    print("已经齐了，什么都没做。" if n == 0 else f"下好了 {n} 样。")
    print("\n模型状态：")
    for desc, path, ok, size in models.status(cfg):
        mark = "ok" if ok else "--"
        print(f"  [{mark}] {desc:24s} {size / 1e6:7.1f} MB  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
