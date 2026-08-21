"""命令行入口。子命令列表见 `python -m lecture --help`。"""
from __future__ import annotations

import argparse
import shutil
import sys
import threading
import time
import unicodedata
from pathlib import Path

from lecture import config, latex, refine
from lecture.audio import CHUNK_SECONDS, list_devices, open_source
from lecture.engine import Recorder, pick_refine_endpoint
from lecture.llm import LlmError, cloud_endpoint
from lecture.store import Session


# ---------------- 终端显示 ----------------

def _width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


class Console:
    """实时字幕用同一行覆盖刷新，定稿和日志正常换行输出。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._partial_w = 0
        self._tty = sys.stdout.isatty()   # 重定向到文件时不刷局部字幕，否则日志全是 \r 垃圾

    def line(self, text: str) -> None:
        with self._lock:
            self._erase()
            sys.stdout.write(text + "\n")
            sys.stdout.flush()

    def partial(self, text: str) -> None:
        if not text or not self._tty:
            return
        cols = shutil.get_terminal_size((100, 24)).columns - 6
        while _width(text) > cols:
            text = text[1:]
        s = "  > " + text
        with self._lock:
            pad = max(0, self._partial_w - _width(s))
            sys.stdout.write("\r" + s + " " * pad)
            sys.stdout.flush()
            self._partial_w = _width(s)

    def _erase(self) -> None:
        if self._partial_w:
            sys.stdout.write("\r" + " " * self._partial_w + "\r")
            self._partial_w = 0


class KeyReader:
    """非阻塞读单个按键，用来在课上暂停/继续。

    Windows 走 msvcrt，POSIX 走 termios cbreak（cbreak 保留 ISIG，Ctrl-C 照样能用）。
    stdin 不是终端时（重定向、后台跑）自动禁用，不报错。
    """

    def __init__(self) -> None:
        try:
            self.enabled = sys.stdin.isatty()
        except Exception:
            self.enabled = False
        self._win = sys.platform == "win32"
        self._saved = None

    def __enter__(self) -> "KeyReader":
        if self.enabled and not self._win:
            try:
                import termios
                import tty
                self._saved = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            except Exception:
                self.enabled = False
        return self

    def get(self) -> str | None:
        if not self.enabled:
            return None
        try:
            if self._win:
                import msvcrt
                return msvcrt.getwch() if msvcrt.kbhit() else None
            import select
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
        except Exception:
            self.enabled = False
        return None

    def __exit__(self, *exc) -> None:
        if self._saved is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._saved)
            except Exception:
                pass


# ---------------- 采集主循环 ----------------

def _capture(cfg: config.Config, source: str, device: int | None, file: str | None,
             title: str, live_refine: bool, glossary: str | None = None) -> Session | None:
    """跑一次采集，把引擎状态渲染到终端。引擎本身在 lecture/engine.py。"""
    console = Console()
    rec = Recorder(cfg, title=title, source=source, device=device, file=file,
                   glossary=glossary, live_refine=live_refine)

    keys = KeyReader()
    if source != "file":
        console.line("[p] 暂停/继续   [q] 或 Ctrl-C 结束" if keys.enabled
                     else "Ctrl-C 结束")
    rec.start()

    n_sent = n_log = 0

    def drain() -> None:
        nonlocal n_sent, n_log
        snap = rec.snapshot(n_sent, n_log)
        for s in snap["sentences"]:
            console.line(f"[{s['t']}] {s['text']}")
        for line in snap["log"]:
            console.line(line)
        n_sent += len(snap["sentences"])
        n_log = snap["n_log"]
        console.partial(snap["partial"])

    try:
        with keys:
            while rec.alive:
                drain()
                key = keys.get()
                if key in ("p", "P"):
                    rec.toggle_pause()
                elif key in ("q", "Q"):
                    rec.request_stop()
                time.sleep(0.08)
    except KeyboardInterrupt:
        console.line("\n收到 Ctrl-C，收尾…")
        rec.request_stop()

    rec.join(timeout=120)
    drain()
    snap = rec.snapshot()
    if snap["error"]:
        console.line(f"\n失败：{snap['error']}")
        return None
    console.line(f"\n共 {snap['n']} 句 → {rec.session.path / 'raw.md'}")
    return rec.session


# ---------------- 子命令 ----------------

def cmd_devices(args, cfg) -> int:      # noqa: ARG001
    print(list_devices())
    return 0


def cmd_models(args, cfg) -> int:       # noqa: ARG001
    from lecture import models

    ok = True
    for desc, path, exists, size in models.status(cfg.asr):
        if exists:
            print(f"  [ok] {desc:24s} {size / 1e6:7.1f} MB  {path.name}")
        else:
            ok = False
            print(f"  [--] {desc:24s} 缺失            {path}")
    if not ok:
        need = models.total_bytes(cfg.asr)
        print(f"\n还要下约 {need / 1e6:.0f} MB（解压瘦身后落盘约一半）。"
              f"\n跑这个补齐：python -m lecture fetch")
    return 0 if ok else 1


def cmd_fetch(args, cfg) -> int:
    """把缺的模型下下来。镜像轮换 + 断点续传都在 lecture/models.py 里。"""
    from lecture import models

    last = ""

    def report(p) -> None:
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

    mirrors = (args.base_url,) if args.base_url else models.MIRRORS
    try:
        n = models.ensure(cfg.asr, report=report, log=log, mirrors=mirrors,
                          keep_fp32=args.keep_fp32, force=args.force)
    except models.DownloadError as e:
        log(f"下载失败：{e}")
        return 1
    if last:
        sys.stderr.write("\n")
    print("模型已经齐了。" if n == 0 else f"下好了 {n} 样。")
    return cmd_models(args, cfg)


def cmd_check(args, cfg) -> int:
    """录一小段，报告音量和识别情况。上课前花 10 秒确认设备选对了、声音够大。"""
    from lecture.asr import Transcriber

    import numpy as np

    where = {"mic": "麦克风", "loopback": "系统音频", "file": "文件"}[args.source]
    tip = "，请正常音量说几句话（带点专业名词更有参考价值）" if args.source != "file" else ""
    # 和 Recorder 一样回退到配置，否则 check 用的设备和实际录课的不是一个，
    # 那这一步就白验了
    device = args.device if args.device is not None else cfg.asr.device
    print(f"从{where}采集 {args.seconds:.0f} 秒{tip}…"
          + (f"（设备 {device}）" if device is not None else "（系统默认设备）"))
    stop = threading.Event()
    stream = open_source(args.source, device=device, file=args.file, stop=stop)
    tr = Transcriber(cfg.asr)

    want = int(args.seconds / CHUNK_SECONDS)
    peak = 0.0
    sq = 0.0
    n_samples = 0
    texts = []
    for i, chunk in enumerate(stream):
        peak = max(peak, float(np.abs(chunk).max()))
        sq += float((chunk.astype(np.float64) ** 2).sum())
        n_samples += len(chunk)
        for f in tr.feed(chunk)[1]:
            texts.append(f.text)
        if i + 1 >= want:
            break
    stop.set()
    stream.close()
    texts += [f.text for f in tr.flush()]

    rms = (sq / max(1, n_samples)) ** 0.5
    dbfs = 20 * __import__("math").log10(rms) if rms > 0 else -99
    print(f"\n时长 {n_samples / 16000:.1f}s   RMS {rms:.4f} ({dbfs:.0f} dBFS)   峰值 {peak:.3f}")

    if rms < 0.002:
        print("→ 几乎是静音。检查：设备选对了吗（devices 看编号，--device N 指定）？"
              "麦克风被系统静音了吗？Windows 上还要确认「隐私设置 → 麦克风」允许了终端。")
    elif rms < 0.01:
        print("→ 偏轻。能识别但错字会多。把麦克风放近点，或在系统里调高输入音量。")
    elif peak > 0.99:
        print("→ 削波了（峰值触顶），失真会导致识别变差。调低输入音量。")
    else:
        print("→ 音量正常。")

    if texts:
        print(f"\n识别出 {len(texts)} 句：")
        for t in texts:
            print(f"  {t}")
        print("\n能出字就说明整条链路是通的，可以上课用了。")
    else:
        print("\n没识别出任何句子。如果刚才确实说了话，八成是音量太低（见上），"
              "或者说话不满 0.25 秒（min_speech_duration）就停了。")
    return 0


def cmd_live(args, cfg) -> int:
    _capture(cfg, args.source, args.device, args.file, args.title,
             live_refine=not args.no_refine, glossary=args.glossary)
    print("下一步：python -m lecture notes")
    return 0


def cmd_transcribe(args, cfg) -> int:
    path = Path(args.audio)
    if not path.exists():
        print(f"找不到文件：{path}", file=sys.stderr)
        return 1
    _capture(cfg, "file", None, str(path), args.title or path.stem, live_refine=False,
             glossary=args.glossary)
    print("下一步：python -m lecture notes")
    return 0


def _resolve(cfg, spec: str | None) -> Session:
    return Session.open(spec) if spec else Session.latest(cfg.sessions_dir)


def _catch_up(session: Session, cfg, glossary: str | None = None) -> None:
    # 课件刻意不传给清洗阶段：实测本地小模型分不清"参考符号"和"照抄内容"，
    # 会把课件的措辞和目录搬进笔记。课上这一步只认 glossary.txt。
    total = len(session.sentences())
    print(f"原稿 {total} 句，已清洗到第 {session.refined_until} 句")
    ep = pick_refine_endpoint(cfg, print)
    if ep is None:
        print("没有可用的清洗后端", file=sys.stderr)
        return
    n = refine.run(session, cfg, ep, flush=True, glossary_spec=glossary)
    print(f"新增 {n} 个片段，清洗到第 {session.refined_until}/{total} 句")


def cmd_refine(args, cfg) -> int:
    session = _resolve(cfg, args.session)
    print(f"session: {session.path}")
    _catch_up(session, cfg, _glossary_spec(args, session, cfg))
    return 0


def _glossary_spec(args, session: Session, cfg) -> str | None:
    """命令行指定优先，否则用 session 记住的那份。"""
    spec = getattr(args, "glossary", None)
    if spec:
        session.update_meta(glossary=spec)
    else:
        spec = session.meta.get("glossary")
    if spec:
        print(f"术语表: {cfg.resolve_glossary(spec).name}")
    return spec


def _load_deck(args, session: Session):
    if not getattr(args, "slides", None):
        return None
    from lecture import slides as slides_mod
    deck = slides_mod.extract(args.slides, session.path)
    n_fig = len(deck.figures)
    print(f"课件: {deck.source.name}，{len(deck.slides)} 页，抽出 {n_fig} 张图"
          + (f" → {session.path / 'figures'}" if n_fig else "（没找到够大的图）"))
    session.update_meta(slides=str(Path(args.slides).resolve()))
    return deck


def _make_notes(session: Session, cfg, args) -> int:
    deck = _load_deck(args, session)     # 课件只给整合用，清洗阶段刻意不给，见 _catch_up
    if session.refined_until < len(session.sentences()):
        print("有没清洗的句子，先补跑清洗…")
        _catch_up(session, cfg, _glossary_spec(args, session, cfg))

    ep = cloud_endpoint(cfg.llm, getattr(args, "model", None))
    tex, md = latex.compose(session, cfg, ep, deck=deck,
                            max_chars=getattr(args, "max_chars", 6000))
    print(f"\n笔记: {md}\nLaTeX: {tex}\n原稿: {session.path / 'raw.md'}")
    # 必须在 session 目录里编译，否则 figures/ 相对路径找不到
    print(f"\n编译: cd {session.path} && xelatex notes.tex")
    return 0


def cmd_notes(args, cfg) -> int:
    session = _resolve(cfg, args.session)
    print(f"session: {session.path}")
    return _make_notes(session, cfg, args)


def cmd_video(args, cfg) -> int:
    """课程视频 → 笔记，一步到底。"""
    path = Path(args.video).expanduser()
    if not path.exists():
        print(f"找不到文件：{path}", file=sys.stderr)
        return 1
    session = _capture(cfg, "file", None, str(path), args.title or path.stem,
                       live_refine=False, glossary=args.glossary)
    if session is None:
        return 1
    print()
    return _make_notes(session, cfg, args)


def cmd_serve(args, cfg) -> int:
    from lecture.web import serve
    serve(cfg, host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lecture",
                                 description="上课语音助手：实时转写 → 清洗 → LaTeX 笔记")
    ap.add_argument("--config", help="配置文件路径（默认 config.yaml）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("devices", help="列出音频设备").set_defaults(fn=cmd_devices)
    sub.add_parser("models", help="检查模型是否就位").set_defaults(fn=cmd_models)

    p = sub.add_parser("fetch", help="下载缺的模型（镜像轮换 + 断点续传）")
    p.add_argument("--base-url", help="只用这一个镜像，默认按内置顺序逐个试")
    p.add_argument("--force", action="store_true", help="已存在也重新下载")
    p.add_argument("--keep-fp32", action="store_true", help="保留 fp32 权重和测试音频")
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("serve", help="启动网页界面（点按钮开始/暂停，字幕显示在页面上）")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8730)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("check", help="上课前自检：录一小段看音量和识别效果")
    p.add_argument("--source", default="mic", choices=["mic", "loopback", "file"])
    p.add_argument("--device", type=int, help="设备号，见 devices 命令")
    p.add_argument("--file", help="--source file 时的音频路径")
    p.add_argument("--seconds", type=float, default=10.0)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("live", help="上课实时转写")
    p.add_argument("--title", default="课堂", help="课程名，用于 session 目录名和笔记标题")
    p.add_argument("--source", default="mic", choices=["mic", "loopback"],
                   help="mic=麦克风(线下课) loopback=系统音频(网课，Windows)")
    p.add_argument("--device", type=int, help="设备号，见 devices 命令")
    p.add_argument("--no-refine", action="store_true", help="课上只转写，不做清洗")
    p.add_argument("--glossary", help="课程术语表：glossaries/ 下的名字（不带 .txt）或路径")
    p.set_defaults(fn=cmd_live, file=None)

    p = sub.add_parser("video", help="课程视频/录音 → 笔记，一条命令到底")
    p.add_argument("video", help="视频或音频文件（mp4/mkv/flv/m4a/mp3… ffmpeg 能读就行）")
    p.add_argument("--title", help="默认取文件名")
    p.add_argument("--glossary", help="课程术语表：glossaries/ 下的名字或路径")
    p.add_argument("--slides", help="课件 PDF/PPTX：校正术语并把图插进笔记")
    p.set_defaults(fn=cmd_video, session=None)

    p = sub.add_parser("transcribe", help="转写已有录音文件")
    p.add_argument("audio")
    p.add_argument("--title")
    p.add_argument("--glossary", help="课程术语表：glossaries/ 下的名字（不带 .txt）或路径")
    p.set_defaults(fn=cmd_transcribe)

    p = sub.add_parser("refine", help="补跑逐段清洗")
    p.add_argument("session", nargs="?", help="session 目录（默认最近一次）")
    p.add_argument("--glossary", help="课程术语表：glossaries/ 下的名字（不带 .txt）或路径")
    p.set_defaults(fn=cmd_refine)

    p = sub.add_parser("notes", help="整合成 LaTeX 笔记")
    p.add_argument("session", nargs="?", help="session 目录（默认最近一次）")
    p.add_argument("--model", help="覆盖配置里的整合模型")
    p.add_argument("--max-chars", type=int, default=6000, help="单次整合的输入字数上限")
    p.add_argument("--slides", help="课件 PDF/PPTX：给整合阶段当参考，并把图插进笔记")
    p.add_argument("--glossary", help="课程术语表：glossaries/ 下的名字（不带 .txt）或路径")
    p.set_defaults(fn=cmd_notes)

    args = ap.parse_args(argv)
    config.load_dotenv()
    cfg = config.load(args.config)
    try:
        return args.fn(args, cfg)
    except (LlmError, FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"\n错误：{e}", file=sys.stderr)
        return 1
