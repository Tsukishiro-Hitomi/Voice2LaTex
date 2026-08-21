"""采集引擎：把音频→原稿→后台清洗这一套跑起来，并线程安全地暴露状态。

终端和 Web 界面都是它的"显示客户端"，暂停和收尾逻辑只有这一份。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from lecture import config, refine
from lecture.audio import open_source
from lecture.llm import (Endpoint, LlmError, available, cloud_endpoint, endpoint,
                         is_local)
from lecture.store import Session, fmt_ms


def pick_refine_endpoint(cfg: config.Config, log) -> Endpoint | None:
    """选清洗后端。云端就直接用（探活只为验 key），本地不通才谈退回。"""
    try:
        ep = endpoint(cfg.llm, "refine")
    except LlmError as e:
        log(f"[清洗] 不可用：{e}")
        return None

    if not is_local(ep):
        if not available(ep):
            log(f"[清洗] {ep} 探活失败（key 不对？网络不通？）仍会尝试，"
                f"不行就下课后跑 refine 补")
        return ep

    if available(ep):
        return ep
    log(f"[清洗] {ep} 连不上（ollama 没启动？模型没拉？）")
    if not cfg.llm.refine_fallback_to_cloud:
        log("[清洗] 已关闭云端回退，课上不做清洗；下课后跑 refine 补")
        return None
    try:
        cloud = cloud_endpoint(cfg.llm)
    except LlmError as e:
        log(f"[清洗] 云端也不可用：{e}；课上只转写，下课后跑 refine 补")
        return None
    log(f"[清洗] 回退到云端 {cloud}")
    return cloud


class _RefineWorker(threading.Thread):
    """课上每隔一会儿把攒够的句子清洗一批。失败就退场，交给课后补跑。"""

    def __init__(self, session: Session, cfg: config.Config, ep: Endpoint,
                 log, glossary: str | None, interval: float = 30.0):
        super().__init__(daemon=True)
        self.session, self.cfg, self.ep, self.log = session, cfg, ep, log
        self.glossary = glossary
        self.interval = interval
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            self._tick(False)

    def finish(self) -> None:
        self._stop.set()
        self._tick(True)

    def _tick(self, flush: bool) -> None:
        try:
            refine.run(self.session, self.cfg, self.ep, flush=flush, log=self.log,
                       glossary_spec=self.glossary)
        except Exception as e:                      # 清洗不该拖垮转写
            self.log(f"[清洗异常] {type(e).__name__}: {e}")


class Recorder:
    """一次采集。start() 起后台线程，状态通过 snapshot() 读。"""

    def __init__(self, cfg: config.Config, *, title: str, source: str = "mic",
                 device: int | None = None, file: str | None = None,
                 glossary: str | None = None, live_refine: bool = True):
        self.cfg = cfg
        self.title, self.source, self.file = title, source, file
        # 没显式指定就用配置里的，见 AsrConfig.device
        self.device = device if device is not None else cfg.asr.device
        self.glossary, self.live_refine = glossary, live_refine

        self._lock = threading.Lock()
        self._sentences: list[dict[str, Any]] = []
        self._log: list[str] = []
        self._partial = ""
        self._elapsed_ms = 0
        self._paused = False
        self._stop_req = threading.Event()
        self._state = "idle"        # idle | loading | running | finishing | done | error
        self._error: str | None = None
        self._thread: threading.Thread | None = None
        self.session: Session | None = None

    # ---------- 对外状态 ----------

    def log(self, msg: str) -> None:
        with self._lock:
            self._log.append(msg)

    def snapshot(self, since: int = 0, log_since: int = 0) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "paused": self._paused,
                "partial": self._partial,
                "elapsed": fmt_ms(self._elapsed_ms),
                "n": len(self._sentences),
                "sentences": self._sentences[since:],
                "log": self._log[log_since:],
                "n_log": len(self._log),
                "session": str(self.session.path) if self.session else None,
                "title": self.title,
                "error": self._error,
            }

    @property
    def alive(self) -> bool:
        return self._state in ("loading", "running", "finishing")

    # ---------- 控制 ----------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("这个 Recorder 已经跑过了，请新建一个")
        self._state = "loading"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def toggle_pause(self) -> bool:
        """返回切换后的暂停状态。"""
        with self._lock:
            self._paused = not self._paused
            return self._paused

    def request_stop(self) -> None:
        self._stop_req.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    # ---------- 主循环 ----------

    def _run(self) -> None:
        from lecture.asr import Transcriber      # 延迟导入，加载模型要一会儿

        try:
            t0 = time.time()
            self.log("加载模型…")
            tr = Transcriber(self.cfg.asr)
            self.log(f"模型就绪（{time.time() - t0:.1f}s）")

            origin = f"file:{Path(self.file).resolve()}" if self.file else self.source
            session = Session.create(self.cfg.sessions_dir, self.title, origin)
            self.session = session
            if self.source != "file":
                session.open_audio()
            if self.glossary:
                session.update_meta(glossary=self.glossary)
                self.log(f"术语表：{self.cfg.resolve_glossary(self.glossary).name}")
            self.log(f"session: {session.path.name}")

            worker = None
            if self.live_refine:
                ep = pick_refine_endpoint(self.cfg, self.log)
                if ep:
                    worker = _RefineWorker(session, self.cfg, ep, self.log, self.glossary)
                    worker.start()
                    self.log(f"[清洗] 后台运行中（{ep}）")

            with self._lock:
                self._state = "running"

            stream = open_source(self.source, device=self.device, file=self.file,
                                 stop=self._stop_req)
            was_paused = False
            is_file = self.source == "file"
            it = iter(stream)
            try:
                while not self._stop_req.is_set():
                    with self._lock:
                        paused = self._paused
                    # 暂停是别的线程切的，边界处理只能在这里做
                    if paused and not was_paused:
                        # 把说了一半的那句收掉，否则恢复后会和课间之后的话接成一句
                        self._absorb(session, tr.pause(), "", tr.elapsed_ms)
                        session.mark_pause(tr.elapsed_ms)
                        self.log(f"[暂停 {fmt_ms(tr.elapsed_ms)}] 不录音也不识别")
                    elif was_paused and not paused:
                        self.log(f"[继续 {fmt_ms(tr.elapsed_ms)}]")
                    was_paused = paused

                    if paused and is_file:
                        # 文件源暂停要停住不读。否则 ffmpeg 照常解码、块被丢弃，
                        # 暂停几秒就等于跳过好几分钟录音。
                        time.sleep(0.1)
                        continue
                    try:
                        chunk = next(it)
                    except StopIteration:
                        break
                    if paused:
                        continue        # 实时源：那段时间真的过去了，只能丢
                    session.append_audio(chunk)
                    partial, finals = tr.feed(chunk)
                    self._absorb(session, finals, partial, tr.elapsed_ms)
            finally:
                with self._lock:
                    self._state = "finishing"
                self._stop_req.set()
                stream.close()
                self._finalize(session, tr, worker)
        except Exception as e:                  # noqa: BLE001 - 任何失败都要让 UI 看到
            with self._lock:
                self._state = "error"
                self._error = f"{type(e).__name__}: {e}"
            self.log(f"[错误] {self._error}")

    def _absorb(self, session: Session, finals, partial: str, elapsed_ms: int) -> None:
        rows = []
        for f in finals:
            s = session.append_sentence(f.start_ms, f.end_ms, f.text, f.stream_text)
            rows.append({"i": s.i, "t": s.stamp, "text": s.text})
        with self._lock:
            self._sentences.extend(rows)
            self._partial = partial
            self._elapsed_ms = elapsed_ms

    def _finalize(self, session: Session, tr, worker) -> None:
        # 尾段识别放守护线程 + 超时：万一 ASR 或音频后端卡住，也不能收不了尾。
        # 最坏只丢最后一句（音频仍在 audio.wav 里），原稿和 raw.md 一定落盘。
        tail: list = []
        flusher = threading.Thread(target=lambda: tail.extend(tr.flush()), daemon=True)
        flusher.start()
        flusher.join(timeout=20.0)
        if flusher.is_alive():
            self.log("[收尾] 尾段识别超时，已跳过（音频仍完整保存在 audio.wav）")
        self._absorb(session, tail, "", tr.elapsed_ms)
        session.close()
        raw = session.write_raw_md()
        self.log(f"原稿已写入 {raw.name}")
        if worker:
            self.log("[清洗] 正在处理最后一批…")
            worker.finish()
            self.log("[清洗] 完成")
        with self._lock:
            self._state = "done"
