"""本地 Web 界面：点按钮开始/暂停，实时字幕显示在页面上。

  python -m lecture serve          然后浏览器打开 http://127.0.0.1:8730

前端只是引擎（lecture/engine.py）的显示客户端，和终端那套共用同一份采集逻辑。
状态用轮询拿，不用 WebSocket——字幕每秒才更新一两次，轮询足够且没有重连问题。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from lecture import config, latex, models, refine
from lecture.audio import IS_WINDOWS
from lecture.compile import compile_tex
from lecture.engine import Recorder, pick_refine_endpoint
from lecture.llm import cloud_endpoint
from lecture.store import Session

STATIC = Path(__file__).resolve().parent / "static"


class StartReq(BaseModel):
    title: str = "课堂"
    source: str = "mic"
    device: int | None = None
    glossary: str | None = None
    file: str | None = None      # source=file 时的录音路径


class VideoReq(BaseModel):
    file: str
    title: str | None = None
    glossary: str | None = None
    slides: str | None = None


class NotesReq(BaseModel):
    session: str | None = None
    slides: str | None = None
    glossary: str | None = None


class NotesJob(threading.Thread):
    """课后生成笔记。跑在后台线程，前端轮询进度。"""

    def __init__(self, cfg: config.Config, session_path: str,
                 slides: str | None, glossary: str | None):
        super().__init__(daemon=True)
        self.cfg, self.session_path = cfg, session_path
        self.slides, self.glossary = slides, glossary
        self._lock = threading.Lock()
        self._log: list[str] = []
        self.state = "running"
        self.result: dict[str, str] = {}

    def log(self, msg: str) -> None:
        with self._lock:
            self._log.append(str(msg))

    def snapshot(self, log_since: int = 0) -> dict[str, Any]:
        with self._lock:
            return {"state": self.state, "log": self._log[log_since:],
                    "n_log": len(self._log), "result": self.result}

    def run(self) -> None:
        try:
            self.result = generate_notes(self.cfg, self.session_path, self.slides,
                                         self.glossary, self.log)
            self.log("完成")
            self.state = "done"
        except Exception as e:                      # noqa: BLE001
            self.log(f"[错误] {type(e).__name__}: {e}")
            self.state = "error"



def generate_notes(cfg: config.Config, session_path: str, slides: str | None,
                   glossary: str | None, log) -> dict[str, str]:
    """补跑清洗 + 抽课件 + 整合成 LaTeX。NotesJob 和 VideoJob 共用这一份。"""
    session = Session.open(session_path)
    spec = glossary or session.meta.get("glossary")
    deck = None
    if slides:
        from lecture import slides as slides_mod
        deck = slides_mod.extract(slides, session.path)
        session.update_meta(slides=str(Path(slides).resolve()))
        log(f"课件 {len(deck.slides)} 页，抽出 {len(deck.figures)} 张图")

    todo = len(session.sentences()) - session.refined_until
    if todo > 0:
        log(f"还有 {todo} 句没清洗，先补跑…")
        ep = pick_refine_endpoint(cfg, log)
        if ep is None:
            raise RuntimeError("没有可用的清洗后端")
        refine.run(session, cfg, ep, flush=True, log=log, glossary_spec=spec)

    tex, md = latex.compose(session, cfg, cloud_endpoint(cfg.llm), deck=deck, log=log)

    # 真的编一遍。编不出来不算失败——没装 TeX 就只给 .tex，
    # 装 TeX 是 GB 级的事，不能因为没装就让整条链断掉
    built = compile_tex(tex, log=log)
    if not built.ok:
        log(f"[编译] 跳过：{built.reason}")
        if built.log_tail:
            log(built.log_tail[:600])

    return {"tex": tex, "md": md, "raw": str(session.path / "raw.md"),
            "dir": str(session.path),
            "pdf": str(built.pdf) if built.ok else "",
            "pdf_skipped": "" if built.ok else built.reason,
            "compile": f"cd {session.path} && xelatex notes.tex"}


class VideoJob(threading.Thread):
    """课程视频 → 笔记，一步到底。

    服务端串起来跑，而不是让前端"等转写完再发第二个请求"：
    浏览器关了、网断了，任务照样跑完。
    """

    def __init__(self, cfg: config.Config, *, file: str, title: str,
                 glossary: str | None, slides: str | None):
        super().__init__(daemon=True)
        self.cfg, self.file, self.title = cfg, file, title
        self.glossary, self.slides = glossary, slides
        self._lock = threading.Lock()
        self._log: list[str] = []
        self.state = "running"
        self.phase = "转写"
        self.result: dict[str, str] = {}
        self.rec: Recorder | None = None

    def log(self, msg: str) -> None:
        with self._lock:
            self._log.append(str(msg))

    def snapshot(self, log_since: int = 0) -> dict[str, Any]:
        with self._lock:
            log = self._log[log_since:]
            n_log = len(self._log)
        rec = self.rec
        snap = rec.snapshot() if rec else None
        return {"state": self.state, "phase": self.phase, "log": log, "n_log": n_log,
                "result": self.result, "title": self.title,
                "n": snap["n"] if snap else 0,
                "elapsed": snap["elapsed"] if snap else "00:00",
                "session": snap["session"] if snap else None}

    def run(self) -> None:
        try:
            self.log(f"视频：{Path(self.file).name}")
            rec = Recorder(self.cfg, title=self.title, source="file", file=self.file,
                           glossary=self.glossary, live_refine=False)
            self.rec = rec
            rec.start()
            seen = 0
            while rec.alive:
                snap = rec.snapshot(log_since=seen)
                for line in snap["log"]:
                    self.log(line)
                seen = snap["n_log"]
                time.sleep(0.4)
            snap = rec.snapshot(log_since=seen)
            for line in snap["log"]:
                self.log(line)
            if snap["error"]:
                raise RuntimeError(snap["error"])
            if rec.session is None:
                raise RuntimeError("转写没有产出 session")
            self.log(f"转写完成，共 {snap['n']} 句")

            self.phase = "生成笔记"
            self.result = generate_notes(self.cfg, str(rec.session.path), self.slides,
                                         self.glossary, self.log)
            self.log("完成")
            self.phase = "完成"
            self.state = "done"
        except Exception as e:                      # noqa: BLE001
            self.log(f"[错误] {type(e).__name__}: {e}")
            self.phase = "失败"
            self.state = "error"


def create_app(cfg: config.Config) -> FastAPI:
    app = FastAPI(title="上课语音助手")
    cur: dict[str, Any] = {"rec": None, "notes": None, "video": None}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/init")
    def init() -> dict[str, Any]:
        import sounddevice as sd
        devs = []
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 or d["max_output_channels"] > 0:
                devs.append({"i": i, "name": d["name"],
                             "input": d["max_input_channels"] > 0,
                             "output": d["max_output_channels"] > 0})
        return {
            "devices": devs,
            "default_input": sd.default.device[0],
            "default_output": sd.default.device[1],
            "glossaries": sorted(f.stem for f in cfg.glossaries_dir.glob("*.txt")),
            "loopback_ok": IS_WINDOWS,
            "sessions": [d.name for d in sorted(cfg.sessions_dir.iterdir(), reverse=True)
                         if d.is_dir() and (d / "meta.json").exists()][:20],
        }

    @app.post("/api/start")
    def start(req: StartReq) -> dict[str, Any]:
        rec: Recorder | None = cur["rec"]
        if rec is not None and rec.alive:
            raise HTTPException(409, "已经在录了，先结束当前这次")
        # 模型没下就别让它跑到加载那一步再抛 FileNotFoundError——那个错
        # 到界面上是一句看不懂的路径，而这里能直接告诉用户该去下
        lack = models.missing(cfg.asr)
        if lack:
            raise HTTPException(
                503, "模型还没下：" + "、".join(it.desc for it in lack)
                + "。先跑 python -m lecture fetch")
        if req.source == "file":
            if not req.file or not Path(req.file).expanduser().exists():
                raise HTTPException(400, f"找不到录音文件：{req.file}")
            req.file = str(Path(req.file).expanduser())
        rec = Recorder(cfg, title=req.title or "课堂", source=req.source,
                       device=req.device, file=req.file, glossary=req.glossary,
                       live_refine=True)
        cur["rec"] = rec
        rec.start()
        return {"ok": True}

    @app.post("/api/pause")
    def pause() -> dict[str, Any]:
        rec: Recorder | None = cur["rec"]
        if rec is None or not rec.alive:
            raise HTTPException(409, "现在没有在录")
        return {"paused": rec.toggle_pause()}

    @app.post("/api/stop")
    def stop() -> dict[str, Any]:
        rec: Recorder | None = cur["rec"]
        if rec is None:
            raise HTTPException(409, "现在没有在录")
        rec.request_stop()
        return {"ok": True}

    @app.get("/api/state")
    def state(since: int = 0, log_since: int = 0) -> dict[str, Any]:
        rec: Recorder | None = cur["rec"]
        if rec is None:
            return {"state": "idle", "sentences": [], "log": [], "n": 0,
                    "n_log": 0, "partial": "", "elapsed": "00:00",
                    "paused": False, "session": None, "error": None}
        return rec.snapshot(since, log_since)

    @app.post("/api/notes")
    def notes(req: NotesReq) -> dict[str, Any]:
        job: NotesJob | None = cur["notes"]
        if job is not None and job.state == "running":
            raise HTTPException(409, "笔记正在生成中")
        target = req.session
        if not target:
            rec: Recorder | None = cur["rec"]
            if rec is None or rec.session is None:
                raise HTTPException(400, "没有可用的 session")
            target = str(rec.session.path)
        elif not Path(target).is_absolute():
            target = str(cfg.sessions_dir / target)
        rec = cur["rec"]
        if rec is not None and rec.alive and rec.session is not None \
                and str(rec.session.path) == str(Path(target)):
            raise HTTPException(409, "这节课还在录，先点「结束」再生成笔记")
        job = NotesJob(cfg, target, req.slides or None, req.glossary or None)
        cur["notes"] = job
        job.start()
        return {"ok": True, "session": target}

    @app.post("/api/video")
    def video(req: VideoReq) -> dict[str, Any]:
        rec: Recorder | None = cur["rec"]
        if rec is not None and rec.alive:
            raise HTTPException(409, "正在录课，先结束再处理视频（会抢同一个 ASR 模型）")
        job: VideoJob | None = cur["video"]
        if job is not None and job.state == "running":
            raise HTTPException(409, "已经有一个视频在处理中")
        path = Path(req.file).expanduser()
        if not path.exists():
            raise HTTPException(400, f"找不到文件：{req.file}")
        job = VideoJob(cfg, file=str(path), title=req.title or path.stem,
                       glossary=req.glossary, slides=req.slides or None)
        cur["video"] = job
        job.start()
        return {"ok": True}

    @app.get("/api/video_state")
    def video_state(log_since: int = 0) -> dict[str, Any]:
        job: VideoJob | None = cur["video"]
        if job is None:
            return {"state": "idle", "phase": "", "log": [], "n_log": 0,
                    "result": {}, "n": 0, "elapsed": "00:00", "session": None}
        return job.snapshot(log_since)

    @app.get("/api/notes_state")
    def notes_state(log_since: int = 0) -> dict[str, Any]:
        job: NotesJob | None = cur["notes"]
        if job is None:
            return {"state": "idle", "log": [], "n_log": 0, "result": {}}
        return job.snapshot(log_since)

    return app


def serve(cfg: config.Config, host: str = "127.0.0.1", port: int = 8730) -> None:
    import uvicorn
    print(f"界面地址： http://{host}:{port}\n（Ctrl-C 停止）")
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")
