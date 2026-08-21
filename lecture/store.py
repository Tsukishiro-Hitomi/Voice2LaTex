"""Session 目录的读写。原稿 append-only，随时崩溃都能保住已转写内容。

sessions/2026-08-20_1403_线性代数/
    meta.json      标题、创建时间、音频源
    raw.jsonl      每句一行，append-only（原稿的权威来源）
    raw.md         人可读原稿，带时间戳
    audio.wav      16k 单声道录音
    notes/0001.md  逐段清洗结果
    notes.tex      课后整合的 LaTeX 笔记
    state.json     已清洗到第几句（断点续跑用）
"""
from __future__ import annotations

import json
import re
import wave
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


@dataclass
class Sentence:
    i: int
    start_ms: int
    end_ms: int
    text: str          # 定稿文本（offline 模型识别）
    stream_text: str   # 流式文本，留档以便对比两遍识别差异

    @property
    def stamp(self) -> str:
        return fmt_ms(self.start_ms)


def fmt_ms(ms: int) -> str:
    total = ms // 1000
    return f"{total // 60:02d}:{total % 60:02d}"


def _slugify(title: str) -> str:
    """保留中英数字，其他压成下划线（Windows 文件名安全）。"""
    s = re.sub(r'[\\/:*?"<>|\s]+', "_", title.strip())
    return s.strip("_")[:60] or "untitled"


class Session:
    def __init__(self, path: Path):
        self.path = path
        self.notes_dir = path / "notes"
        self._raw_jsonl = path / "raw.jsonl"
        self._wav: wave.Wave_write | None = None
        self._n_sentences = len(self.sentences())

    # ---------- 创建 / 打开 ----------

    @classmethod
    def create(cls, sessions_dir: Path, title: str, source: str) -> "Session":
        now = datetime.now()
        path = sessions_dir / f"{now:%Y-%m-%d_%H%M}_{_slugify(title)}"
        path.mkdir(parents=True, exist_ok=True)
        (path / "notes").mkdir(exist_ok=True)
        (path / "meta.json").write_text(
            json.dumps({"title": title, "created": now.isoformat(timespec="seconds"),
                        "source": source}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        return cls(path)

    @classmethod
    def open(cls, path: str | Path) -> "Session":
        p = Path(path)
        if not (p / "raw.jsonl").exists() and not (p / "meta.json").exists():
            raise FileNotFoundError(f"不像是一个 session 目录：{p}")
        (p / "notes").mkdir(exist_ok=True)
        return cls(p)

    @classmethod
    def latest(cls, sessions_dir: Path) -> "Session":
        dirs = [d for d in sessions_dir.iterdir() if d.is_dir() and (d / "meta.json").exists()]
        if not dirs:
            raise FileNotFoundError(f"{sessions_dir} 下还没有任何 session")
        return cls.open(max(dirs, key=lambda d: d.name))

    @property
    def meta(self) -> dict:
        f = self.path / "meta.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}

    def update_meta(self, **kv) -> None:
        m = self.meta
        m.update(kv)
        (self.path / "meta.json").write_text(
            json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

    @property
    def title(self) -> str:
        return self.meta.get("title", self.path.name)

    # ---------- 原稿 ----------

    def next_index(self) -> int:
        return self._n_sentences

    def append_sentence(self, start_ms: int, end_ms: int, text: str, stream_text: str = "") -> Sentence:
        s = Sentence(self._n_sentences, start_ms, end_ms, text, stream_text)
        with self._raw_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
            f.flush()
        self._n_sentences += 1
        return s

    def sentences(self) -> list[Sentence]:
        if not self._raw_jsonl.exists():
            return []
        out = []
        for line in self._raw_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(Sentence(**json.loads(line)))
        return out

    def write_raw_md(self) -> Path:
        """把 jsonl 渲染成人可读原稿。"""
        sents = self.sentences()
        lines = [f"# {self.title} · 原稿", "",
                 f"共 {len(sents)} 句"
                 + (f"，时长 {fmt_ms(sents[-1].end_ms)}" if sents else ""), ""]
        pauses = sorted(self.meta.get("pauses", []))
        for s in sents:
            while pauses and pauses[0] <= s.start_ms:
                lines.append(f"\n**—— 暂停（{fmt_ms(pauses.pop(0))}）——**\n")
            lines.append(f"`[{s.stamp}]` {s.text}")
        out = self.path / "raw.md"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out

    def mark_pause(self, at_ms: int) -> None:
        """记下暂停位置。时间戳是录音里的位置，暂停不占时间，
        所以 raw.md 里若不标出来，课间前后会被无缝接在一起，看不出断点。"""
        m = self.meta
        m.setdefault("pauses", []).append(at_ms)
        (self.path / "meta.json").write_text(
            json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 录音 ----------

    def open_audio(self) -> None:
        w = wave.open(str(self.path / "audio.wav"), "wb")
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        self._wav = w

    def append_audio(self, samples: np.ndarray) -> None:
        if self._wav is None:
            return
        pcm = np.clip(samples, -1.0, 1.0) * 32767.0
        self._wav.writeframes(pcm.astype("<i2").tobytes())

    def close(self) -> None:
        if self._wav is not None:
            self._wav.close()   # wave 在 close 时补写头部长度
            self._wav = None

    # ---------- 清洗进度 / 笔记 ----------

    @property
    def refined_until(self) -> int:
        f = self.path / "state.json"
        if not f.exists():
            return 0
        return int(json.loads(f.read_text(encoding="utf-8")).get("refined_until", 0))

    def set_refined_until(self, n: int) -> None:
        (self.path / "state.json").write_text(
            json.dumps({"refined_until": n}, indent=2), encoding="utf-8")

    def write_note_chunk(self, index: int, body: str, first: int, last: int,
                         start_ms: int, end_ms: int) -> Path:
        out = self.notes_dir / f"{index:04d}.md"
        header = (f"<!-- 原稿句 {first}-{last} | "
                  f"{fmt_ms(start_ms)}-{fmt_ms(end_ms)} -->\n")
        out.write_text(header + body.strip() + "\n", encoding="utf-8")
        return out

    def note_chunks(self) -> list[Path]:
        return sorted(self.notes_dir.glob("*.md"))
