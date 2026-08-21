"""双通道 ASR（2pass）：

- 流式通道：Paraformer streaming，边说边出字，只用来显示实时字幕。
- 定稿通道：VAD 切出完整句子后，交给 SenseVoice 重新识别一遍，作为原稿。

这样既有低延迟的观感，又不把流式识别的错误写进原稿。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx

from lecture.config import AsrConfig

SAMPLE_RATE = 16000


def _pick(model_dir: Path, *stems: str) -> str:
    """优先用 int8 量化版（CPU 上快一截），退回 fp32。"""
    for stem in stems:
        for name in (f"{stem}.int8.onnx", f"{stem}.onnx"):
            p = model_dir / name
            if p.exists():
                return str(p)
    raise FileNotFoundError(f"{model_dir} 里找不到 {stems} 对应的 onnx 文件")


@dataclass
class Final:
    """一句定稿。"""
    start_ms: int
    end_ms: int
    text: str
    stream_text: str


class Transcriber:
    def __init__(self, cfg: AsrConfig):
        missing = [p for p in (cfg.streaming_dir, cfg.offline_dir, cfg.vad_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "缺少模型：\n  " + "\n  ".join(str(m) for m in missing)
                + "\n先跑：python scripts/fetch_models.py")

        self.online = sherpa_onnx.OnlineRecognizer.from_paraformer(
            tokens=str(cfg.streaming_dir / "tokens.txt"),
            encoder=_pick(cfg.streaming_dir, "encoder"),
            decoder=_pick(cfg.streaming_dir, "decoder"),
            num_threads=cfg.num_threads,
            provider=cfg.provider,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
        )
        self.offline = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=_pick(cfg.offline_dir, "model"),
            tokens=str(cfg.offline_dir / "tokens.txt"),
            num_threads=cfg.num_threads,
            provider=cfg.provider,
            language="auto",     # 中英混说交给模型自己判
            use_itn=True,        # 数字/标点规整化
        )

        vad_cfg = sherpa_onnx.VadModelConfig()
        vad_cfg.silero_vad.model = str(cfg.vad_path)
        vad_cfg.silero_vad.min_silence_duration = cfg.min_silence_duration
        vad_cfg.silero_vad.min_speech_duration = cfg.min_speech_duration
        vad_cfg.silero_vad.max_speech_duration = cfg.max_speech_duration
        vad_cfg.sample_rate = SAMPLE_RATE
        vad_cfg.num_threads = 1
        self.vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=60)

        self._stream = self.online.create_stream()
        self._partial = ""
        self._samples_seen = 0

    @property
    def elapsed_ms(self) -> int:
        """已喂进来的音频时长，等于 audio.wav 里的当前位置。"""
        return int(self._samples_seen / SAMPLE_RATE * 1000)

    # ---------- 主入口 ----------

    def feed(self, chunk: np.ndarray) -> tuple[str, list[Final]]:
        """喂一块音频，返回 (当前实时字幕, 本次切出的定稿句列表)。"""
        self._samples_seen += len(chunk)

        # 通道 1：流式识别（只为显示）
        self._stream.accept_waveform(SAMPLE_RATE, chunk)
        while self.online.is_ready(self._stream):
            self.online.decode_stream(self._stream)
        self._partial = self.online.get_result(self._stream)

        # 通道 2：VAD 切句 → 整句重识别
        self.vad.accept_waveform(chunk)
        finals: list[Final] = []
        while not self.vad.empty():
            start, samples = self._take_segment()
            text = self._recognize(samples)
            start_ms = int(start / SAMPLE_RATE * 1000)
            end_ms = start_ms + int(len(samples) / SAMPLE_RATE * 1000)
            if text:
                finals.append(Final(start_ms, end_ms, text, self._partial))
            # 一句结束就重置流式解码器，避免它无限累积
            self.online.reset(self._stream)
            self._partial = ""

        return self._partial, finals

    def pause(self) -> list[Final]:
        """暂停时收尾：把 VAD 缓冲里已经说了一半的话识别掉，并重置流式解码器。

        不做这一步的话，恢复后 VAD 会把暂停前后的音频接成一句，中间的时间全被算进去。
        """
        finals = self.flush()
        self.online.reset(self._stream)
        self._partial = ""
        return finals

    def flush(self) -> list[Final]:
        """收尾：把 VAD 缓冲里剩下的语音也识别掉。"""
        self.vad.flush()
        finals: list[Final] = []
        while not self.vad.empty():
            start, samples = self._take_segment()
            text = self._recognize(samples)
            if text:
                start_ms = int(start / SAMPLE_RATE * 1000)
                end_ms = start_ms + int(len(samples) / SAMPLE_RATE * 1000)
                finals.append(Final(start_ms, end_ms, text, ""))
        return finals

    def _take_segment(self) -> tuple[int, np.ndarray]:
        """取一段语音。必须在 pop 之前把数据拷出来——front 给的是内部缓冲的引用，
        pop 之后 samples 就空了（这个坑很安静，只表现为识别结果全空）。"""
        seg = self.vad.front
        start = seg.start
        samples = np.array(seg.samples, dtype=np.float32)
        self.vad.pop()
        return start, samples

    def _recognize(self, samples: np.ndarray) -> str:
        s = self.offline.create_stream()
        s.accept_waveform(SAMPLE_RATE, samples)
        self.offline.decode_stream(s)
        return s.result.text.strip()
