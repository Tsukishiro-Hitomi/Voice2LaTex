"""音频输入：麦克风 / 系统音频(Windows WASAPI loopback) / 文件回放。

统一产出 16kHz 单声道 float32 数据块，供 ASR 消费。
"""
from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterator

import numpy as np
import sounddevice as sd

TARGET_RATE = 16000
CHUNK_SECONDS = 0.1
IS_WINDOWS = sys.platform == "win32"


class Resampler:
    """把任意采样率降到 16k。降采样时先做箱式滤波抗混叠，再线性插值。

    语音识别够用；不追求音乐级质量，为的是不引入 scipy 依赖。
    """

    def __init__(self, src_rate: int, dst_rate: int = TARGET_RATE):
        self.src_rate, self.dst_rate = src_rate, dst_rate
        self.step = src_rate / dst_rate
        self._tail = np.zeros(0, dtype=np.float32)
        self._pos = 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.src_rate == self.dst_rate:
            return x.astype(np.float32, copy=False)
        buf = np.concatenate([self._tail, x.astype(np.float32, copy=False)])
        k = int(self.step)
        smooth = np.convolve(buf, np.ones(k, np.float32) / k, mode="same") if k > 1 else buf

        n = int(np.floor((len(buf) - 1 - self._pos) / self.step)) + 1
        if n <= 0:
            self._tail = buf
            return np.zeros(0, dtype=np.float32)

        idx = self._pos + self.step * np.arange(n)
        out = np.interp(idx, np.arange(len(smooth)), smooth).astype(np.float32)

        next_pos = self._pos + self.step * n
        keep = min(int(next_pos), len(buf))
        self._tail = buf[keep:]
        self._pos = next_pos - keep
        return out


def list_devices() -> str:
    lines = [f"默认输入: {sd.default.device[0]}   默认输出: {sd.default.device[1]}", ""]
    for i, d in enumerate(sd.query_devices()):
        api = sd.query_hostapis(d["hostapi"])["name"]
        tag = []
        if d["max_input_channels"] > 0:
            tag.append(f"in:{d['max_input_channels']}")
        if d["max_output_channels"] > 0:
            tag.append(f"out:{d['max_output_channels']}")
        lines.append(f"  {i:3d}  {d['name'][:45]:45s} {api[:18]:18s} "
                     f"{int(d['default_samplerate'])}Hz  {' '.join(tag)}")
    return "\n".join(lines)


def _loopback_settings(device: int | None) -> tuple[int, dict]:
    """Windows: 用 WASAPI loopback 录「电脑正在放的声音」。返回(设备号, 额外参数)。"""
    if not IS_WINDOWS:
        raise RuntimeError(
            "系统音频采集目前只在 Windows 上实现（WASAPI loopback）。\n"
            "mac 上调试请改用 --source file --file 某个录音，或 --source mic。")
    if device is None:
        device = sd.default.device[1]          # 默认输出设备
    try:
        extra = sd.WasapiSettings(loopback=True)
    except TypeError as e:                     # sounddevice 太老
        raise RuntimeError("当前 sounddevice 版本不支持 loopback，请升级到 >=0.5.0") from e
    return device, {"extra_settings": extra}


def _stream_from_device(device: int | None, extra: dict, stop: threading.Event,
                        rate_hint: float | None) -> Iterator[np.ndarray]:
    info = sd.query_devices(device if device is not None else sd.default.device[0])
    native_rate = int(rate_hint or info["default_samplerate"])

    # loopback 必须跟随渲染设备的原生格式；麦克风则优先直接要 16k。
    if extra:
        rate = native_rate
        channels = max(1, min(2, int(info["max_output_channels"] or 1)))
    else:
        rate = TARGET_RATE
        channels = 1

    q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            print(f"\n[音频] {status}", file=sys.stderr)
        q.put(indata.copy())

    def open_stream(r: int, ch: int):
        return sd.InputStream(samplerate=r, channels=ch, dtype="float32",
                              blocksize=int(r * CHUNK_SECONDS), device=device,
                              callback=callback, **extra)

    try:
        stream = open_stream(rate, channels)
    except Exception:
        if extra or rate == native_rate:
            raise
        rate, channels = native_rate, max(1, min(2, int(info["max_input_channels"] or 1)))
        stream = open_stream(rate, channels)

    resample = Resampler(rate)
    with stream:
        while not stop.is_set():
            try:
                block = q.get(timeout=0.2)
            except queue.Empty:
                continue
            mono = block.mean(axis=1) if block.ndim > 1 and block.shape[1] > 1 else block.reshape(-1)
            out = resample(mono)
            if len(out):
                yield out


def _stream_from_file(path: Path) -> Iterator[np.ndarray]:
    """用 ffmpeg 解码任意音视频为 16k 单声道 float32（跨平台一致）。"""
    # -vn 显式丢掉视频流：录播文件常带 h264，不解码能省不少 CPU
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path),
           "-vn", "-f", "f32le", "-ac", "1", "-ar", str(TARGET_RATE), "-"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as e:
        raise RuntimeError("需要 ffmpeg 来解码音频文件，请先安装并加入 PATH") from e

    nbytes = int(TARGET_RATE * CHUNK_SECONDS) * 4
    assert proc.stdout is not None
    while True:
        buf = proc.stdout.read(nbytes)
        if not buf:
            break
        yield np.frombuffer(buf, dtype=np.float32)
    proc.stdout.close()
    err = proc.stderr.read().decode("utf-8", "ignore") if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg 解码失败：{err.strip()[:400]}")


def open_source(source: str, device: int | None = None, file: str | Path | None = None,
                stop: threading.Event | None = None) -> Iterator[np.ndarray]:
    """source: mic | loopback | file"""
    stop = stop or threading.Event()
    if source == "file":
        if not file:
            raise ValueError("--source file 需要同时给 --file 路径")
        return _stream_from_file(Path(file))
    if source == "loopback":
        dev, extra = _loopback_settings(device)
        return _stream_from_device(dev, extra, stop, None)
    if source == "mic":
        return _stream_from_device(device, {}, stop, None)
    raise ValueError(f"未知音频源：{source}")
