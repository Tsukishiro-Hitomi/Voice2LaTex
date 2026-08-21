from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


# config.yaml / .env / models 的落脚处：仓库根目录。
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AsrConfig:
    models_dir: Path
    streaming_model: str
    offline_model: str
    vad_model: str
    provider: str
    num_threads: int
    min_silence_duration: float
    min_speech_duration: float
    max_speech_duration: float
    # 默认录音设备。None = 用系统默认，但系统默认不总是能用——实测有的机器上
    # 录不到东西，会安静地录一整节课空白。可以写编号，也可以写设备名的一段
    # （sounddevice 支持模糊匹配）；写名字更稳，编号会随插拔耳机而变。
    device: int | str | None = None

    @property
    def streaming_dir(self) -> Path:
        return self.models_dir / self.streaming_model

    @property
    def offline_dir(self) -> Path:
        return self.models_dir / self.offline_model

    @property
    def vad_path(self) -> Path:
        return self.models_dir / self.vad_model


@dataclass
class LlmConfig:
    refine_backend: str
    refine_model: str
    ollama_base_url: str
    compose_backend: str
    compose_model: str
    deepseek_base_url: str
    refine_fallback_to_cloud: bool
    thinking: bool


@dataclass
class RefineConfig:
    batch_seconds: int
    batch_min_sentences: int
    context_sentences: int


@dataclass
class Config:
    sessions_dir: Path
    glossary: Path
    glossaries_dir: Path
    asr: AsrConfig
    llm: LlmConfig
    refine: RefineConfig

    def resolve_glossary(self, spec: str | None) -> Path:
        """spec 可以是路径，也可以是 glossaries/ 下的课程名（不带 .txt）。"""
        if not spec:
            return self.glossary
        p = Path(spec).expanduser()
        if p.exists():
            return p
        named = self.glossaries_dir / f"{spec}.txt"
        if named.exists():
            return named
        raise FileNotFoundError(
            f"找不到术语表 {spec!r}。可用的：" +
            ", ".join(sorted(f.stem for f in self.glossaries_dir.glob("*.txt")) or ["(无)"]))

    def glossary_text(self, spec: str | None = None) -> str:
        """术语表内容，注释行和空行已剔除。"""
        path = self.resolve_glossary(spec)
        if not path.exists():
            return ""
        lines = [
            ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        return "\n".join(lines)


def _device(raw) -> int | str | None:
    """config.yaml 里的 device。数字按编号用，字符串按设备名匹配，空就是系统默认。

    `device: 1` 会被 YAML 读成 int，`device: "Realtek"` 是 str，两种都直接能喂给
    sounddevice。写成 `"1"` 这种带引号的数字也当编号——那多半是手误，而按名字
    去找一个叫「1」的设备只会静默失败。
    """
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):       # YAML 的 yes/no 会变成 bool，那肯定是写错了
        return None
    if isinstance(raw, int):
        return raw
    text = str(raw).strip()
    return int(text) if text.lstrip("-").isdigit() else text


def _abs(root: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else root / p


def load(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    root = cfg_path.resolve().parent

    a, l, r = raw["asr"], raw["llm"], raw["refine"]

    # 录音落在哪可以用环境变量覆盖，优先于 config.yaml。
    sessions = os.environ.get("LECTURE_SESSIONS_DIR")

    return Config(
        sessions_dir=Path(sessions).expanduser() if sessions
        else _abs(root, raw["sessions_dir"]),
        glossary=_abs(root, raw["glossary"]),
        glossaries_dir=_abs(root, raw.get("glossaries_dir", "glossaries")),
        asr=AsrConfig(
            models_dir=_abs(root, a["models_dir"]),
            streaming_model=a["streaming_model"],
            offline_model=a["offline_model"],
            vad_model=a["vad_model"],
            provider=a.get("provider", "cpu"),
            num_threads=int(a.get("num_threads", 4)),
            min_silence_duration=float(a.get("min_silence_duration", 0.35)),
            min_speech_duration=float(a.get("min_speech_duration", 0.25)),
            max_speech_duration=float(a.get("max_speech_duration", 25.0)),
            device=_device(a.get("device")),
        ),
        llm=LlmConfig(
            refine_backend=l["refine_backend"],
            refine_model=l["refine_model"],
            ollama_base_url=l["ollama_base_url"],
            compose_backend=l["compose_backend"],
            compose_model=l["compose_model"],
            deepseek_base_url=l["deepseek_base_url"],
            refine_fallback_to_cloud=bool(l.get("refine_fallback_to_cloud", True)),
            thinking=bool(l.get("thinking", True)),
        ),
        refine=RefineConfig(
            batch_seconds=int(r.get("batch_seconds", 120)),
            batch_min_sentences=int(r.get("batch_min_sentences", 6)),
            context_sentences=int(r.get("context_sentences", 3)),
        ),
    )


def load_dotenv() -> None:
    """把仓库根目录的 .env 读进环境变量（不覆盖已有值），省一个依赖。"""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
