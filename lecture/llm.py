"""统一的 OpenAI 兼容客户端：本地 Ollama 和云端 DeepSeek 走同一套调用。"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from openai import OpenAI

from lecture.config import LlmConfig

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class LlmError(RuntimeError):
    pass


@dataclass
class Endpoint:
    backend: str
    model: str
    base_url: str
    api_key: str
    thinking: bool = True   # DeepSeek V4 默认开思考，关掉能省一大半输出 token

    def __str__(self) -> str:
        return f"{self.backend}:{self.model}"


def endpoint(cfg: LlmConfig, role: str) -> Endpoint:
    """role: refine（课上逐段清洗）| compose（课后整合 LaTeX）"""
    backend = cfg.refine_backend if role == "refine" else cfg.compose_backend
    model = cfg.refine_model if role == "refine" else cfg.compose_model
    return _for_backend(cfg, backend, model)


def cloud_endpoint(cfg: LlmConfig, model: str | None = None) -> Endpoint:
    return _for_backend(cfg, cfg.compose_backend, model or cfg.compose_model)


def _for_backend(cfg: LlmConfig, backend: str, model: str) -> Endpoint:
    if backend == "ollama":
        return Endpoint(backend, model, cfg.ollama_base_url, "ollama")
    if backend == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            raise LlmError("缺少 DEEPSEEK_API_KEY：把 .env.example 复制成 .env 并填入 key")
        return Endpoint(backend, model, cfg.deepseek_base_url, key, cfg.thinking)
    raise LlmError(f"未知 LLM 后端：{backend}（支持 ollama / deepseek）")


def chat(ep: Endpoint, system: str, user: str, temperature: float = 0.2,
         timeout: float = 300.0, max_tokens: int | None = None) -> tuple[str, str]:
    """返回 (正文, finish_reason)。qwen3 之类的思考块会被剥掉。"""
    client = OpenAI(base_url=ep.base_url, api_key=ep.api_key, timeout=timeout, max_retries=2)
    kwargs: dict = {}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if ep.backend == "ollama":
        # qwen3 系列默认开思考，关掉能省一半时间
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    elif ep.backend == "deepseek" and not ep.thinking:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    try:
        resp = client.chat.completions.create(
            model=ep.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature,
            **kwargs,
        )
    except Exception as e:
        raise LlmError(f"{ep} 调用失败：{type(e).__name__}: {e}") from e
    choice = resp.choices[0]
    text = _THINK.sub("", choice.message.content or "").strip()
    return text, (choice.finish_reason or "")


def is_local(ep: Endpoint) -> bool:
    """只有本地后端才需要"连不上就退回云端"那套逻辑。"""
    return ep.backend == "ollama"


def available(ep: Endpoint) -> bool:
    """探活。ollama 要连模型是否已拉一起查——服务起着但模型没拉会在调用时才 404，
    那时候一节课已经开始了，降级已经来不及。"""
    try:
        listed = OpenAI(base_url=ep.base_url, api_key=ep.api_key, timeout=5.0,
                        max_retries=0).models.list()
    except Exception:
        return False
    if ep.backend != "ollama":
        return True                       # 云端有 deepseek-chat 这类别名，不在列表里也能用
    names = {m.id for m in (listed.data or [])}   # 没拉过任何模型时 data 是 None
    return ep.model in names or f"{ep.model}:latest" in names
