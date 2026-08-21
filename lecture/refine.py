"""逐段清洗：把 ASR 原稿改写成通顺、术语正确、公式已 LaTeX 化的笔记片段。

课上每攒够一批就跑一次（本地模型，隐私且免费），课后也能用 `refine` 子命令补跑。
"""
from __future__ import annotations

import threading
from typing import Callable

from lecture.config import Config
from lecture.llm import Endpoint, LlmError, chat
from lecture.store import Sentence, Session, fmt_ms

SYSTEM = """你是一个理工科课堂笔记整理助手。输入是语音识别得到的课堂原稿片段，可能有错字、口头语、重复和断句错误。

你的任务是把它改写成可读的中文笔记片段，规则：
1. 只做整理，不做总结，不许新增原稿里没有的知识点、例子或结论。信息量必须与原稿一致。
2. 删掉口头语（"啊""这个""就是说""对吧"）、重复的话、与课程内容无关的闲聊（点名、通知作业时间除外，那些用引用块保留）。
3. 修正明显的同音错字，尤其按下方术语表纠正专业名词。
4. 口述的数学式子转成 LaTeX：行内用 $...$，独立成行的用 $$...$$。
   例："x 的平方加 y 的平方等于 r 的平方" → $x^2 + y^2 = r^2$。
   例："对 f 求导" → 对 $f$ 求导。
5. 保持讲课的原有顺序和逻辑，用短段落组织。该分点的地方用 markdown 列表。
6. 老师明确强调的地方（"这个要考""重点""记住"）用 **加粗** 标出。
7. 听不清或识别明显残缺的地方，原样保留并在后面加 `[?]`，不要猜内容。
   **语法通顺但语义讲不通的地方同样要标 `[?]`**——语音识别经常把词听成另一个同音词，
   结果是一句读得下去但没有意义的话。判断标准：这句话放在本节课的语境里说得通吗？
   说不通就标 `[?]`，宁可留个疑问也不要硬编成一个听起来合理的说法。
   这种地方**绝对不要加粗**——加粗等于告诉读者"这是重点"，把乱码标成重点是最糟的结果。
8. 老师交代的课程结构信息必须保留：第几章第几节、这节讲什么主题、"下面看另一个问题"这类
   过渡，用 markdown 三级标题（###）写出来。这是笔记的骨架，不算闲聊，删了就没法组织章节了。
9. 直接输出 markdown 正文，不要写"以下是整理结果"之类的话，不要用代码块包裹。"""


def _user_prompt(sentences: list[Sentence], context: list[Sentence], glossary: str) -> str:
    parts = []
    if glossary:
        parts.append("【课程术语表】（`错=>对` 表示把左边的识别错误纠正为右边）\n" + glossary)
    if context:
        parts.append("【上文，仅供衔接参考，不要重复输出】\n"
                     + "".join(s.text for s in context))
    body = "\n".join(f"[{s.stamp}] {s.text}" for s in sentences)
    parts.append("【待整理的原稿片段】\n" + body)
    return "\n\n".join(parts)


def _batches(sentences: list[Sentence], start: int, batch_seconds: int,
             min_sentences: int, flush: bool) -> list[tuple[int, int]]:
    """把未清洗的句子切成 (起, 止) 区间（止不含）。"""
    out: list[tuple[int, int]] = []
    i = start
    n = len(sentences)
    while i < n:
        j = i
        while j < n:
            span = (sentences[j].end_ms - sentences[i].start_ms) / 1000
            if j - i + 1 >= min_sentences and span >= batch_seconds:
                j += 1
                break
            j += 1
        if j - i < min_sentences and not flush:
            break
        out.append((i, j))
        i = j
    return out


# 同一个 session 不能有两个清洗同时跑。课上的后台 worker、结束时的 flush、
# web 的笔记任务都可能同时调进来；并发时两边读到同一个 refined_until，
# 会处理同一批句子、算出同一个 chunk 编号，一份输出被覆盖而进度照常前进 = 静默丢内容。
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(path), threading.Lock())


def run(session: Session, cfg: Config, ep: Endpoint, flush: bool = False,
        log: Callable[[str], None] = print, glossary_spec: str | None = None) -> int:
    """清洗所有还没处理的句子。返回本次新写的片段数。同一 session 串行执行。"""
    with _lock_for(session.path):
        return _run_locked(session, cfg, ep, flush, log, glossary_spec)


def _run_locked(session: Session, cfg: Config, ep: Endpoint, flush: bool,
                log: Callable[[str], None], glossary_spec: str | None) -> int:
    sentences = session.sentences()
    start = session.refined_until
    batches = _batches(sentences, start, cfg.refine.batch_seconds,
                       cfg.refine.batch_min_sentences, flush)
    if not batches:
        return 0

    glossary = cfg.glossary_text(glossary_spec)
    written = 0
    chunk_no = len(session.note_chunks())
    for first, last in batches:
        group = sentences[first:last]
        ctx_from = max(0, first - cfg.refine.context_sentences)
        context = sentences[ctx_from:first]
        try:
            body, _ = chat(ep, SYSTEM, _user_prompt(group, context, glossary))
        except LlmError as e:
            log(f"[清洗失败] 句 {first}-{last - 1}：{e}")
            break
        if not body:
            log(f"[清洗为空] 句 {first}-{last - 1}，跳过")
            break
        chunk_no += 1
        session.write_note_chunk(chunk_no, body, first, last - 1,
                                 group[0].start_ms, group[-1].end_ms)
        session.set_refined_until(last)
        written += 1
        log(f"[清洗] {fmt_ms(group[0].start_ms)}-{fmt_ms(group[-1].end_ms)} "
            f"（{last - first} 句）→ notes/{chunk_no:04d}.md")
    return written
