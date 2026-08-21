"""课后整合：把逐段清洗的 markdown 片段合成一份完整的 LaTeX 笔记。

preamble 由代码固定给出（不让模型生成，避免每次 documentclass/宏包漂移），
模型只负责正文：分章节、统一公式写法、保留时间戳锚点。
"""
from __future__ import annotations

import re
from typing import Callable

from lecture.config import Config
from lecture.llm import Endpoint, LlmError, chat
from lecture.store import Session

PREAMBLE = r"""\documentclass[UTF8, 11pt]{ctexart}
\usepackage{amsmath, amssymb, amsthm}
\usepackage[margin=2.4cm]{geometry}
\usepackage{enumitem}
\usepackage{graphicx}
\graphicspath{{./}{figures/}}
\usepackage[colorlinks=true, linkcolor=black, urlcolor=blue]{hyperref}

\newtheorem{theorem}{定理}
\newtheorem{lemma}{引理}
\newtheorem{corollary}{推论}
\newtheorem{proposition}{命题}
\theoremstyle{definition}
\newtheorem{definition}{定义}
\newtheorem{example}{例}
\theoremstyle{remark}
\newtheorem{remark}{注}
\setlist{nosep}
"""

SYSTEM = r"""你把已经初步整理过的课堂笔记片段，合成为规范的 LaTeX 正文。

规则：
1. 只输出正文，不要 \documentclass、不要宏包、不要 \begin{document}/\end{document}、不要 \title/\maketitle。
2. 用 \section{} 和 \subsection{} 按内容组织层级。章节标题要自己概括，不要照抄口语。
3. 数学一律用 LaTeX：行内 $...$，独立公式用 \[ ... \] 或 equation 环境（需要编号时）。
   把 markdown 里的 $$...$$ 改成 \[ ... \]。
4. markdown 语法全部转成 LaTeX：**粗体**→\textbf{}，列表→itemize/enumerate。
   定理类环境**只能用这几个**（preamble 里只定义了这些，用别的会编译失败）：
   theorem 定理、lemma 引理、corollary 推论、proposition 命题、
   definition 定义、example 例、remark 注、proof 证明。
   不许自创新环境，也不许自己写 \newtheorem。
5. 时间戳注释只能照抄片段头部给出的区间，形如 % [12:34-15:02]。
   **不许自己编时间**——这个注释是回查录音的锚点，编了就毁了它的用途。
6. 不许新增原文没有的内容，不许把 [?] 标记删掉（那是听不清的地方）。
7. 公式必须自洽：移项、分配律的括号一个都不能丢。
   $(\hat{y}-y)\cdot x$ 不能写成 $\hat{y}-y\cdot x$，这是两个不同的式子。
8. 保持中文，专业名词首次出现可在括号里给英文。
9. 特殊字符要转义：% & # _ 分别写成 \% \& \# \_（数学模式内除外）。
10. 直接输出 LaTeX，不要用 ``` 代码块包裹，不要任何解释性文字。"""


SLIDE_RULES = r"""【怎么用课件】
- 符号、术语、公式以课件为准。转写里和课件不一致的地方按课件改——语音识别经常听错符号。
  例：课件写 alpha 是学习率，转写里成了 "L" 或 "LF"，就必须写成 $\alpha$。
  不要换成你自己习惯的符号（比如把课件的 alpha 改成 eta），照课件写。
- **待整合片段里已经用了某个符号，不等于那个符号是对的**——上一步清洗时是模型自己挑的。
  只要课件用的是另一个符号，就按课件改。课件是唯一权威。
- 章节组织可以参考课件，但只保留老师实际讲到的部分。
- 课件里老师没讲的内容不要搬进笔记。笔记的主线是老师讲的话。"""

FIGURE_RULES = SLIDE_RULES + r"""
- 插图：笔记里出现"看这个图""看黑板上""如图"这类指代，或讲到某页课件对应的内容时，
  就把那一页的图插进去：
  \begin{figure}[htbp]\centering
  \includegraphics[width=0.75\textwidth]{figures/xxx.png}
  \caption{图的说明}
  \end{figure}
- 路径只能从上面【可用图片】清单里原样复制，一个字符都不许改，不许编文件名。
- 每张图最多用一次，插在讲到它的那段文字后面。"""

# 去掉模型偶尔套上的 ``` 代码块
_FENCE = re.compile(r"^\s*```(?:latex|tex)?\s*|\s*```\s*$", re.MULTILINE)

# preamble 定义的 + LaTeX/amsmath 自带的，超出这个范围就是模型自创的
KNOWN_ENVS = {
    "theorem", "lemma", "corollary", "proposition", "definition", "example",
    "remark", "proof", "itemize", "enumerate", "description", "center",
    "equation", "equation*", "align", "align*", "gather", "gather*", "multline",
    "multline*", "split", "cases", "array", "matrix", "pmatrix", "bmatrix",
    "vmatrix", "Vmatrix", "smallmatrix", "tabular", "tabular*", "table", "table*",
    "figure", "figure*", "minipage", "quote", "quotation", "verbatim", "abstract",
    "flushleft", "flushright", "document",
}
_ENV = re.compile(r"\\begin\{([A-Za-z*]+)\}")


_GRAPHICS = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def missing_graphics(tex: str, session_dir: Path) -> list[str]:
    """模型偶尔会编图片文件名，编译时才报错，提前查出来。"""
    out = []
    for m in _GRAPHICS.finditer(tex):
        rel = m.group(1).strip()
        if not (session_dir / rel).exists():
            out.append(rel)
    return out


def unknown_envs(tex: str) -> set[str]:
    """揪出模型自创的环境——这类错误只在编译时才炸，提前报出来。"""
    return {m.group(1) for m in _ENV.finditer(tex)} - KNOWN_ENVS


_TS_ANCHOR = re.compile(r"%\s*\[(\d{2}:\d{2})-(\d{2}:\d{2})\]")
_TS_CITED = re.compile(r"\\texttt\{\[(\d{2}:\d{2})(?:-(\d{2}:\d{2}))?\]\}")


def fabricated_timestamps(tex: str) -> list[str]:
    """概要里引用的时间戳必须来自正文的 % [mm:ss-mm:ss] 锚点。

    模型编时间戳是实测发生过的（12 分钟的课引用到了 01:33-02:15）。
    伪造的锚点会让"回查录音"这条路失效，而且不看录音发现不了，所以要主动查。
    """
    anchors = set()
    for a, b in _TS_ANCHOR.findall(tex):
        anchors.update((a, b, f"{a}-{b}"))
    bad = []
    for a, b in _TS_CITED.findall(tex):
        cited = f"{a}-{b}" if b else a
        if cited not in anchors:
            bad.append(cited)
    return bad


def repair_envs(tex: str, log: Callable[[str], None] = print) -> str:
    """把模型自创的环境改写成 remark。

    实测它会无视白名单造出 \begin{note} 之类的东西。只警告不够——
    编译不过的 .tex 对赶笔记的人来说等于没有产出，而语义上 remark 就是它想要的。
    """
    bad = unknown_envs(tex)
    if not bad:
        return tex
    for name in sorted(bad):
        tex = tex.replace(f"\\begin{{{name}}}", "\\begin{remark}")
        tex = tex.replace(f"\\end{{{name}}}", "\\end{remark}")
    log(f"[修正] 模型自创了环境 {sorted(bad)}，已改写成 remark")
    return tex


SUMMARY_SYSTEM = r"""你为一份课堂笔记写开头的「本讲概要」。输入是这份笔记的正文（LaTeX）。

输出结构：

\section*{本讲概要}
用 3 到 5 句话说清这节课讲了什么：从什么问题出发、推导或论述了什么、得到什么结论。
按老师实际讲课的脉络写，不要写成教科书的目录式罗列。

\subsection*{重点与考点}
\begin{itemize}
\item 每条一个重点，末尾用 \texttt{[mm:ss]} 标出它在正文里对应的时间，方便回查录音
\end{itemize}

规则：
1. 只依据正文，不许新增正文里没有的内容。概要是对正文的压缩，不是补充。
2. 老师明确强调过的地方（「考试会考」「记住」「重点」「必考」「年年都考」）都要收进
   重点列表。明确说「不考」的也要收，但要写明「（老师说不考）」——知道什么不考同样有用。
3. 正文里标了 [?] 的地方不要写进概要，那是听不清的内容，不该出现在最显眼的位置。
4. 正文里完全没有强调过的内容时，整个 \subsection*{重点与考点} 一节都省略，不要硬凑。
5. 时间戳只能用正文 % [mm:ss-mm:ss] 注释里出现过的，不许自己编。
6. 数学符号沿用正文的写法，公式用行内 $...$。
7. 只输出这两节的 LaTeX，不要 \documentclass、不要宏包、不要 \begin{document}，
   不要用 ``` 包裹，不要任何解释性文字。"""


def _digest(bodies: list[str], budget: int = 14000) -> str:
    """喂给概要的正文。太长就按比例截断每一段，保证概要那次调用不会超预算。"""
    joined = "\n\n".join(bodies)
    if len(joined) <= budget:
        return joined
    each = max(600, budget // max(1, len(bodies)))
    parts = [b[:each] + ("\n…（此段后略）" if len(b) > each else "") for b in bodies]
    return "\n\n".join(parts)


def summarize(ep: Endpoint, bodies: list[str],
              log: Callable[[str], None] = print) -> str:
    """生成开头的概要。失败不影响笔记本身，返回空串。"""
    try:
        text, _ = chat(ep, SUMMARY_SYSTEM, _digest(bodies), temperature=0.2, max_tokens=2000)
    except LlmError as e:
        log(f"[概要] 生成失败，跳过：{e}")
        return ""
    text = _FENCE.sub("", text).strip()
    if not text:
        log("[概要] 返回为空，跳过")
    return text


def _groups(chunks: list[str], max_chars: int) -> list[list[str]]:
    """按字数把片段分组，避免单次输出被 max_tokens 截断。"""
    groups: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for c in chunks:
        if cur and size + len(c) > max_chars:
            groups.append(cur)
            cur, size = [], 0
        cur.append(c)
        size += len(c)
    if cur:
        groups.append(cur)
    return groups


def _tex_escape_title(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("_", r"\_"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def compose(session: Session, cfg: Config, ep: Endpoint, deck=None,
            max_chars: int = 6000, log: Callable[[str], None] = print) -> tuple[str, str]:
    """返回 (notes.tex 路径, notes.md 路径)。"""
    files = session.note_chunks()
    if not files:
        raise LlmError("还没有清洗过的片段，先跑 refine（或用 notes 子命令，它会自动补跑）")

    chunks = [f.read_text(encoding="utf-8") for f in files]

    md_path = session.path / "notes.md"
    md_path.write_text(f"# {session.title} · 笔记（逐段清洗）\n\n" + "\n\n".join(chunks) + "\n",
                       encoding="utf-8")

    groups = _groups(chunks, max_chars)
    log(f"[整合] {len(chunks)} 个片段 → {len(groups)} 次调用（{ep}）")

    reference = ""
    if deck is not None:
        reference = "\n\n【课件参考】（来源：" + deck.source.name + "）\n" + deck.outline()
        manifest = deck.figure_manifest()
        if manifest:
            reference += "\n\n【可用图片】\n" + manifest
        reference += "\n\n" + (FIGURE_RULES if manifest else SLIDE_RULES)
        log(f"[整合] 课件 {len(deck.slides)} 页、可用图 {len(deck.figures)} 张")

    bodies = []
    for k, group in enumerate(groups, 1):
        head = (f"这是第 {k}/{len(groups)} 部分。"
                + ("从 \\section 开始写。" if k == 1 else
                   "接着前面的内容写，不要重复已有章节，需要新章节就直接开 \\section。"))
        text, finish = chat(ep, SYSTEM,
                            head + reference + "\n\n【待整合的笔记片段】\n" + "\n\n".join(group),
                            temperature=0.2, max_tokens=8000)
        if finish == "length":
            log(f"[警告] 第 {k} 部分输出被长度限制截断，考虑把 --max-chars 调小")
        bodies.append(_FENCE.sub("", text).strip())
        log(f"[整合] 第 {k}/{len(groups)} 部分完成（{len(bodies[-1])} 字符）")

    log("[概要] 生成本讲概要…")
    summary = summarize(ep, bodies, log)

    meta = session.meta
    tex = "\n".join([
        PREAMBLE,
        rf"\title{{{_tex_escape_title(session.title)}}}",
        r"\author{课堂语音助手自动整理}",
        rf"\date{{{meta.get('created', '')[:10]}}}",
        "",
        r"\begin{document}",
        r"\maketitle",
        summary,                    # 概要放最前面，用 \section* 所以不进目录
        "",
        r"\tableofcontents",
        r"\newpage",
        "",
        "\n\n".join(bodies),
        "",
        r"\end{document}",
        "",
    ])
    lost = missing_graphics(tex, session.path)
    if lost:
        log(f"[警告] 引用了不存在的图片 {lost}，编译会失败；手动删掉或改正路径")

    tex = repair_envs(tex, log)
    faked = fabricated_timestamps(tex)
    if faked:
        log(f"[警告] 概要引用了正文里不存在的时间戳 {faked}——"
            f"这些位置回查不到录音，看笔记时留意")

    tex_path = session.path / "notes.tex"
    tex_path.write_text(tex, encoding="utf-8")
    return str(tex_path), str(md_path)
