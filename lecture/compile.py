r"""把 notes.tex 编译成 notes.pdf。

三个必须守住的点：

1. **cwd 必须是 session 目录**，不能靠 `-output-directory`。笔记里的图是
   `figures/xxx.png` 这种相对路径（见 `latex.PREAMBLE` 的 `\graphicspath`），
   在别处编译就找不到图。
2. **要跑两遍**。第一遍才生成 .aux/.toc，`\tableofcontents` 在第二遍才有内容。
   只跑一遍的表现是目录页空白——PDF 出来了，但缺了东西，很容易没注意。
3. **编不出来不算失败**。装 MiKTeX / TeXLive 是 GB 级的事，不能因为没装就让
   整条链断掉。这里返回 None 并说明原因，调用方照常输出 .tex。

中文要用 xelatex（ctexart），pdflatex 不行，所以不做后备引擎。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# 一遍的上限。90 分钟的课正文几万字，正常几秒到几十秒；卡住多半是
# 缺宏包时 tex 在等交互输入（-interaction=nonstopmode 已经挡了大部分）
TIMEOUT = 180.0

# PATH 里没有时按常见安装位置找。Windows 用户装 MiKTeX 之后不一定重启过终端
_CANDIDATES_WIN = (
    r"C:\Program Files\MiKTeX\miktex\bin\x64",
    r"C:\Program Files (x86)\MiKTeX\miktex\bin",
    r"C:\texlive\2026\bin\windows",
    r"C:\texlive\2025\bin\windows",
    r"C:\texlive\2024\bin\windows",
)


def _user_candidates() -> tuple[Path, ...]:
    home = Path.home()
    out = [
        home / "AppData/Local/Programs/MiKTeX/miktex/bin/x64",
        home / "AppData/Roaming/TinyTeX/bin/windows",
        home / "AppData/Roaming/TinyTeX/bin/win32",
        # TinyTeX 在 mac 上按架构分目录，两种都试
        home / "Library/TinyTeX/bin/universal-darwin",
        home / "Library/TinyTeX/bin/x86_64-darwin",
        home / "bin",
    ]
    return tuple(out)


def find_engine() -> Path | None:
    """找 xelatex。PATH 优先，再按常见安装位置翻。找不到返回 None。"""
    found = shutil.which("xelatex")
    if found:
        return Path(found)

    exe = "xelatex.exe" if os.name == "nt" else "xelatex"
    roots: tuple = _user_candidates()
    if os.name == "nt":
        roots = tuple(Path(p) for p in _CANDIDATES_WIN) + roots
    for root in roots:
        candidate = root / exe
        if candidate.is_file():
            return candidate
    return None


@dataclass
class Result:
    pdf: Path | None
    reason: str = ""        # pdf 为 None 时说明为什么
    log_tail: str = ""      # 出错时 .log 的尾巴，够定位问题

    @property
    def ok(self) -> bool:
        return self.pdf is not None


# 从 .log 里挑真正的报错。TeX 的 log 几万行，绝大部分是宏包在报到货
_ERROR = re.compile(r"^(?:! |!pdfTeX error|! LaTeX Error|! Package .* Error)", re.MULTILINE)


def _errors(log_text: str, limit: int = 6) -> str:
    """把 .log 里的报错抽出来。每条带上后面两行，那才看得出是哪个环境/宏包。"""
    lines = log_text.splitlines()
    hits: list[str] = []
    for i, line in enumerate(lines):
        if _ERROR.match(line):
            hits.append("\n".join(lines[i:i + 3]))
            if len(hits) >= limit:
                break
    return "\n---\n".join(hits)


def compile_tex(tex: str | Path, *, log: Callable[[str], None] = print,
                passes: int = 2, timeout: float = TIMEOUT) -> Result:
    """编译 tex，返回 Result。不抛异常——编不出来是可接受的结果。"""
    tex_path = Path(tex).resolve()
    if not tex_path.exists():
        return Result(None, f"找不到 {tex_path}")

    engine = find_engine()
    if engine is None:
        return Result(None,
                      "机器上没有 xelatex，跳过编译。中文笔记要装 MiKTeX、TeXLive "
                      "或 TinyTeX（装完新开一个终端，或者把它的 bin 目录加进 PATH）。")

    workdir = tex_path.parent
    pdf = workdir / (tex_path.stem + ".pdf")
    # 上一次的产物留着会造成「编译失败但 PDF 还在」的假象
    pdf.unlink(missing_ok=True)

    log(f"[编译] {engine.name} × {passes} 遍，在 {workdir.name}/ 里")
    last = None
    for i in range(1, passes + 1):
        try:
            last = subprocess.run(
                [str(engine), "-interaction=nonstopmode", "-halt-on-error",
                 tex_path.name],
                cwd=workdir, timeout=timeout,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
        except subprocess.TimeoutExpired:
            return Result(None, f"第 {i} 遍编译超过 {timeout:.0f}s，放弃")
        except OSError as exc:
            return Result(None, f"起不来 {engine}：{exc}")
        # 第一遍失败就不用跑第二遍了；但第二遍失败而 PDF 已经在了，
        # 那多半只是目录/引用没收敛，PDF 仍然可用
        if last.returncode != 0 and not (i > 1 and pdf.exists()):
            break

    log_file = workdir / (tex_path.stem + ".log")
    log_text = ""
    if log_file.exists():
        log_text = log_file.read_text(encoding="utf-8", errors="replace")

    if not pdf.exists():
        detail = _errors(log_text) or (
            (last.stdout or b"").decode("utf-8", "replace")[-1200:] if last else "")
        return Result(None, "xelatex 编译失败", detail.strip())

    if last is not None and last.returncode != 0:
        log("[编译] 有报错但 PDF 生成了（多半是目录没收敛），当成功用")
    log(f"[编译] → {pdf.name}（{pdf.stat().st_size / 1e6:.1f} MB）")
    return Result(pdf)


if __name__ == "__main__":       # 手动试一份：python -m lecture.compile 某个/notes.tex
    if len(sys.argv) < 2:
        print("用法：python -m lecture.compile <notes.tex>")
        raise SystemExit(2)
    result = compile_tex(sys.argv[1])
    if result.ok:
        print(f"OK → {result.pdf}")
        raise SystemExit(0)
    print(f"没出 PDF：{result.reason}")
    if result.log_tail:
        print("\n--- log ---\n" + result.log_tail)
    raise SystemExit(1)
