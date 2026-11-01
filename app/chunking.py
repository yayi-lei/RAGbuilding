"""结构化分块 + 结构上下文提取。

设计目标（对比原来的「按字符数硬切」）：
  1. 优先在**标题 / 条款 / 表格**边界断块 —— 这些是文档天然的语义单元；
  2. 表格整块不拆（拆了就失去行列对应关系，等于废掉）；
  3. 记录每个块的**标题路径**（section），作为廉价而精准的结构上下文；
  4. 重叠从「上一块的尾部整行」取，绝不把句子拦腰切断。

原来的 chunk_text 用 600 字硬切，实测会把「第一章~第三章」塞进同一块，
且块边界切在句子中间。本模块修掉这两个问题。
"""
import os
import re

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP_LINES = int(os.getenv("CHUNK_OVERLAP_LINES", "2"))
# 任何块都不许超过这个长度：嵌入模型上下文有限（实测超长块会把 llama-server
# 拖进 context-shift 死循环，499% CPU 卡死半小时）
CHUNK_MAX = int(os.getenv("CHUNK_MAX", "1800"))


def _split_big_table(tbl, limit):
    """超大表格按行分组切分，每组重复表头（首行+分隔行），保住行列对应关系。"""
    header = []
    body_start = 0
    if tbl and RE_TABLE_ROW.match(tbl[0]):
        header.append(tbl[0])
        body_start = 1
        if len(tbl) > 1 and RE_TABLE_SEP.match(tbl[1]):
            header.append(tbl[1])
            body_start = 2
    head_len = sum(len(h) for h in header) + len(header)
    groups, cur, cur_len = [], [], 0
    for row in tbl[body_start:]:
        add = len(row) + 1
        if cur and head_len + cur_len + add > limit:
            groups.append(cur)
            cur, cur_len = [], 0
        cur.append(row)
        cur_len += add
    if cur:
        groups.append(cur)
    return ["\n".join(header + g) if header else "\n".join(g) for g in groups]


def _hard_cap(text, limit):
    """按行硬切兜底（绝不截断行内）。"""
    if len(text) <= limit:
        return [text]
    segs, cur, cur_len = [], [], 0
    for ln in text.split("\n"):
        add = len(ln) + 1
        if cur and cur_len + add > limit:
            segs.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(ln)
        cur_len += add
    if cur:
        segs.append("\n".join(cur))
    return segs

# markitdown 输出的 Markdown 结构标记
RE_HEADING = re.compile(r"^\s*#{1,6}\s+\S")
RE_ATX_LINE = re.compile(r"^\s*#{1,6}\s+(.*\S)\s*$")
# 粗体标题行：整行就是一个 **xxx** （markitdown 常把 Word 标题转成这样）
RE_BOLD_TITLE = re.compile(r"^\s*\*\*(.+?)\*\*\s*$")
# 注意：Python 不允许给零宽断言 \b 加量词（\b? 会报 "nothing to repeat"），
# 所以这里不能写 \b?，靠后面的边界判断即可。
RE_CLAUSE = re.compile(
    r"^\s*\**\s*第\s*[一二三四五六七八九十百零〇0-9]{1,8}\s*"
    r"(条|章|节|款|项|部分)"
)
RE_TABLE_ROW = re.compile(r"^\s*\|")
RE_TABLE_SEP = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
# 只抠条款编号本身（用作 section 名），不要把条款正文一起带上
RE_CLAUSE_NUM = re.compile(r"第\s*[一二三四五六七八九十百零〇0-9]{1,8}\s*(?:条|章|节|款|项|部分)")


def _kind(line):
    if RE_TABLE_ROW.match(line):
        return "table"
    if RE_HEADING.match(line):
        return "heading"
    if RE_CLAUSE.match(line):
        return "clause"
    return "text"


def _title_of(line, kind=None):
    """从标题行里抠出干净的标题文本（用于 section 路径）。"""
    if kind == "clause":
        m = RE_CLAUSE_NUM.search(line)
        if m:
            return m.group(0)
    m = RE_ATX_LINE.match(line)
    if m:
        return m.group(1).strip()
    m = RE_BOLD_TITLE.match(line)
    if m:
        return m.group(1).strip()
    if kind == "clause":
        m = RE_CLAUSE_NUM.search(line)
        if m:
            return m.group(0)
    return line.strip()[:60]


def _level_of(line):
    """标题层级：## 是 2。粗体标题按 3 处理。"""
    m = re.match(r"^\s*(#{1,6})\s+", line)
    if m:
        return len(m.group(1))
    return 3


def chunk_document(text, size=None, overlap_lines=None):
    """把文档切成带结构信息的块。

    返回 [{"text": str, "section": str, "kinds": [str,...], "parts": int}, ...]

    section 是该块所属的标题路径，形如 "第三章 产品设计 > 第七条"。
    """
    size = size or CHUNK_SIZE
    overlap_lines = CHUNK_OVERLAP_LINES if overlap_lines is None else overlap_lines

    raw_lines = [ln.rstrip() for ln in (text or "").split("\n")]
    lines = [ln for ln in raw_lines if ln.strip()]
    if not lines:
        return []

    chunks = []
    buf = []              # 当前块的行
    buf_len = 0
    heading_stack = []    # [(level, title)]，维护标题路径
    cur_clause = ""       # 当前条款标题（会覆盖 section 末段）

    def section_path():
        parts = [t for _, t in heading_stack if t]
        if cur_clause:
            parts = parts + [cur_clause]
        return " > ".join(parts[-3:])   # 最多保留 3 层，避免过长

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        body = "\n".join(buf).strip()
        if body:
            kinds = sorted({_kind(x) for x in buf})
            sec = section_path()
            for seg in _hard_cap(body, CHUNK_MAX):
                chunks.append({
                    "text": seg,
                    "section": sec,
                    "kinds": kinds,
                    "parts": len(buf),
                })
        buf = []
        buf_len = 0

    def tail_lines(n):
        """取上一块末尾 n 行作为重叠（整行，不切句子）。"""
        if not chunks or n <= 0:
            return []
        prev = chunks[-1]["text"].split("\n")
        return prev[-n:]

    i = 0
    while i < len(lines):
        line = lines[i]
        k = _kind(line)

        # ---- 表格：整块吞下，绝不拆 ----
        if k == "table":
            tbl = []
            while i < len(lines) and _kind(lines[i]) == "table":
                tbl.append(lines[i])
                i += 1
            tbl_text = "\n".join(tbl)
            if len(tbl_text) > size:
                # 大表：按行分组切，每组重复表头（既保结构又不超长）
                flush()
                sections = section_path()
                for piece in _split_big_table(tbl, CHUNK_MAX):
                    chunks.append({
                        "text": piece,
                        "section": sections,
                        "kinds": ["table"],
                        "parts": len(tbl),
                    })
            else:
                if buf and buf_len + len(tbl_text) > size:
                    flush()
                buf.extend(tbl)
                buf_len += len(tbl_text)
            continue

        # ---- 标题：更新路径，并开启新块 ----
        if k == "heading":
            # 关键：先用「旧」标题路径结算上一块，再更新标题栈，
            # 否则上一块会被错误地打上新章节的名字（off-by-one）。
            if buf:
                flush()
            lvl = _level_of(line)
            title = _title_of(line)
            while heading_stack and heading_stack[-1][0] >= lvl:
                heading_stack.pop()
            heading_stack.append((lvl, title))
            cur_clause = ""
            buf.append(line)
            buf_len += len(line)
            i += 1
            continue

        # ---- 条款：新条款开新块（除非当前块还很小）----
        if k == "clause":
            cur_clause = _title_of(line, kind="clause")
            if buf and buf_len >= size // 3:
                flush()
            buf.append(line)
            buf_len += len(line)
            i += 1
            continue

        # ---- 普通行：累积，超限则断 ----
        if buf and buf_len + len(line) > size:
            flush()
            for tl in tail_lines(overlap_lines):
                buf.append(tl)
            buf_len = sum(len(x) for x in buf)
        buf.append(line)
        buf_len += len(line)
        i += 1

    flush()
    return chunks


def chunk_text(text, size=None, overlap=None):
    """向后兼容：只要纯文本列表。"""
    return [c["text"] for c in chunk_document(text, size=size)]