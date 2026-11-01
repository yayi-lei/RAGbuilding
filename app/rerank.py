"""LLM 重排（rerank）。

做法：一次拿到 N 个候选（默认 20），交给 glm-5.3-flash 做 **listwise** 排序，
返回按相关性从高到低排列的候选。

为什么用 listwise 而不是 pointwise 打分：
  pointwise 要对每个候选各发一次请求（20 次），listwise 只要 1 次，
  而且跨候选比较的一致性更好（模型能看到全集，不会各打各的分数）。

失败降级：LLM 不可用或解析失败 → 原样返回（保持 RRF 的排序），绝不阻断检索。
"""
import os
import re

from llm import chat, extract_json

PROMPT = """你是一个检索结果相关性排序员。

用户问题：{query}

下面是 {n} 个候选片段（已编号）。请按「能否回答用户问题」的相关性从高到低排序。

评判标准：
1. 片段是否直接回答了问题（最重要）
2. 是否包含问题中的关键概念/名词/编号
3. 只是主题相近但没有实质内容的，排后面

候选片段：
{cands}

只输出一行 JSON，格式为 {{"order":[最相关的编号, 次相关的编号, ...]}}，
必须包含全部 {n} 个编号、不重不漏、不要解释、不要 markdown 代码块。"""

SNIPPET = int(os.getenv("RERANK_SNIPPET", "420"))


def _fmt(i, c):
    src = c.get("source", "")
    sec = c.get("section") or ""
    where = f"{src}" + (f" / {sec}" if sec else "")
    body = " ".join((c.get("text") or "").split())[:SNIPPET]
    return f"[{i}] （{where}）\n{body}"


def rerank(query, cands, top_n=None):
    """对 cands 重排。失败时原样返回前 top_n 个。"""
    if not cands:
        return []
    top_n = top_n or len(cands)
    if len(cands) == 1:
        return cands[:top_n]

    listing = "\n\n".join(_fmt(i, c) for i, c in enumerate(cands))
    prompt = PROMPT.format(query=query, n=len(cands), cands=listing)
    raw = chat(prompt, max_tokens=int(os.getenv("RERANK_MAX_TOKENS", "1200")))

    order = None
    obj = extract_json(raw)
    if isinstance(obj, dict):
        order = obj.get("order") or obj.get("ranking") or obj.get("ids")
    elif isinstance(obj, list):
        order = obj
    if order is None and raw:
        # 兜底：从文本里抠出所有编号序列
        nums = re.findall(r"\d+", raw)
        if nums:
            order = nums

    if not order:
        return cands[:top_n]

    seen, out = set(), []
    for x in order:
        try:
            i = int(x)
        except Exception:
            continue
        if 0 <= i < len(cands) and i not in seen:
            seen.add(i)
            out.append(cands[i])
    # 模型漏掉的补在后面（保持原顺序），保证不丢结果
    for i, c in enumerate(cands):
        if i not in seen:
            out.append(c)
    return out[:top_n]