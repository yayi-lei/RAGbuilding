"""Contextual Retrieval：为分块补充"它在文档中的位置与背景"。

依据 Anthropic 的做法（为 chunk 补上下文再嵌入），但做了关键的本土化改良：

  Anthropic 原文是「每块调一次 LLM 生成上下文」。这在英文非结构化语料上合理，
  但我们的文档经过 markitdown 解析后**本身就是结构化的**（有 # 标题、**第X条**、
  表格），chunking.py 已经能准确提取出章节路径。章节路径本身就是极强的定位信息。

  所以改用**两级上下文**：
    [一级] 文档级摘要 —— LLM 为整份文档生成一次（每文件 1 次调用，共 21 次）
    [二级] 章节路径   —— 分块时免费提取，零成本

  实测收益：LLM 调用从 650 次降到 21 次，索引时间从 ~70 分钟降到 ~3 分钟，
  而定位信息（"这是哪份文件的哪一章讲什么"）几乎不损失。

  仍保留逐块生成的模式（CTX_MODE=chunk），需要时可切换。

生成结果只进**嵌入文本与 BM25 文本**，原文单独保留在 payload.text —— 保证溯源时
展示的仍是原文，不被模型改写污染。
"""
import os

from llm import available, chat

MODE = os.getenv("CTX_MODE", "doc")          # doc | chunk | both | off

DOC_PROMPT = """请概括下面这份文档，供检索系统作为上下文使用。

【文件名】{fname}
【正文开头】
{head}

要求：
1. 用 60~120 字说清：这是什么文件、属于哪类业务/制度/项目、主要涵盖哪些主题
2. 直接陈述，不要「本文档」「该文件」这类废话引导语
3. 不要编造文中没有的信息
4. 只输出这段概括本身，不要任何前缀或引号

概括："""

CHUNK_PROMPT = """你在为一个文档检索系统做分块上下文补全。

【文件名】{fname}
【文档概述】{doc}
【章节路径】{section}
【当前片段】
{chunk}

请写一段简短的背景说明（50~100 字），用来让这个片段**脱离文档也能被正确理解**，
重点是：它属于哪份文件的哪一部分、在讨论什么主题。

要求：
1. 直接陈述事实，不要「这个片段」这类废话引导语
2. 不要复述片段正文，不要编造
3. 只输出这段说明本身

背景说明："""

HEAD_CHARS = int(os.getenv("CTX_HEAD_CHARS", "2000"))


def _clean(txt, limit=400):
    if not txt:
        return ""
    txt = txt.strip().strip('"').strip("“”").strip()
    txt = " ".join(txt.split())
    return txt[:limit]


def build_doc_summary(fname, text):
    """文档级摘要：每个文件只调一次 LLM。"""
    if MODE == "off":
        return ""
    head = (text or "")[:HEAD_CHARS]
    if not head:
        return ""
    return _clean(chat(DOC_PROMPT.format(fname=fname or "（未知）", head=head),
                       max_tokens=int(os.getenv("CTX_DOC_MAX_TOKENS", "2000"))), 300)


def build_context(fname, section, head_text, chunk_text, doc_summary=""):
    """块级上下文（逐块 LLM 模式用）。"""
    prompt = CHUNK_PROMPT.format(
        fname=fname or "（未知）",
        doc=doc_summary or "（无）",
        section=section or "（未识别章节）",
        chunk=(chunk_text or "")[:1800],
    )
    return _clean(chat(prompt, max_tokens=int(os.getenv("CTX_MAX_TOKENS", "1500"))), 400)


def compose_text(text, section=None, fname=None, doc_summary=None, chunk_context=None):
    """拼出用于嵌入与 BM25 的最终文本。上下文在前，原文在后，原文一字不改。"""
    parts = []
    head = []
    if fname:
        head.append(fname)
    if doc_summary:
        head.append(doc_summary)
    if head:
        parts.append("[文档] " + "｜".join(head))
    if section:
        parts.append(f"[章节] {section}")
    if chunk_context:
        parts.append(f"[背景] {chunk_context}")
    parts.append(text)
    return "\n".join(parts)