"""用 LLM 给文档推荐归档位置（知识库 / 分组 / 标签）。

设计要点（参考 dsh-knowledge 的自动回写策略）：
  1. 把「已有分组清单」一起给模型，要求**优先复用**，逼它别乱建新组；
  2. 拿不准就返回 null —— 落到「未分组」等人工处理，宁可留白；
  3. 有重试与降级：第一次带正文，失败则只用文件名再试一次；
  4. 输出必须是 JSON，解析失败视为失败（不猜）；
  5. 模型/地址/Key 一律从 ai_settings 现读（管理台可改，改完立即生效），
     网关换了模型名不用重建容器。
"""
import json
import os
import re
import time

import httpx

import ai_settings

HEAD_CHARS = int(os.getenv("CLASSIFY_HEAD_CHARS", "1200"))

PROMPT = """你是企业文档管理员，负责把新文档归档。请判断它应该放进哪个分组。

【已有分组】（**优先从这些里选**，不要轻易新建）：
{groups}

【可用知识库】{kbs}

【文件名】{fname}
【文档正文开头】
{head}

要求：
1. `kb`：从这个列表里选一个最合适的知识库名；拿不准就 null
2. `group`：从已有分组里选一个最合适的；**确实都不合适**才建议新分组名（≤12 字，可用「父/子」两级）；
   拿不准就 null
3. `tags`：给 1~3 个标签（如：合同、招标、政策、模板、财务、技术、会议）
4. `confidence`：high / medium / low
5. **只输出一行 JSON**，不要解释、不要 markdown 代码块：
{{"kb": "...", "group": "...", "tags": ["..."], "confidence": "high"}}"""


def _clean_head(text, limit=None):
    """清洗正文开头：去掉 markdown 表格的管道噪声、压缩空白。"""
    limit = limit or HEAD_CHARS
    if not text:
        return ""
    t = text.replace("\r", "\n")
    # xlsx 转出来的表经常是 "| a | b | NaN | NaN |" 这种，砍掉大量管道与 NaN
    t = re.sub(r"\bNaN\b", "", t)
    t = re.sub(r"[ \t]*\|[ \t]*", " | ", t)
    t = re.sub(r"(?: *\| *){3,}", " | ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()[:limit]


def _extract_json(s):
    """从模型输出里抠出第一个 JSON 对象。"""
    if not s:
        return None
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # 直接解析
    try:
        return json.loads(s)
    except Exception:
        pass
    # 退而求其次：找第一个 { ... }
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _call(prompt, max_tokens=None, cfg=None):
    """返回 (content, finish_reason)。配置每次现读。"""
    cfg = cfg or ai_settings.current()
    if not cfg.get("enabled"):
        raise RuntimeError("AI 建议已在设置里关闭")
    if not (cfg.get("base_url") and cfg.get("api_key")):
        raise RuntimeError("未配置 VLM_BASE_URL / VLM_API_KEY")
    body = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens or cfg.get("max_tokens") or 2000),
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
    with httpx.Client(timeout=float(cfg.get("timeout") or 120)) as c:
        r = c.post(f"{cfg['base_url']}/chat/completions", headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
    ch = (data.get("choices") or [{}])[0]
    content = ((ch.get("message") or {}).get("content") or "").strip()
    return content, ch.get("finish_reason")


def _kind_of(err):
    """把失败原因归类，前端好显示人话。"""
    low = (err or "").lower()
    if "not supported by any configured account" in low or "model_not_found" in low:
        return "model_not_found"
    if "已在设置里关闭" in (err or ""):
        return "disabled"
    if "401" in low or "403" in low or "unauthorized" in low:
        return "auth"
    if "未配置" in (err or ""):
        return "unconfigured"
    if "timeout" in low or "connect" in low or "getaddrinfo" in low:
        return "network"
    return "error"


def classify(source, head_text, existing_groups, kb_names):
    """返回 {"kb","group","tags","confidence","model","seconds"} 或 None。"""
    cfg = ai_settings.current()
    model = cfg.get("model") or ""
    mt = int(cfg.get("max_tokens") or 2000)
    fname = source.split("/")[-1]
    groups_txt = "\n".join(f"- {g}" for g in (existing_groups or [])) or "（暂无已有分组）"
    kbs_txt = "、".join(kb_names or []) or "（无）"

    attempts = []
    head = _clean_head(head_text)
    if head:
        attempts.append(PROMPT.format(groups=groups_txt, kbs=kbs_txt, fname=fname, head=head))
    # 降级：只用文件名
    attempts.append(PROMPT.format(groups=groups_txt, kbs=kbs_txt, fname=fname,
                                  head="（正文不可用，仅凭文件名判断；拿不准就输出 null）"))

    t0 = time.time()
    last_err = ""
    for i, prompt in enumerate(attempts):
        try:
            raw, fr = _call(prompt, mt, cfg=cfg)
            # content 为空且被截断（思考链吃光了配额）→ 加倍 max_tokens 重试
            if (not raw) and fr == "length":
                raw, fr = _call(prompt, mt * 2, cfg=cfg)
            # 仍为空 → 极简提示再试一次（不带正文，最短输出）
            if not raw:
                short = ('只输出一行 JSON，不要任何思考与解释。文件名：%s。'
                         '从这些分组里选：%s。格式：'
                         '{"kb":"...","group":"...","tags":["..."],"confidence":"high"}'
                         ) % (source.split("/")[-1], "、".join(existing_groups or []) or "无")
                raw, fr = _call(short, mt * 2, cfg=cfg)

            obj = _extract_json(raw)
            if isinstance(obj, dict):
                g = obj.get("group")
                k = obj.get("kb")
                tags = obj.get("tags") or []
                if isinstance(tags, str):
                    tags = [x.strip() for x in re.split(r"[，,;；\s]+", tags) if x.strip()]
                tags = [str(x).strip() for x in tags if str(x).strip()][:3]
                return {
                    "kb": (str(k).strip() if k else None),
                    "group": (str(g).strip() if g else None),
                    "tags": tags,
                    "confidence": (obj.get("confidence") or "medium"),
                    "model": model,
                    "seconds": round(time.time() - t0, 1),
                    "attempt": i + 1,
                }
            last_err = ("截断" if fr == "length" else ("空的 content" if not raw else "JSON 解析失败")) + \
                       f" (finish={fr}, len={len(raw)})"
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:100]}"
        time.sleep(0.8)

    # 全部失败：把原因带回去，方便排查（前端直接显示 error）
    return {"kb": None, "group": None, "tags": [], "confidence": "low",
            "model": model, "seconds": round(time.time() - t0, 1),
            "attempt": len(attempts), "error": last_err,
            "error_kind": _kind_of(last_err)}