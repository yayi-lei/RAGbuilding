"""统一的 LLM 调用封装（给上下文生成 / 重排 / 归类共用）。

踩过的坑（来自 classify.py 的实战经验）：
  glm-5.3 系会先输出 reasoning_content（思考链），
  若 max_tokens 给小了，思考链会把配额吃光导致 content 为空。
  这里统一做「加倍重试 + 极简提示兜底」。

配置同样走 ai_settings（管理台可改，改完立即生效）。
"""
import json
import os
import random
import re
import threading
import time

import httpx

import ai_settings

LLM_RETRY = int(os.getenv("LLM_RETRY", "3"))
LLM_SEMAPHORE = threading.Semaphore(int(os.getenv("LLM_CONCURRENCY", "4")))

_local = threading.local()


def _client(timeout):
    c = getattr(_local, "client", None)
    if c is None or getattr(_local, "timeout", None) != timeout:
        c = httpx.Client(timeout=timeout)
        _local.client = c
        _local.timeout = timeout
    return c


def _cfg():
    return ai_settings.current()


def available():
    c = _cfg()
    return bool(c.get("enabled") and c.get("base_url") and c.get("api_key"))


def _once(prompt, max_tokens, temperature=0.0, cfg=None):
    """一次调用，返回 (content, finish_reason)。"""
    cfg = cfg or _cfg()
    body = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
    r = _client(float(cfg.get("timeout") or 180)).post(
        f"{cfg['base_url']}/chat/completions", headers=headers, json=body)
    r.raise_for_status()
    data = r.json()
    ch = (data.get("choices") or [{}])[0]
    content = ((ch.get("message") or {}).get("content") or "").strip()
    return content, ch.get("finish_reason")


def chat(prompt, max_tokens=None, temperature=0.0, retry=None):
    """带重试与降级。返回文本，失败返回 ""。"""
    cfg = _cfg()
    if not (cfg.get("enabled") and cfg.get("base_url") and cfg.get("api_key")):
        return ""
    retry = LLM_RETRY if retry is None else retry
    mt = int(max_tokens or cfg.get("max_tokens") or 3000)
    last = ""
    with LLM_SEMAPHORE:
        for attempt in range(retry):
            try:
                txt, fr = _once(prompt, mt, temperature, cfg=cfg)
                if txt:
                    return txt
                # content 为空且被截断（思考链吃光配额）→ 加倍
                if fr == "length":
                    mt *= 2
                last = f"empty(finish={fr})"
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(min(2 ** attempt, 8) + random.random())
    print(f"    [llm 失败] {last}")
    return ""


def extract_json(s):
    """从模型输出里抠出第一个 JSON（兼容 ```json 围栏）。"""
    if not s:
        return None
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"[\{\[].*[\}\]]", s, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None