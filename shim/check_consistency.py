#!/usr/bin/env python3
"""一致性自检：同一批文本，分别用 Cloudflare 和本地 Ollama 嵌入，比较余弦相似度。

必须在 embed-shim 容器里跑（那里有 CF_* 和 OLLAMA_URL 环境变量）：
    docker exec embed-shim python /shim/check_consistency.py

判定口径：
    cos >= 0.9990  → 两后端实质等价，向量可混写，不必重嵌
    0.990 ~ 0.999  → 同源但量化差异，建议统一走一条通道（重嵌）
    < 0.990        → 视为不同模型，绝对禁止混写
"""
from __future__ import annotations

import json
import os
import sys

import httpx

CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "").strip()
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "").strip()
CF_MODEL = os.getenv("CF_MODEL", "@cf/qwen/qwen3-embedding-0.6b")
CF_PAYLOAD_KEY = os.getenv("CF_PAYLOAD_KEY", "text")
OLLAMA_URL = os.getenv("OLLAMA_URL", "").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-embedding:0.6b")

SAMPLES = [
    "这是一段用于比对两种嵌入后端输出的中文样例文本，长度接近真实分块。",
    "Instruct: Given a Chinese question, retrieve relevant document passages that answer the question\nQuery:如何申请报销",
    "# 第三章 产品设计 第七条 产品的定价与审批流程说明",
    "示例文档/产品管理办法（示例版）.docx",
    "安全服务内容包括边界防护、主机防护与日常安全运营。",
]


def cos(a, b):
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def get_cf(texts):
    url = (f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}"
           f"/ai/run/{CF_MODEL}")
    with httpx.Client(timeout=90) as c:
        r = c.post(url, headers={"Authorization": f"Bearer {CF_API_TOKEN}"},
                   json={CF_PAYLOAD_KEY: texts})
        r.raise_for_status()
        d = r.json()
    res = d.get("result", d)
    data = res.get("data") if isinstance(res, dict) else res
    return data


def get_ollama(texts):
    with httpx.Client(timeout=240) as c:
        r = c.post(f"{OLLAMA_URL}/api/embed",
                   json={"model": OLLAMA_MODEL, "input": texts})
        r.raise_for_status()
        return r.json()["embeddings"]


def main() -> int:
    if not (CF_ACCOUNT_ID and CF_API_TOKEN):
        print("缺少 CF_ACCOUNT_ID / CF_API_TOKEN")
        return 2
    print(f"cf_model={CF_MODEL}  ollama={OLLAMA_URL} model={OLLAMA_MODEL}")

    cf = get_cf(SAMPLES)
    ol = get_ollama(SAMPLES)
    if cf is None or len(cf) != len(SAMPLES):
        print(f"CF 返回异常：{str(cf)[:300]}")
        return 1

    print(f"cf_dim={len(cf[0])}  ollama_dim={len(ol[0])}")
    if len(cf[0]) != len(ol[0]):
        print("维度不同，禁止混写")
        return 1

    vals = []
    for i, t in enumerate(SAMPLES):
        s = cos(cf[i], ol[i])
        vals.append(s)
        print(f"  [{i}] cos={s:.6f}   {t[:44]}")

    avg = sum(vals) / len(vals)
    print(f"\n平均 cos = {avg:.6f}   最小 = {min(vals):.6f}")
    if avg >= 0.9990:
        verdict = "等价：向量可混写，可以不重嵌"
    elif avg >= 0.990:
        verdict = "同源量化差异：建议全量重嵌到主通道"
    else:
        verdict = "不同模型：必须全量重嵌"
    print("结论：" + verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
