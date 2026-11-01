#!/usr/bin/env python3
"""用当前嵌入后端（embed-shim）重建全部向量，payload 原样保留。

用途：把旧机本地模型算出的 1187 个向量，换成新主模型（Cloudflare）算出的向量，
保证整个 collection 的向量空间一致。不重新解析文档、不做 OCR、不重新分块。

跑法（在 kb-mcp 容器里）：
    docker exec kb-mcp python /app/reembed_all.py [--dry-run] [--limit N] [--batch 4]
"""
from __future__ import annotations

import argparse
import sys
import time

import common
from qdrant_client import models


def fetch_all(limit: int | None = None):
    pts, offset, out = [], None, []
    while True:
        batch, offset = common.qdrant.scroll(
            collection_name=common.COLLECTION, limit=256, offset=offset,
            with_payload=True, with_vectors=False,
        )
        out.extend(batch)
        if limit and len(out) >= limit:
            return out[:limit]
        if offset is None:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    print(f"collection={common.COLLECTION} dim={common.EMBED_DIM} "
          f"model={common.EMBED_MODEL} ollama_url={common.OLLAMA_URL}", flush=True)

    pts = fetch_all(args.limit)
    print(f"待重嵌分块：{len(pts)}", flush=True)
    if not pts:
        return 1

    texts, ids, payloads = [], [], []
    for p in pts:
        pl = dict(p.payload or {})
        src = pl.get("embed_text") or pl.get("text") or ""
        if not src.strip():
            print(f"  [skip] 无文本 {p.id}", flush=True)
            continue
        texts.append(src)
        ids.append(p.id)
        payloads.append(pl)

    print(f"有效样本：{len(texts)}", flush=True)
    if args.dry_run:
        print("dry-run：不写入", flush=True)
        return 0

    t0 = time.time()
    vecs: list[list[float]] = []
    bs = args.batch
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        got = common.embed_documents(chunk, batch_size=len(chunk))
        vecs.extend(got)
        done = min(i + bs, len(texts))
        rate = done / max(time.time() - t0, 1e-6)
        eta = (len(texts) - done) / max(rate, 1e-6)
        print(f"  {done}/{len(texts)}  {rate:.1f} 块/s  ETA {eta/60:.1f} min", flush=True)

    bad = [i for i, v in enumerate(vecs) if len(v) != common.EMBED_DIM]
    if bad:
        print(f"维度异常 {len(bad)} 条，中止写入", flush=True)
        return 1

    print("写入 qdrant ...", flush=True)
    for i in range(0, len(vecs), 128):
        batch = [
            models.PointStruct(id=ids[j], vector=vecs[j], payload=payloads[j])
            for j in range(i, min(i + 128, len(vecs)))
        ]
        common.qdrant.upsert(collection_name=common.COLLECTION, points=batch)
        print(f"  upsert {min(i+128, len(vecs))}/{len(vecs)}", flush=True)

    n = common.collection_count()
    print(f"完成。collection 点数={n}，耗时 {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
