"""公共模块：向量库连接、嵌入调用、分块、混合检索、重排。

索引（index.py）和检索（server.py）共用这里，保证两边用的是同一个嵌入模型 ——
这是整套方案唯一不能出错的地方：索引和查询的向量空间必须一致。

v2 改造（2026-10-28）：
  1. 分块改为**结构化**（见 chunking.py）：标题/条款/表格边界优先断块；
  2. 检索改为**混合**：稠密向量 + BM25 全文索引，Qdrant 原生 RRF 融合；
  3. 接入 LLM 重排（rerank.py）；
  4. payload 下沉 section / kb / group / tags，支持向量级过滤。
"""
import os

import ollama
from qdrant_client import QdrantClient, models

from chunking import chunk_document, chunk_text  # noqa: F401  （chunk_text 供兼容调用）

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
COLLECTION = os.getenv("COLLECTION", "office_docs")

DOCS_DIR = os.getenv("DOCS_DIR", "/docs")
STATE_FILE = os.getenv("STATE_FILE", "/state/index_state.json")
OCR_CACHE = os.getenv("OCR_CACHE", "/ocr_cache")
TMP_DIR = os.getenv("TMP_DIR", "/tmp/ragtmp")

# 中文文档 600 字一块（结构化分块下的软上限）
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# 混合检索：每路召回多少候选，最终返回多少，重排前取多少
RECALL_N = int(os.getenv("RECALL_N", "30"))
RERANK_N = int(os.getenv("RERANK_N", "20"))
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1") not in ("0", "false", "False")

TEXT_FIELD = "text"          # BM25 全文索引字段（同时是 dense 的解析字段）

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
# 嵌入客户端加超时：runner 劣化时请求会挂死，必须能超时退出再重试
EMBED_TIMEOUT = int(os.getenv("EMBED_TIMEOUT", "240"))
ollama_client = ollama.Client(host=OLLAMA_URL, timeout=EMBED_TIMEOUT)

# qwen3-embedding 系列要求：查询加指令前缀，文档不加（非对称设计）
QUERY_INSTRUCTION = os.getenv(
    "QUERY_INSTRUCTION",
    "Given a Chinese question, retrieve relevant document passages that answer the question",
)


class Hit:
    """统一的检索结果对象（兼容 qdrant ScoredPoint 的 .payload / .score 访问）。"""
    __slots__ = ("id", "score", "payload")

    def __init__(self, pid, score, payload):
        self.id = pid
        self.score = score
        self.payload = payload or {}


def _vectors(resp):
    """兼容 ollama-python 不同版本的返回类型。"""
    if isinstance(resp, dict):
        return resp["embeddings"]
    return resp.embeddings


# ---------------------------------------------------------------- 建库 / 索引

def ensure_text_index():
    """给 text 字段建 BM25 全文索引（混合检索的另一条腿）。幂等。"""
    try:
        qdrant.create_payload_index(
            collection_name=COLLECTION,
            field_name=TEXT_FIELD,
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.MULTILINGUAL,
                min_token_len=1,
                max_token_len=30,
                lowercase=True,
                scoring=models.TextScoringParams(type=models.TextScoringType.BM25),
            ),
        )
        print(f"已创建 BM25 全文索引：{COLLECTION}.{TEXT_FIELD}")
    except Exception as e:
        msg = str(e)
        if "already exists" in msg or "already" in msg.lower():
            return
        print(f"[warn] 全文索引创建失败（混合检索将退化为纯向量）：{msg[:200]}")


def ensure_sortable_indexes():
    """给过滤字段建 payload 索引（kb/group/tags/source）。"""
    for f, schema in (
        ("source", models.PayloadSchemaType.KEYWORD),
        ("kb", models.PayloadSchemaType.KEYWORD),
        ("group", models.PayloadSchemaType.KEYWORD),
        ("tags", models.PayloadSchemaType.KEYWORD),
        ("chunk", models.PayloadSchemaType.INTEGER),
    ):
        try:
            qdrant.create_payload_index(collection_name=COLLECTION, field_name=f, field_schema=schema)
        except Exception:
            pass


def ensure_collection():
    if not qdrant.collection_exists(COLLECTION):
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
        )
        print(f"已创建集合 {COLLECTION}（维度 {EMBED_DIM}）")
    ensure_text_index()
    ensure_sortable_indexes()


# 嵌入输入硬上限：超长输入会把 llama-server 拖进 context-shift 死循环
EMBED_MAX_CHARS = int(os.getenv("EMBED_MAX_CHARS", "2400"))


def embed_documents(texts, batch_size=None, on_progress=None):
    """文档嵌入：不加指令前缀。分批调用，支持进度回调。

    带重试：llama-server runner 在低主频 CPU 上会周期性劣化（ watchdog 会杀之），
    被杀时进行中的请求 EOF。重试让本批落到 ollama 自动重拉的新 runner 上，
    避免整个文件从头再来。批次默认调小到 4，降低单批触发劣化的概率。
    """
    import time as _t
    if batch_size is None:
        # 每请求固定开销 ~35s，大批次能摊薄；但批次越大，进度回调的间隔越长
        # （16 块批 ≈ 230s 才动一格，界面看起来像卡死）。
        # 折中取 4 块：约 60s 动一格，速度损失约 10~20%。
        batch_size = int(os.getenv("EMBED_BATCH", "4"))
    texts = [t[:EMBED_MAX_CHARS] if len(t) > EMBED_MAX_CHARS else t for t in texts]
    out = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        err = None
        for attempt in range(4):
            # 每次尝试前先报一次进度：重试等待期间界面也能看到心跳
            if on_progress:
                try:
                    on_progress(min(i, total), total, attempt)
                except Exception:
                    pass
            try:
                out.extend(_vectors(ollama_client.embed(model=EMBED_MODEL, input=batch)))
                err = None
                break
            except Exception as e:
                err = e
                print(f"    [嵌入重试 {attempt + 1}/4] batch@{i}: {str(e)[:120]}", flush=True)
                _t.sleep(15 * (attempt + 1))
        if err is not None:
            raise err
        if on_progress:
            try:
                on_progress(min(i + batch_size, total), total, 0)
            except TypeError:
                try:
                    on_progress(min(i + batch_size, total), total)
                except Exception:
                    pass
            except Exception:
                pass
    return out


def embed_query(text):
    """查询嵌入：必须加指令前缀，和文档侧不对称。"""
    prompt = f"Instruct: {QUERY_INSTRUCTION}\nQuery:{text}"
    return _vectors(ollama_client.embed(model=EMBED_MODEL, input=[prompt]))[0]


# ---------------------------------------------------------------- 过滤构造

def build_filter(sources=None, kb=None, group=None, tags=None):
    """把各种范围条件拼成 Qdrant filter。全为空则返回 None。"""
    must = []
    if sources:
        must.append(models.FieldCondition(key="source", match=models.MatchAny(any=list(sources))))
    if kb:
        must.append(models.FieldCondition(key="kb", match=models.MatchValue(value=kb)))
    if group:
        must.append(models.FieldCondition(key="group", match=models.MatchValue(value=group)))
    if tags:
        must.append(models.FieldCondition(key="tags", match=models.MatchAny(any=list(tags))))
    if not must:
        return None
    return models.Filter(must=must)


# ---------------------------------------------------------------- 混合检索

def hybrid_hits(query, limit=6, recall_n=None, flt=None, use_rerank=None):
    """混合召回（BM25 + 稠密，RRF 融合）→ 可选 LLM 重排。

    返回 [Hit, ...]，按相关性降序。
    """
    recall_n = recall_n or RECALL_N
    use_rerank = RERANK_ENABLED if use_rerank is None else use_rerank

    vector = embed_query(query)
    prefetch = [
        models.Prefetch(query=models.TextQuery(text=query), using=TEXT_FIELD, limit=recall_n, filter=flt),
        models.Prefetch(query=vector, using="", limit=recall_n, filter=flt),
    ]
    try:
        res = qdrant.query_points(
            collection_name=COLLECTION,
            prefetch=prefetch,
            query=models.RrfQuery(rrf=models.Rrf()),
            limit=max(limit, RERANK_N if use_rerank else limit),
            with_payload=True,
        )
        hits = [Hit(p.id, getattr(p, "score", None), p.payload) for p in res.points]
    except Exception as e:
        # 退化：纯向量检索（全文索引缺失时也能用）
        print(f"[warn] 混合检索失败，退化为向量检索：{str(e)[:160]}")
        res = qdrant.query_points(
            collection_name=COLLECTION, query=vector, limit=max(limit, recall_n),
            with_payload=True, query_filter=flt,
        )
        hits = [Hit(p.id, getattr(p, "score", None), p.payload) for p in res.points]

    if not hits:
        return []

    if use_rerank and len(hits) > 1:
        try:
            from rerank import rerank as _rr
            cands = [{"id": h.id, "score": h.score, "payload": h.payload,
                      "source": (h.payload or {}).get("source"),
                      "section": (h.payload or {}).get("section"),
                      "text": (h.payload or {}).get("text")} for h in hits]
            ordered = _rr(query, cands, top_n=limit)
            out = []
            for rank, c in enumerate(ordered):
                h = Hit(c["id"], c.get("score"), c.get("payload"))
                h.payload = dict(h.payload or {})
                h.payload["rrf_score"] = c.get("score")
                h.payload["rerank_pos"] = rank + 1
                out.append(h)
            return out
        except Exception as e:
            print(f"[warn] 重排失败，保留 RRF 顺序：{str(e)[:160]}")

    return hits[:limit]


# ---------------------------------------------------------------- 兼容接口

def search(query, top_k=6, flt=None, use_rerank=None):
    """语义+关键词混合检索。返回 Hit 列表。"""
    return hybrid_hits(query, limit=top_k, flt=flt, use_rerank=use_rerank)


def search_in(query, top_k=6, sources=None, kb=None, group=None, tags=None, use_rerank=None):
    """带范围过滤的检索。"""
    flt = build_filter(sources=sources, kb=kb, group=group, tags=tags)
    return hybrid_hits(query, limit=top_k, flt=flt, use_rerank=use_rerank)


def collection_count():
    try:
        return qdrant.count(COLLECTION).count
    except Exception:
        return 0


# ============================================================
# 分块级操作（供管理台的「分块可视化 / 编辑」用）
# ============================================================

def _source_filter(source):
    return models.Filter(must=[
        models.FieldCondition(key="source", match=models.MatchValue(value=source))
    ])


def list_chunks(source, limit=200, offset=None):
    pts, nxt = qdrant.scroll(
        collection_name=COLLECTION, scroll_filter=_source_filter(source),
        limit=limit, offset=offset, with_payload=True, with_vectors=False,
    )
    pts = sorted(pts, key=lambda x: (x.payload or {}).get("chunk", 0))
    return pts, nxt


def count_chunks(source):
    return qdrant.count(collection_name=COLLECTION, count_filter=_source_filter(source)).count


def get_point(point_id):
    res = qdrant.retrieve(collection_name=COLLECTION, ids=[point_id], with_payload=True, with_vectors=False)
    return res[0] if res else None


def update_chunk(point_id, text):
    """改分块的文本 -> 重新嵌入 -> 覆盖写入（id 不变，payload 其余字段保留）。"""
    old = get_point(point_id)
    if old is None:
        raise ValueError("分块不存在")
    payload = dict(old.payload or {})
    payload["text"] = text
    payload["edited"] = True
    import time as _t
    payload["edited_at"] = int(_t.time())
    # 重新生成上下文（如果模块可用），保证嵌入文本与原文一致
    embed_src = text
    try:
        from contextual import compose_text
        embed_src = compose_text(
            text,
            section=payload.get("section"),
            fname=payload.get("source"),
            doc_summary=payload.get("doc_summary"),
            chunk_context=payload.get("context") or None,
        )
    except Exception:
        pass
    payload["embed_text"] = embed_src
    vec = embed_documents([embed_src])[0]
    qdrant.upsert(collection_name=COLLECTION, points=[
        models.PointStruct(id=point_id, vector=vec, payload=payload)])
    return payload


def delete_chunks(point_ids):
    if not point_ids:
        return 0
    qdrant.delete(collection_name=COLLECTION, points_selector=models.PointIdsList(points=list(point_ids)))
    return len(point_ids)


def add_chunk(source, text, chunk_index=None, page=None, meta=None):
    """在某个文档下新增一个分块（序号不传则自动取 max+1）。"""
    import uuid as _uuid
    if chunk_index is None:
        pts, _ = list_chunks(source, limit=1000)
        chunk_index = max([(p.payload or {}).get("chunk", -1) for p in pts], default=-1) + 1
    meta = meta or {}
    payload = {
        "text": text,
        "source": source,
        "chunk": int(chunk_index),
        "chunks_total": count_chunks(source) + 1,
        "extract": meta.get("extract", "manual"),
        "manual": True,
        "section": meta.get("section", ""),
        "kb": meta.get("kb", ""),
        "group": meta.get("group", ""),
        "tags": meta.get("tags", []),
    }
    if page is not None:
        payload["page"] = int(page)
    embed_src = text
    try:
        from contextual import compose_text
        embed_src = compose_text(
            text,
            section=payload.get("section"),
            fname=payload.get("source"),
            doc_summary=payload.get("doc_summary"),
            chunk_context=payload.get("context") or None,
        )
    except Exception:
        pass
    payload["embed_text"] = embed_src
    vec = embed_documents([embed_src])[0]
    pid = str(_uuid.uuid5(_uuid.NAMESPACE_URL, f"{source}#{chunk_index}"))
    qdrant.upsert(collection_name=COLLECTION, points=[
        models.PointStruct(id=pid, vector=vec, payload=payload)])
    return pid, payload


def refresh_chunks_total(source):
    pts, _ = list_chunks(source, limit=2000)
    if not pts:
        return 0
    total = len(pts)
    for i, p in enumerate(pts):
        qdrant.set_payload(collection_name=COLLECTION,
                           payload={"chunks_total": total, "chunk": i}, points=[p.id])
    return total


def delete_document(source):
    return delete_chunks_by_source(source)


def delete_by_source(source):
    """兼容旧名（index.py 在用）。"""
    return delete_chunks_by_source(source)


def delete_chunks_by_source(source):
    n = count_chunks(source)
    qdrant.delete(collection_name=COLLECTION,
                  points_selector=models.FilterSelector(filter=_source_filter(source)))
    return n


def refresh_file_meta(source, kb=None, group=None, tags=None):
    """把某个文件所有分块的 kb/group/tags 批量刷新（改归类/改标签后调用）。"""
    payload = {}
    if kb is not None:
        payload["kb"] = kb
    if group is not None:
        payload["group"] = group
    if tags is not None:
        payload["tags"] = list(tags)
    if not payload:
        return 0
    qdrant.set_payload(collection_name=COLLECTION, payload=payload,
                       points=models.FilterSelector(filter=_source_filter(source)))
    return 1


def all_sources():
    seen, offset = set(), None
    while True:
        pts, offset = qdrant.scroll(collection_name=COLLECTION, limit=1000, offset=offset,
                                    with_payload=True, with_vectors=False)
        for p in pts:
            s = (p.payload or {}).get("source")
            if s:
                seen.add(s)
        if offset is None:
            break
    return sorted(seen)