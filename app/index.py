"""索引脚本：把 /docs 里的文档解析、结构化分块、生成上下文、嵌入、写进 Qdrant。

用法（服务器上）：
    docker compose --profile tools run --rm indexer

v2 改造（2026-10-28）：
  1. 结构化分块（标题/条款/表格边界），并记录 section 标题路径；
  2. Contextual Retrieval：每块调 LLM 生成背景说明，前置进嵌入文本；
  3. payload 下沉 section / kb / group / tags，支持向量级过滤；
  4. 并发调用 LLM（默认 4 路），650 块约几分钟。

增量：按「文件修改时间 + 大小」做指纹，没变的文件直接跳过。
重跑安全：改过的文件先删旧向量再写新向量。
进度上报：每处理一个阶段就打印一行 @@PROGRESS@@{...}，管理台解析这一行渲染进度条。
"""
import json
import pathlib
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from common import (
    COLLECTION,
    DOCS_DIR,
    STATE_FILE,
    chunk_document,
    collection_count,
    delete_by_source,
    embed_documents,
    ensure_collection,
    qdrant,
)
from contextual import MODE as CTX_MODE
from contextual import available as llm_available
from contextual import build_context, build_doc_summary, compose_text
from extract import extract_text
from llm import LLM_SEMAPHORE

SUPPORTED = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
    ".md", ".txt", ".csv", ".html", ".htm",
}

PAGE_RE = re.compile(r"\[第(\d+)页\]")
CTX_WORKERS = int(__import__("os").getenv("CTX_WORKERS", "4"))
# 索引完成后，给「还没归类」的文档跑一次 AI 分级。
# 结果只写进候选池（candidates.json），**不直接改归类** —— 等人在管理台点「采纳」。
# 1=开（默认），0=关。
AUTO_CLASSIFY = __import__("os").getenv("AUTO_CLASSIFY", "1") not in ("0", "false", "False")

STATE_DIR = pathlib.Path(STATE_FILE).parent
KBMETA_FILE = STATE_DIR / "kbmeta.json"
CAND_FILE = STATE_DIR / "candidates.json"
TAGS_FILE = STATE_DIR / "tags.json"
DEFAULT_KB = "默认知识库"
UNGROUPED = "未分组"

# ---------------- 进度状态（供管理台轮询） ----------------
# 进度同时写两处：
#   1) stdout 的 @@PROGRESS@@ 行（手动触发时 server.py 直接读）
#   2) data/state/progress.json（共享文件，cron 后台跑时管理台靠它）
PROGRESS_FILE = STATE_DIR / "progress.json"
TRIGGER = __import__("os").getenv("INDEX_TRIGGER", "manual")

# 单文档的处理步骤（与界面的步骤条一一对应）
STEPS = ["parse", "chunk", "context", "embed", "store", "classify"]
STEP_NAME = {
    "parse": "解析文档",
    "chunk": "结构分块",
    "context": "生成上下文",
    "embed": "向量化",
    "store": "写入向量库",
    "classify": "AI 归档建议",
}

# ---- 阶段权重（实测校准：embed 占 87.6%，是绝对大头）----
STEP_WEIGHT = {
    "parse": 0.02,
    "chunk": 0.001,
    "context": 0.03,
    "embed": 0.87,
    "store": 0.001,
    "classify": 0.08,
}
# 扫描件 OCR 时 parse 会占 60%~85%，按这个权重表临时替换
STEP_WEIGHT_OCR = {
    "parse": 0.72,
    "chunk": 0.005,
    "context": 0.02,
    "embed": 0.20,
    "store": 0.005,
    "classify": 0.05,
}

# ---- 耗时基线（持续学习，越跑越准）----
TIMING_FILE = STATE_DIR / "timing.json"
TIMING_DEFAULTS = {
    "per_chunk_seconds": 14.4,      # 每块嵌入
    "per_doc_overhead": 35.0,       # 每文档/每批固定开销
    "per_ocr_page_seconds": 16.5,   # OCR 每页
    "avg_chunks_per_doc": 20.0,     # 用于估算未知文档的块数
    "classify_seconds": 36.0,
    "samples": {"docs": 0, "chunks": 0},
}


def _load_timing():
    try:
        d = load_json(TIMING_FILE) or {}
    except Exception:
        d = {}
    out = dict(TIMING_DEFAULTS)
    for k, v in (d or {}).items():
        if k in out and isinstance(out[k], (int, float)):
            try:
                out[k] = float(v)
            except Exception:
                pass
    out["samples"] = (d or {}).get("samples") or dict(TIMING_DEFAULTS["samples"])
    return out


def _save_timing(d):
    try:
        TIMING_FILE.parent.mkdir(parents=True, exist_ok=True)
        TIMING_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _learn_timing(chunks, seconds, how=""):
    """用刚跑完的这个文档更新基线（指数滑动平均，新样本权重 0.3）。"""
    try:
        cur = _load_timing()
        a = 0.3
        if chunks and seconds:
            per_chunk = max(0.5, (seconds - cur["per_doc_overhead"]) / chunks)
            cur["per_chunk_seconds"] = round((1 - a) * cur["per_chunk_seconds"] + a * per_chunk, 2)
            cur["avg_chunks_per_doc"] = round(
                (1 - a) * cur["avg_chunks_per_doc"] + a * chunks, 2)
        s = cur.get("samples") or {}
        s["docs"] = int(s.get("docs", 0)) + 1
        s["chunks"] = int(s.get("chunks", 0)) + int(chunks or 0)
        cur["samples"] = s
        cur["updated_at"] = int(time.time())
        _save_timing(cur)
    except Exception:
        pass


def _eta_seconds(chunks_left, docs_left, classify_left=0, ocr_left=0):
    """剩余秒数估算。"""
    b = TIMING
    return int(
        b["per_chunk_seconds"] * max(0, chunks_left)
        + 14.0 * max(0, docs_left)
        + b["classify_seconds"] * max(0, classify_left)
        + 1100.0 * max(0, ocr_left)
    )


def _active_weight():
    """当前文档用哪套权重（扫描件走 OCR 表）。"""
    return STEP_WEIGHT_OCR if PROG.get("_ocr") else STEP_WEIGHT


def _weighted_pct(step, chunk_done=0, chunk_total=0):
    """按权重算「当前文档完成了多少」0..1。"""
    w = _active_weight()
    acc = 0.0
    for s in STEPS:
        if s == step:
            frac = 0.0
            if s in ("embed", "context") and chunk_total:
                frac = min(1.0, (chunk_done or 0) / chunk_total)
            return acc + w[s] * frac
        acc += w[s]
    return 1.0


TIMING = _load_timing()

PROG = {
    "running": False,
    "trigger": TRIGGER,
    "started_at": 0.0,
    "updated_at": 0.0,
    "pid": 0,
    "total": 0,          # 本次需处理的文档数
    "done": 0,           # 已完成的
    "failed": 0,
    "skipped": 0,        # 指纹未变跳过的
    "total_all": 0,      # docs 目录里的文件总数
    "phase": "start",
    "step": "",          # 当前文档处在 STEPS 的哪一步
    "step_index": 0,     # 1..6
    "step_total": len(STEPS),
    "stage": "",         # 给人看的一句话
    "current": "",       # 当前文档
    "chunks_done": 0,    # 当前文档已处理的块
    "chunks_total": 0,   # 当前文档总块数
    "pct": 0.0,          # 总进度 0..100
    "vectors": 0,
    "chunks_written": 0,
    "items": [],         # 每个文档的状态
    "finished_at": 0.0,
    "returncode": None,
    # v3 新增
    "eta_seconds": None,     # 预计剩余秒数
    "last_beat": 0.0,        # 最后一次「块进度变化」的时间（判卡死用）
    "stall_seconds": 0.0,    # 距上次心跳多久
    "stall_level": "",       # "" | "slow" | "stall"
    "retry_round": 0,        # 当前是第几轮重试（0=首轮）
    "retry_total": 0,        # 计划重试几轮
    "timing": {},            # 当前学到的基线（给界面显示）
    "_ocr": False,           # 当前文档是否走 OCR
    "_last_chunks_done": 0,
}


def _flush_progress():
    """把 PROG 原子写进共享文件，供管理台轮询。"""
    PROG["updated_at"] = time.time()
    try:
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = PROGRESS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(PROG, ensure_ascii=False), encoding="utf-8")
        tmp.replace(PROGRESS_FILE)
    except Exception:
        pass


def _item(name):
    """找到/创建某个文档的进度条目。"""
    for it in PROG["items"]:
        if it["name"] == name:
            return it
    it = {"name": name, "step": "", "step_index": 0, "pct": 0.0,
          "chunks": 0, "chunks_done": 0, "seconds": 0.0, "status": "pending"}
    PROG["items"].append(it)
    return it


def _recalc_pct():
    """总进度（v3）：按真实耗时加权 + embed 按块推进。"""
    total = PROG["total"] or 0
    if not total:
        PROG["pct"] = 0.0
        return
    cur = 0.0
    step = PROG.get("step") or ""
    if step in STEPS:
        cur = _weighted_pct(step, PROG.get("chunks_done") or 0,
                            PROG.get("chunks_total") or 0)
    PROG["pct"] = round(min(100.0, (PROG["done"] + cur) / total * 100), 1)

    # ---- 心跳与卡死判定（自适应：块进度多久没变）----
    cd = PROG.get("chunks_done") or 0
    if cd != PROG.get("_last_chunks_done"):
        PROG["_last_chunks_done"] = cd
        PROG["last_beat"] = time.time()
    beat = PROG.get("last_beat") or PROG.get("started_at") or time.time()
    stall = time.time() - beat
    PROG["stall_seconds"] = round(stall, 1)
    # 阈值随学到的速度自适应：正常一块 b 秒，超过 6 倍算慢、20 倍算卡
    per_chunk = max(3.0, TIMING.get("per_chunk_seconds", 14.4))
    if PROG.get("running"):
        if stall > max(180.0, per_chunk * 20):
            PROG["stall_level"] = "stall"
        elif stall > max(60.0, per_chunk * 6):
            PROG["stall_level"] = "slow"
        else:
            PROG["stall_level"] = ""
    else:
        PROG["stall_level"] = ""

    # ---- ETA ----
    if PROG.get("running"):
        left_docs = max(0, total - PROG["done"])
        # 当前文档还剩多少块
        ct = PROG.get("chunks_total") or 0
        cd2 = PROG.get("chunks_done") or 0
        avg_ch = TIMING.get("avg_chunks_per_doc", 20.0)
        chunks_left = max(0, ct - cd2) + max(0, left_docs - 1) * avg_ch
        classify_left = max(0, left_docs)
        PROG["eta_seconds"] = _eta_seconds(chunks_left, left_docs, classify_left)
    else:
        PROG["eta_seconds"] = None

    PROG["timing"] = {
        "per_chunk_seconds": TIMING.get("per_chunk_seconds"),
        "avg_chunks_per_doc": TIMING.get("avg_chunks_per_doc"),
        "samples": TIMING.get("samples"),
    }


# phase -> 当前文档处在哪一步
PHASE_STEP = {
    "parse": "parse",
    "chunk": "chunk",
    "context": "context", "context_progress": "context",
    "embed": "embed", "embed_progress": "embed",
    "store": "store",
    "classify": "classify",
    "item_done": "done", "item_fail": "failed",
}


def emit(phase, done=0, total=0, current="", **extra):
    payload = {"phase": phase, "done": done, "total": total, "current": current}
    payload.update(extra)

    # --- 维护共享进度状态 ---
    try:
        PROG["phase"] = phase
        PROG["done"] = done
        PROG["total"] = total if phase != "start" else PROG.get("total", 0)
        if current:
            PROG["current"] = current
        if phase == "start":
            PROG["running"] = True
            PROG["started_at"] = time.time()
            PROG["pid"] = __import__("os").getpid()
            PROG["items"] = []
            PROG["done"] = 0
            PROG["failed"] = 0
            PROG["chunks_written"] = 0
        if phase == "scan":
            PROG["total"] = total
            PROG["total_all"] = extra.get("total_all") or PROG.get("total_all")
            PROG["skipped"] = extra.get("skipped") or 0
        if phase in PHASE_STEP:
            st = PHASE_STEP[phase]
            PROG["step"] = st
            PROG["step_index"] = (STEPS.index(st) + 1) if st in STEPS else 0
            PROG["stage"] = extra.get("stage") or STEP_NAME.get(st, phase)
            it = _item(current) if current else None
            if it is not None:
                it["step"] = st
                it["step_index"] = PROG["step_index"]
                it["status"] = "doing"
                if st in ("done", "failed"):
                    it["status"] = "done" if st == "done" else "failed"
                    it["pct"] = 100.0 if st == "done" else it.get("pct", 0.0)
                    if extra.get("chunks"):
                        it["chunks"] = extra["chunks"]
                    if extra.get("seconds"):
                        it["seconds"] = extra["seconds"]
                    PROG["done"] = done
                elif st in STEPS:
                    it["pct"] = round(_weighted_pct(st) * 100, 1)
        if phase in ("embed_progress", "context_progress"):
            PROG["chunks_done"] = (extra.get("chunks_done") or extra.get("chunk_done")
                                   or extra.get("ctx_done") or 0)
            PROG["chunks_total"] = (extra.get("chunks_total")
                                    or extra.get("ctx_total") or 0)
            if extra.get("retrying"):
                # 重试等待期间也算心跳（证明没死）
                PROG["last_beat"] = time.time()
            it = _item(current) if current else None
            if it is not None:
                it["chunks_done"] = PROG["chunks_done"]
                it["chunks_total"] = PROG["chunks_total"]
                it["pct"] = round(_weighted_pct(it.get("step") or "",
                                                PROG["chunks_done"],
                                                PROG["chunks_total"]) * 100, 1)
        if phase == "embed":
            PROG["chunks_total"] = extra.get("chunks") or 0
            PROG["chunks_done"] = 0
        if phase == "item_done":
            it = _item(current) if current else None
            if it is not None:
                it["status"] = "done"
                it["pct"] = 100.0
                it["chunks"] = extra.get("chunks") or it.get("chunks") or 0
                it["seconds"] = extra.get("seconds") or 0
            # 用这个文档的实测数据更新耗时基线（越跑越准）
            try:
                global TIMING
                _learn_timing(extra.get("chunks") or 0,
                              extra.get("seconds") or 0,
                              extra.get("how") or "")
                TIMING = _load_timing()
            except Exception:
                pass
            PROG["chunks_written"] = extra.get("vectors") or PROG.get("chunks_written") or 0
        if phase == "item_fail":
            it = _item(current) if current else None
            if it is not None:
                it["status"] = "failed"
                it["reason"] = extra.get("reason") or ""
            PROG["failed"] = (PROG.get("failed") or 0) + 1
        if phase == "finish":
            PROG["running"] = False
            PROG["step"] = "done"
            PROG["step_index"] = len(STEPS)
            PROG["pct"] = 100.0
            PROG["finished_at"] = time.time()
            PROG["vectors"] = extra.get("vectors") or PROG.get("vectors")
        _recalc_pct()
        _flush_progress()
    except Exception:
        pass

    print("@@PROGRESS@@" + json.dumps(payload, ensure_ascii=False), flush=True)


def load_state():
    p = pathlib.Path(STATE_FILE)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state):
    p = pathlib.Path(STATE_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(p):
    try:
        return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_meta():
    """读世界知识库归类与标签，用于下沉到 payload。"""
    kbmeta = load_json(KBMETA_FILE) or {}
    assign = kbmeta.get("assign") or {}
    tags = load_json(TAGS_FILE) or {}
    return assign, tags


def meta_for(source, assign, tags):
    a = assign.get(source) or {}
    return {
        "kb": a.get("kb") or DEFAULT_KB,
        "group": a.get("group") or UNGROUPED,
        "tags": list(tags.get(source) or []),
    }


def annotate_pages(chunks):
    """给每个块标注它落在第几页（依赖 OCR 时插入的 [第N页] 标记）。"""
    pages, cur = [], None
    for c in chunks:
        found = PAGE_RE.findall(c["text"])
        if found:
            try:
                cur = int(found[-1])
            except Exception:
                pass
        pages.append(cur)
    return pages


def gen_contexts(fname, head, chunks, idx=None, total=None, source=None, doc_summary=""):
    """并发为每个块生成上下文。失败静默降级为空串。"""
    if not llm_available() or CTX_MODE == "off":
        return [""] * len(chunks)

    done = [0]

    def one(c):
        txt = build_context(fname, c.get("section", ""), head, c["text"], doc_summary=doc_summary)
        done[0] += 1
        if idx is not None and source is not None and done[0] % 5 == 0:
            emit("context_progress", idx, total, source,
                 stage=f"生成上下文 {done[0]}/{len(chunks)}", ctx_done=done[0], ctx_total=len(chunks))
        return txt

    with ThreadPoolExecutor(max_workers=CTX_WORKERS) as ex:
        return list(ex.map(one, chunks))


def _load_kbmeta_raw():
    return load_json(KBMETA_FILE) or {}


def _save_kbmeta_raw(d):
    KBMETA_FILE.parent.mkdir(parents=True, exist_ok=True)
    KBMETA_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_tags_raw(d):
    TAGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TAGS_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def maybe_auto_classify(rel, text, assign, tags, idx, total, short):
    """索引时给未归类文档生成【归档建议】，写进候选池等人工确认。

    - 只处理「还没归到分组」的文档，绝不覆盖人工设置
    - 不写 kbmeta、不改向量 payload —— 归类仍由管理台的人工「采纳」触发
    - 任何异常都静默降级，不影响索引主流程
    """
    cur = assign.get(rel) or {}
    if cur.get("group") and cur.get("group") != UNGROUPED:
        return assign, tags                     # 已归类 -> 跳过

    try:
        from classify import classify as _cls
    except Exception as e:
        print(f"    [AI 建议] 模块不可用：{e}", flush=True)
        return assign, tags

    try:
        kbmeta = _load_kbmeta_raw()
        kb_cur = cur.get("kb") or DEFAULT_KB

        groups = set((kbmeta.get("groups") or {}).get(kb_cur) or [])
        for _s, _a in (kbmeta.get("assign") or {}).items():
            if (_a.get("kb") or DEFAULT_KB) == kb_cur and _a.get("group"):
                groups.add(_a["group"])
        kbs = [k["name"] for k in (kbmeta.get("kbs") or [])] or [DEFAULT_KB]

        emit("classify", idx, total, short, stage="生成归档建议")
        print(f"  [AI 建议] {rel}", flush=True)
        res = _cls(rel, text[:2000], sorted(groups), kbs)

        if not (res and (res.get("group") or res.get("kb"))):
            err = (res or {}).get("error") or "无可用结果"
            print(f"    -> 未生成建议（{err}）", flush=True)
            return assign, tags

        # --- 只写候选池 ---
        cands = load_json(CAND_FILE) or {}
        cands[rel] = {
            "kb": res.get("kb") or kb_cur,
            "suggest_kb": res.get("kb") or kb_cur,
            "group": res.get("group"),
            "tags": res.get("tags") or [],
            "confidence": res.get("confidence") or "medium",
            "model": res.get("model"),
            "seconds": res.get("seconds"),
            "attempt": res.get("attempt"),
            "created": int(time.time()),
            "source": rel,
            "origin": "index",          # 来源：索引时自动生成
        }
        CAND_FILE.parent.mkdir(parents=True, exist_ok=True)
        CAND_FILE.write_text(json.dumps(cands, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"    -> 建议: {cands[rel]['kb']} / {cands[rel]['group'] or '不分类'}"
              f"  标签: {','.join(cands[rel]['tags'])}"
              f"  ({cands[rel]['confidence']})  ← 待人工确认", flush=True)
    except Exception as e:
        print(f"    -> 生成建议异常（已忽略）：{type(e).__name__}: {e}", flush=True)
    return assign, tags


def models_point(rel, idx, vec, payload):
    from qdrant_client import models
    return models.PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{rel}#{idx}")),
        vector=vec,
        payload=payload,
    )


def process_one(item, idx, total, state, ctx):
    """处理单个文档。

    返回 (ok, reason, chunks, seconds)。
    副作用：写 state、写 PROG、调 emit、可能更新 ctx["assign"]/["tags"]。
    任何异常都被捕获成人话，不会中断整批。
    """
    path, rel, st, sig = item
    short = rel if len(rel) <= 46 else "…" + rel[-44:]
    PROG["_last_chunks_done"] = 0
    PROG["_ocr"] = False
    PROG["last_beat"] = time.time()

    emit("parse", idx, total, short, stage="解析文档")
    print(f"  [解析] {rel}", flush=True)
    t0 = time.time()

    try:
        text, how = extract_text(path)
    except Exception as e:
        from extract import human_error
        msg = human_error(e, path.suffix)
        print(f"  [解析异常] {rel}  -> {msg}", flush=True)
        emit("item_fail", idx + 1, total, short, reason=msg)
        return False, msg, 0, round(time.time() - t0, 1)

    if not text:
        print(f"  [解析失败] {rel}  -> {how}", flush=True)
        emit("item_fail", idx + 1, total, short, reason=how)
        return False, how, 0, round(time.time() - t0, 1)

    PROG["_ocr"] = bool(how and str(how).lower().startswith("vlm-ocr"))
    PROG["_last_chunks_done"] = 0
    PROG["last_beat"] = time.time()

    emit("chunk", idx, total, short, stage="结构分块")
    chunks = chunk_document(text)
    emit("chunk", idx, total, short,
         stage=f"结构分块 {len(chunks)} 块", chunks=len(chunks))
    if not chunks:
        msg = "解析出来是空的（可能是纯图片或空文件）"
        print(f"  [无内容]   {rel}", flush=True)
        emit("item_fail", idx + 1, total, short, reason=msg)
        return False, msg, 0, round(time.time() - t0, 1)

    fname = rel.split("/")[-1]
    head = text[:2000]

    # ---- 一级：文档级摘要 ----
    doc_summary = ""
    if CTX_MODE in ("doc", "both") and llm_available():
        emit("context", idx, total, short, stage="生成文档摘要")
        print(f"  [文档摘要] {rel}", flush=True)
        try:
            doc_summary = build_doc_summary(fname, text)
        except Exception as e:
            doc_summary = ""
            print(f"    -> 摘要失败（忽略）：{type(e).__name__}", flush=True)
        else:
            print(f"    -> {doc_summary[:80] if doc_summary else '（无摘要）'}", flush=True)
    elif not llm_available():
        print("    [跳过上下文] 未配置 LLM", flush=True)

    # ---- 二/三级：块级上下文 ----
    if CTX_MODE in ("chunk", "both") and llm_available():
        emit("context", idx, total, short, stage="生成块级上下文", chunks=len(chunks))
        try:
            ctxs = gen_contexts(fname, head, chunks, idx, total, short,
                                doc_summary=doc_summary)
        except Exception as e:
            ctxs = [""] * len(chunks)
            print(f"    [块上下文失败] {type(e).__name__}", flush=True)
    else:
        ctxs = [""] * len(chunks)

    embed_texts = [
        compose_text(ch["text"], section=ch.get("section"), fname=fname,
                     doc_summary=doc_summary, chunk_context=ctxs[i] or None)
        for i, ch in enumerate(chunks)
    ]

    emit("embed", idx, total, short, stage=f"向量化 {len(chunks)} 块",
         chunks=len(chunks), seconds=round(time.time() - t0, 1))
    print(f"  [向量化] {rel}  -> {len(chunks)} 块", flush=True)

    try:
        vectors = embed_documents(
            embed_texts,
            on_progress=lambda d, tt, a=0: emit(
                "embed_progress", idx, total, short,
                stage=("向量化重试中 " if a else "向量化 ") + f"{d}/{tt} 块",
                chunks_total=tt, chunks_done=d,
                chunk_done=d, retrying=bool(a)))
    except Exception as e:
        msg = "向量化失败（嵌入服务无响应，通常是并发抢资源）"
        print(f"  [嵌入失败] {rel}  -> {str(e)[:120]}", flush=True)
        emit("item_fail", idx + 1, total, short, reason=msg)
        return False, msg, 0, round(time.time() - t0, 1)

    delete_by_source(rel)

    # ---- AI 归档建议（只进候选池）----
    if AUTO_CLASSIFY:
        try:
            a2, tg2 = maybe_auto_classify(rel, text, ctx["assign"], ctx["tags"],
                                          idx, total, short)
            ctx["assign"], ctx["tags"] = a2, tg2
        except Exception as e:
            print(f"    [AI 建议失败] {type(e).__name__}", flush=True)

    pages = annotate_pages(chunks)
    m = meta_for(rel, ctx["assign"], ctx["tags"])
    points = []
    for i, vec in enumerate(vectors):
        payload = {
            "text": chunks[i]["text"],
            "embed_text": embed_texts[i],
            "context": ctxs[i] or "",
            "section": chunks[i].get("section", ""),
            "kinds": chunks[i].get("kinds", []),
            "source": rel,
            "chunk": i,
            "chunks_total": len(chunks),
            "mtime": int(st.st_mtime),
            "extract": how,
            "kb": m["kb"],
            "group": m["group"],
            "tags": m["tags"],
            "doc_summary": doc_summary,
        }
        if pages[i] is not None:
            payload["page"] = pages[i]
        points.append(models_point(rel, i, vec, payload))

    emit("store", idx, total, short,
         stage=f"写入向量库 {len(points)} 块", chunks=len(points))
    try:
        qdrant.upsert(collection_name=COLLECTION, points=points)
    except Exception as e:
        msg = "写入向量库失败：" + str(e)[:60]
        print(f"  [入库失败] {rel}  -> {e}", flush=True)
        emit("item_fail", idx + 1, total, short, reason=msg)
        return False, msg, 0, round(time.time() - t0, 1)

    state[rel] = {
        "sig": sig,
        "v": 2,
        "chunks": len(chunks),
        "indexed_at": int(time.time()),
        "extract": how,
        "with_context": sum(1 for x in ctxs if x),
        "doc_summary": doc_summary[:200],
    }
    save_state(state)

    elapsed = round(time.time() - t0, 1)
    emit("item_done", idx + 1, total, short,
         chunks=len(chunks), vectors=len(points), seconds=elapsed, how=how)
    print(f"  [已索引] {rel}  -> {len(chunks)} 块  ({how}, {elapsed}s)", flush=True)
    return True, "", len(chunks), elapsed


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DOCS_DIR)
    if not root.exists():
        print(f"目录不存在：{root}")
        sys.exit(1)

    RETRY_ROUNDS = int(__import__("os").getenv("RETRY_ROUNDS", "2"))

    PROG["total_all"] = 0
    emit("start", 0, 0, "初始化")
    ensure_collection()
    state = load_state()
    assign, tags = load_meta()
    ctx = {"assign": assign, "tags": tags}

    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in SUPPORTED)
    total = len(files)
    print(f"扫描 {root}：发现 {total} 个可索引文件\n", flush=True)
    print(f"LLM 上下文生成：{'开启（%d 并发）' % CTX_WORKERS if llm_available() else '关闭（未配置）'}\n",
          flush=True)

    todo = []
    for p2 in files:
        rel = str(p2.relative_to(root))
        st2 = p2.stat()
        sig = f"{int(st2.st_mtime)}:{st2.st_size}"
        if state.get(rel, {}).get("sig") == sig and state.get(rel, {}).get("v") == 2:
            continue
        todo.append((p2, rel, st2, sig))

    skipped = total - len(todo)
    emit("scan", 0, len(todo),
         f"共 {total} 个文件，需处理 {len(todo)} 个（跳过 {skipped} 个未变化）",
         skipped=skipped, total_all=total)

    if not todo:
        # 没事可做时不要把上一轮的真实结果覆盖掉（cron 每 10 分钟就会跑一次），
        # 保留上次的 added / failed_items / vectors，只更新"检查过"的时间戳。
        prev = {}
        try:
            prev = json.loads(PROGRESS_FILE.read_text(encoding="utf-8")) if PROGRESS_FILE.exists() else {}
        except Exception:
            prev = {}
        keep_failed = prev.get("failed_items") or []
        keep_added = prev.get("added") or prev.get("done") or 0
        keep_vectors = prev.get("vectors") or collection_count()
        keep_chunks = prev.get("chunks_written") or 0
        keep_started = prev.get("started_at") or 0
        keep_finished = prev.get("finished_at") or 0

        PROG["skipped"] = skipped
        PROG["total"] = 0
        emit("idle", 0, 0, f"没有需要处理的文件（跳过 {skipped} 个未变化）",
             stage="无变化，无需处理", skipped=skipped)

        # 覆盖回上一轮的真实结果
        PROG.update(
            running=False, phase="idle", step="done",
            done=keep_added, added=keep_added,
            failed=len(keep_failed), failed_items=keep_failed,
            vectors=keep_vectors, chunks_written=keep_chunks,
            started_at=keep_started, finished_at=keep_finished,
            pct=100.0, eta_seconds=None,
        )
        _flush_progress()
        print(f"完成：新增/更新 0，跳过 {skipped}，集合共 {keep_vectors} 个向量（沿用上一轮统计）")
        return

    added = total_chunks = 0
    failed_items = []

    # ================= 首轮 =================
    for idx, item in enumerate(todo):
        ok, reason, nch, secs = process_one(item, idx, len(todo), state, ctx)
        if ok:
            added += 1
            total_chunks += nch
        else:
            failed_items.append((item, reason))

    # ================= 自动重试 =================
    for rnd in range(1, RETRY_ROUNDS + 1):
        if not failed_items:
            break
        print(f"\n=== 第 {rnd} 轮重试：{len(failed_items)} 个失败文档 ===", flush=True)
        PROG["retry_round"] = rnd
        PROG["retry_total"] = RETRY_ROUNDS
        PROG["total"] = len(failed_items)
        PROG["done"] = 0
        PROG["failed"] = 0
        PROG["items"] = []
        PROG["pct"] = 0.0
        PROG["last_beat"] = time.time()
        emit("retry_start", 0, len(failed_items),
             f"第 {rnd} 轮重试（{len(failed_items)} 个文档）",
             stage=f"第 {rnd}/{RETRY_ROUNDS} 轮重试",
             retry_round=rnd, retry_total=RETRY_ROUNDS)

        still = []
        for idx, (item, _prev) in enumerate(failed_items):
            ok, reason, nch, secs = process_one(item, idx, len(failed_items),
                                                state, ctx)
            if ok:
                added += 1
                total_chunks += nch
            else:
                still.append((item, reason))
        failed_items = still

    # ================= 收尾 =================
    save_state(state)
    count = collection_count()

    # 仍失败的：把文件名 + 人话原因留在进度里，界面据此弹窗/挂红条
    PROG["failed_items"] = [{"name": it[0][1], "reason": it[1]}
                            for it in failed_items]
    PROG["retry_round"] = 0
    PROG["retry_total"] = 0

    n_fail = len(failed_items)
    print(f"\n完成：新增/更新 {added}，跳过 {skipped}，失败 {n_fail}，"
          f"本次写入 {total_chunks} 块，集合共 {count} 个向量", flush=True)
    if n_fail:
        print("失败清单：", flush=True)
        for it, rsn in failed_items:
            print(f"   × {it[1]}  —— {rsn}", flush=True)

    emit("finish", len(todo), len(todo),
         f"完成：新增 {added}，跳过 {skipped}，失败 {n_fail}",
         added=added, skipped=skipped, failed=n_fail, vectors=count)
    PROG["added"] = added
    _flush_progress()


if __name__ == "__main__":
    main()