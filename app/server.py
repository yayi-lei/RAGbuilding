"""知识库 MCP 服务 + 管理页（同一端口 8765）。

MCP 工具（给 DSH 调用）：
    search_kb   语义检索，返回带出处的原文片段
    kb_stats    库状态

管理页：
    GET    /                    页面
    GET    /api/status          库状态
    GET    /api/files           docs 目录文件 + 索引状态
    POST   /api/upload          上传文件
    DELETE /api/files/{name}    删除文件
    POST   /api/index           触发增量索引
    GET    /api/index           索引进度与日志
    POST   /api/search          检索测试

鉴权：除首页 HTML 外，其余路径都要 Authorization: Bearer <KB_TOKEN>。
"""
import fcntl
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

import uvicorn
import httpx
from starlette.concurrency import run_in_threadpool
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.routing import Route

import classify as _classify

from common import (
    COLLECTION,
    DOCS_DIR,
    STATE_FILE,
    add_chunk,
    all_sources,
    collection_count,
    count_chunks,
    delete_chunks,
    delete_chunks_by_source,
    ensure_collection,
    list_chunks,
    qdrant,
    refresh_chunks_total,
    refresh_file_meta,
    search,
    search_in,
    update_chunk,
)

MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8765"))
KB_TOKEN = os.getenv("KB_TOKEN", "")

PROGRESS_PREFIX = "@@PROGRESS@@"
# 索引进程把进度写在这里；cron 后台跑的也能被管理台看到
PROGRESS_FILE = str(pathlib.Path(STATE_FILE).parent / "progress.json")
# 索引全局串行锁：cron 用 flock 锁同一个文件，server 用 fcntl 锁它。
# 避免"定时任务 + 手动触发"同时跑，抢 llama-server 导致嵌入全线失败。
INDEX_LOCK_FILE = str(pathlib.Path(STATE_FILE).parent / "index.lock")


def _acquire_index_lock():
    """非阻塞获取索引锁。成功返回 fd，失败返回 None。"""
    try:
        fd = os.open(INDEX_LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o666)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except (OSError, IOError):
        return None


def _release_index_lock(fd):
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        os.close(fd)
    except Exception:
        pass


def _index_running_elsewhere():
    """别处（cron 进程）是不是正握着锁。"""
    fd = _acquire_index_lock()
    if fd is None:
        return True
    _release_index_lock(fd)
    return False
LOCK_FILE = str(pathlib.Path(STATE_FILE).parent / "index.lock")

# 网页登录用的账号（从 .env 读；留空则禁用网页登录，退回 token 模式）
KB_ADMIN_USER = os.getenv("KB_ADMIN_USER", "admin")
KB_ADMIN_PASS = os.getenv("KB_ADMIN_PASS", "")

SUPPORTED = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
    ".md", ".txt", ".csv", ".html", ".htm",
}

# ---------------------------------------------------------------- MCP 工具

# stateless_http=True：无会话模式。容器重建后客户端的旧 session-id 直接失效会被
# 服务端忽略而不是报 "Session not found"，DSH 侧无需任何重连动作。
mcp = FastMCP("office-kb", host=MCP_HOST, port=MCP_PORT, stateless_http=True)


@mcp.tool()
async def search_kb(query: str, top_k: int = 6) -> str:
    """在本地文档知识库里做语义检索，返回最相关的原文片段。

    用于回答关于历史项目文档、招标文件、合同、方案、评审材料的问题。
    返回内容包含【出处】和原文，请基于原文作答，并标注出处；
    如果检索结果不足以回答，请明确说明「知识库里没有找到」，不要自行编造。

    Args:
        query: 检索用的自然语言问题，用中文描述你要找什么。
        top_k: 返回的片段数量，默认 6，最多 15。
    """
    top_k = max(1, min(int(top_k), 15))

    def _do():
        ensure_collection()
        return search(query, top_k=top_k)

    try:
        # 同上：同步检索必须在线程池里跑，避免堵住 MCP 的事件循环
        hits = await run_in_threadpool(_do)
    except Exception as e:
        return f"检索失败：{e}"

    if not hits:
        return "知识库里没有找到相关内容。库可能是空的，或者还没索引文档。"

    parts = [f"共检索到 {len(hits)} 个片段：\n"]
    for i, h in enumerate(hits, 1):
        p = h.payload or {}
        source = p.get("source", "未知文件")
        chunk = p.get("chunk", "?")
        page = p.get("page")
        where = f"第 {page} 页" if page else f"第 {chunk} 块"
        sec = p.get("section")
        ctx = p.get("context") or p.get("doc_summary")
        score = getattr(h, "score", None)
        score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "?"
        head = f"--- 片段 {i} ---\n【出处】{source}（{where}）\n"
        if sec:
            head += f"【章节】{sec}\n"
        if ctx:
            head += f"【背景】{ctx}\n"
        head += f"【相关度】{score_s}\n【原文】\n{p.get('text', '').strip()}\n"
        parts.append(head)
    return "\n".join(parts)


@mcp.tool()
async def kb_stats() -> str:
    """查看本地知识库的规模：向量总数、集合名、已被索引的文件数、最近索引时间。"""
    def _count():
        ensure_collection()
        return collection_count()

    try:
        vec_count = await run_in_threadpool(_count)
    except Exception as e:
        return f"无法连接向量库：{e}"

    files = _scan_docs()
    indexed = [f for f in files if f["indexed"]]
    newest = "无记录"
    try:
        state = _load_state()
        if state:
            ts = max(v.get("indexed_at", 0) for v in state.values())
            if ts:
                newest = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    except Exception:
        pass

    return (
        f"集合名：{COLLECTION}\n"
        f"向量总数：{vec_count}\n"
        f"文档目录：{DOCS_DIR}（{len(files)} 个文件，已索引 {len(indexed)}）\n"
        f"待索引：{len(files) - len(indexed)}\n"
        f"最近一次索引：{newest}\n"
        f"检索方式：混合（稠密向量 + BM25 关键词，RRF 融合）"
        f"{' + LLM 重排' if os.getenv('RERANK_ENABLED', '1') not in ('0', 'false', 'False') else ''}\n"
        f"上下文增强：{'已开启' if os.getenv('VLM_API_KEY') else '未配置'}"
        f"（Contextual Retrieval，为每块生成背景说明）\n"
        f"扫描件 OCR：{'已配置' if os.getenv('VLM_API_KEY') else '未配置（无 VLM_API_KEY）'}"
    )


# ---------------------------------------------------------------- 状态与工具

JOB = {"running": False, "log": "", "started_at": 0, "finished_at": 0, "returncode": None, "progress": {}}
JOB_LOCK = threading.Lock()
APP_DIR = pathlib.Path(__file__).resolve().parent


def _load_state():
    p = pathlib.Path(STATE_FILE)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _scan_docs():
    root = pathlib.Path(DOCS_DIR)
    state = _load_state()
    out = []
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED:
            continue
        rel = str(p.relative_to(root))
        st = p.stat()
        sig = f"{int(st.st_mtime)}:{st.st_size}"
        rec = state.get(rel)
        out.append({
            "name": rel,
            "size": st.st_size,
            "mtime": int(st.st_mtime),
            "indexed": bool(rec) and rec.get("sig") == sig,
            "chunks": (rec or {}).get("chunks", 0),
            "extract": (rec or {}).get("extract", ""),
        })
    return out


def _index_worker(backend="auto"):
    # 与 cron 共用同一把文件锁，防止两个索引同时跑
    try:
        _lock = open(LOCK_FILE, "w")
        fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        with JOB_LOCK:
            JOB.update(running=False, returncode=-2,
                       log="另一个索引任务正在运行中（cron 或手动），本次跳过。\n")
        print("跳过：索引锁被占用", flush=True)
        return
    except Exception as e:
        print(f"锁异常（继续执行）: {e}", flush=True)
        _lock = None

    with JOB_LOCK:
        JOB.update(running=True, log="", started_at=time.time(), finished_at=0, returncode=None, progress={})

    # 手动选择嵌入通道：只影响本次索引子进程，cron 走的还是默认路径（本地优先、CF 兜底）
    child_env = dict(os.environ)
    suffix = {"local": "/local", "cloudflare": "/cf"}.get(backend, "")
    if suffix:
        base = (child_env.get("OLLAMA_URL") or "").rstrip("/")
        if base and not base.endswith(("/local", "/cf")):
            child_env["OLLAMA_URL"] = base + suffix
    with JOB_LOCK:
        JOB["log"] += f"[嵌入通道] {backend}" + (f" → {child_env.get('OLLAMA_URL')}" if suffix else "") + "\n"

    rc = -1
    try:
        proc = subprocess.Popen(
            [sys.executable, "index.py", DOCS_DIR],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=str(APP_DIR), env=child_env,
        )
        for line in proc.stdout:
            if PROGRESS_PREFIX in line:
                try:
                    obj = json.loads(line.split(PROGRESS_PREFIX, 1)[1].strip())
                    with JOB_LOCK:
                        JOB["progress"] = obj
                except Exception:
                    pass
                continue          # 进度行不入日志，保持日志干净
            with JOB_LOCK:
                JOB["log"] += line
        rc = proc.wait()
    except Exception as e:
        with JOB_LOCK:
            JOB["log"] += f"\n[异常] {e}\n"
    with JOB_LOCK:
        JOB.update(running=False, finished_at=time.time(), returncode=rc)


# ---------------------------------------------------------------- HTTP 端点

# ---------------------------------------------------------------- 标签 / 分组

TAGS_FILE = str(pathlib.Path(STATE_FILE).parent / "tags.json")


def _load_tags():
    p = pathlib.Path(TAGS_FILE)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_tags(d):
    p = pathlib.Path(TAGS_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


async def api_tags_get(request):
    """返回 {source: [标签...]} 全量映射 + 所有出现过的标签。"""
    d = _load_tags()
    allt = sorted({t for v in d.values() for t in (v or [])})
    return JSONResponse({"map": d, "all": allt})


async def api_tags_set(request):
    """给一批文件设置标签：{sources:[...], tags:[...], mode:"replace"|"add"|"remove"}"""
    body = await request.json()
    sources = body.get("sources") or []
    tags = [t.strip() for t in (body.get("tags") or []) if t and t.strip()]
    mode = body.get("mode") or "add"
    if not sources:
        return JSONResponse({"error": "sources 为空"}, status_code=400)
    d = _load_tags()
    for s in sources:
        cur = set(d.get(s) or [])
        if mode == "replace":
            cur = set(tags)
        elif mode == "remove":
            cur -= set(tags)
        else:
            cur |= set(tags)
        if cur:
            d[s] = sorted(cur)
        else:
            d.pop(s, None)
    _save_tags(d)
    # 标签下沉到向量 payload，检索时才能做向量级过滤
    for s in sources:
        try:
            refresh_file_meta(s, tags=list(d.get(s) or []))
        except Exception:
            pass
    return JSONResponse({"ok": True, "map": d})


async def api_tree(request):
    """把 docs/ 的目录结构返回给前端（含每个文件是否已索引）。"""
    root = pathlib.Path(DOCS_DIR)
    files = _scan_docs()
    idx = {f["name"]: f for f in files}
    tree = {}
    for f in files:
        parts = f["name"].split("/")
        cur = tree
        for seg in parts[:-1]:
            cur = cur.setdefault(seg, {"_dir": True, "_children": {}})["_children"]
        cur[parts[-1]] = {"_dir": False, "_file": f}
    def conv(node, prefix=""):
        out = []
        for name, v in sorted(node.items()):
            path = prefix + name
            if v.get("_dir"):
                out.append({"name": name, "key": path, "path": path, "dir": True,
                            "children": conv(v["_children"], path + "/")})
            else:
                out.append({"name": name, "key": path, "path": path, "dir": False,
                            "info": v["_file"], "size": v["_file"].get("size", 0),
                            "indexed": v["_file"].get("indexed", False)})
        return out
    return JSONResponse({"tree": conv(tree), "total": len(files)})


# ---------------------------------------------------------------- 分块管理

async def api_chunks_list(request):
    src = request.query_params.get("source")
    if not src:
        return JSONResponse({"error": "缺少 source"}, status_code=400)
    try:
        pts, _ = list_chunks(src, limit=2000)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    total = len(pts)
    items = []
    for p in pts:
        pay = p.payload or {}
        items.append({
            "id": str(p.id),
            "chunk": pay.get("chunk"),
            "text": pay.get("text", ""),
            "page": pay.get("page"),
            "extract": pay.get("extract"),
            "manual": bool(pay.get("manual")),
            "edited": bool(pay.get("edited")),
            "len": len(pay.get("text") or ""),
            "section": pay.get("section") or "",
            "context": pay.get("context") or "",
            "kinds": pay.get("kinds") or [],
            "kb": pay.get("kb") or "",
            "group": pay.get("group") or "",
            "tags": pay.get("tags") or [],
        })
    return JSONResponse({"source": src, "total": total, "chunks": items})


async def api_chunk_update(request):
    """改某一块的文本 -> 重新嵌入（约 8~12 秒）。"""
    pid = request.path_params["pid"]
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "文本不能为空"}, status_code=400)
    try:
        pay = update_chunk(pid, text)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    try:
        refresh_chunks_total(pay.get("source"))
    except Exception:
        pass
    return JSONResponse({"ok": True, "payload": pay})


async def api_chunk_delete(request):
    pid = request.path_params["pid"]
    src = request.query_params.get("source")
    try:
        pt = qdrant.retrieve(collection_name=COLLECTION, ids=[pid], with_payload=True, with_vectors=False)
        if not pt:
            return JSONResponse({"error": "分块不存在"}, status_code=404)
        source = (pt[0].payload or {}).get("source") or src
        delete_chunks([pid])
        if source:
            refresh_chunks_total(source)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"ok": True})


async def api_chunk_add(request):
    body = await request.json()
    src = (body.get("source") or "").strip()
    text = (body.get("text") or "").strip()
    if not src or not text:
        return JSONResponse({"error": "source / text 不能为空"}, status_code=400)
    try:
        pid, pay = add_chunk(src, text, page=body.get("page"))
        refresh_chunks_total(src)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"ok": True, "id": pid, "payload": pay})


# ---------------------------------------------------------------- 批量操作

async def api_files_batch(request):
    """批量操作：{action:"delete"|"reindex", sources:[...]}"""
    body = await request.json()
    action = body.get("action")
    names = [str(x) for x in (body.get("sources") or [])]
    if not names:
        return JSONResponse({"error": "sources 为空"}, status_code=400)

    root = pathlib.Path(DOCS_DIR)
    state = _load_state()
    done, failed = [], []

    for name in names:
        try:
            if action == "delete":
                p = root / name
                if p.exists() and p.is_file():
                    p.unlink()
                try:
                    delete_chunks_by_source(name)
                except Exception:
                    pass
                state.pop(name, None)
                done.append(name)
            elif action == "reindex":
                state.pop(name, None)          # 抹掉指纹 -> 下次索引会重跑
                try:
                    delete_chunks_by_source(name)   # 顺便清掉旧向量，避免残留
                except Exception:
                    pass
                done.append(name)
            else:
                failed.append({"name": name, "error": f"未知 action: {action}"})
        except Exception as e:
            failed.append({"name": name, "error": str(e)})

    _save_state_local(state)

    # 标签里也清掉
    if action == "delete":
        tg = _load_tags()
        for n in done:
            tg.pop(n, None)
        _save_tags(tg)

    return JSONResponse({"ok": True, "done": done, "failed": failed})


def _save_state_local(state):
    p = pathlib.Path(STATE_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ================================================================
# 知识库 / 分组（元数据，不碰文件和向量）
#   结构：kbmeta.json = { "kbs":[{"name":..,"desc":..,"created":..}],
#                        "assign":{"<source>":{"kb":"..","group":".."}} }
#   设计参考 dsh-knowledge：分组是库内元数据；未指定时落「未分组」；
#   归类/取消归类都不动文件、正文和向量。
# ================================================================

KBMETA_FILE = str(pathlib.Path(STATE_FILE).parent / "kbmeta.json")
DEFAULT_KB = "默认知识库"
UNGROUPED = "未分组"
GROUP_SEP = "/"


def _load_kbmeta():
    p = pathlib.Path(KBMETA_FILE)
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                d.setdefault("kbs", [])
                d.setdefault("assign", {})
                d.setdefault("groups", {})     # {kbName: [显式声明的分组路径]}（含空分组）
                return d
        except Exception:
            pass
    return {"kbs": [], "assign": {}, "groups": {}}


def _save_kbmeta(d):
    p = pathlib.Path(KBMETA_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def _refresh_payloads(d=None):
    """把 kbmeta/tags 的最新状态整体刷进 Qdrant payload。

    在「知识库改名/删除、分组改名/删除、采纳 AI 候选」这类直接改元数据的
    操作之后调用，保证向量级过滤（kb/group/tags）不落伍。只改 payload，
    不重新嵌入，开销很小。
    """
    try:
        d = d if d is not None else _ensure_default_kb(_load_kbmeta())
        tg = _load_tags()
        sources = set((d.get("assign") or {}).keys()) | set(tg.keys())
        for src in sources:
            kb, grp = _meta_of(d, src)
            try:
                refresh_file_meta(src, kb=kb, group=grp, tags=list(tg.get(src) or []))
            except Exception:
                pass
    except Exception:
        pass


def _norm_name(s, limit=64):
    """规范化分组/知识库名：去首尾空白、压缩多空格、全角转半角空格、限长。"""
    s = (s or "").replace("\u3000", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    s = "/".join(part.strip() for part in s.split("/") if part.strip())
    return s[:limit]


def _ensure_default_kb(d):
    if not d.get("kbs"):
        d["kbs"] = [{"name": DEFAULT_KB, "desc": "系统默认知识库（未归类的文档都在这里）",
                     "created": int(time.time())}]
    return d


def _find_kb(d, name):
    for k in d.get("kbs") or []:
        if k["name"] == name:
            return k
    return None


def _meta_of(d, source):
    a = (d.get("assign") or {}).get(source) or {}
    kb = a.get("kb") or DEFAULT_KB
    grp = a.get("group") or UNGROUPED
    return kb, grp


def _groups_in_kb(d, kb):
    """该库下所有分组路径 = 显式声明的（含空分组） ∪ 从文件推导的。"""
    out = set((d.get("groups") or {}).get(kb) or [])
    for src, a in (d.get("assign") or {}).items():
        if (a.get("kb") or DEFAULT_KB) != kb:
            continue
        g = a.get("group") or UNGROUPED
        if g and g != UNGROUPED:
            out.add(g)
    return sorted(out)


def _set_groups(d, kb, paths):
    d.setdefault("groups", {})[kb] = sorted({p for p in paths if p})


def _sync_groups(d, kb):
    """把「从文件推导的分组」也写进显式列表，避免某组文件全移走后分组消失。"""
    merged = set(_groups_in_kb(d, kb))
    _set_groups(d, kb, merged)


def _build_group_tree(d, files, kb):
    """把分组路径拼成树，并把文件挂到对应节点；未归类的进「未分组」。"""
    nodes = {}          # path -> {"name","path","children":{},"files":[]}

    def node(path):
        if path in nodes:
            return nodes[path]
        name = path.split(GROUP_SEP)[-1] if path else UNGROUPED
        parent = GROUP_SEP.join(path.split(GROUP_SEP)[:-1]) if GROUP_SEP in path else ""
        n = {"name": name or UNGROUPED, "path": path, "children": {}, "files": []}
        nodes[path] = n
        if parent:
            pn = node(parent)
            pn["children"][path] = n
        return n

    roots = {}

    # 先把显式声明的分组（含空分组）建出来
    for gp in _groups_in_kb(d, kb):
        node(gp)
        top = gp.split(GROUP_SEP)[0]
        roots.setdefault(top, node(top))

    for f in files:
        kb_of_f, grp = _meta_of(d, f["name"])
        if kb_of_f != kb:
            continue
        if not grp or grp == UNGROUPED:
            roots.setdefault("__ungrouped__", {"name": UNGROUPED, "path": "",
                                               "children": {}, "files": []})
            roots["__ungrouped__"]["files"].append(f)
            continue
        n = node(grp)
        n["files"].append(f)
        # 挂到根
        top = grp.split(GROUP_SEP)[0]
        roots.setdefault(top, node(top))

    def conv(n):
        return {
            "name": n["name"],
            "path": n["path"],
            "dir": True,
            "files": sorted(n["files"], key=lambda x: x["name"]),
            "children": [conv(c) for c in sorted(n["children"].values(), key=lambda x: x["name"])],
        }

    # 根节点：正常分组在前，未分组在后
    ordered = [conv(n) for k, n in sorted(roots.items()) if k != "__ungrouped__"]
    if "__ungrouped__" in roots:
        ordered.append(conv(roots["__ungrouped__"]))
    return ordered


# ---------------------------------------------------------------- API

async def api_kb_list(request):
    """知识库列表 + 每个库的统计 + 分组清单。"""
    d = _ensure_default_kb(_load_kbmeta())
    _save_kbmeta(d)
    files = _scan_docs()
    stats = {}
    for f in files:
        kb, grp = _meta_of(d, f["name"])
        s = stats.setdefault(kb, {"files": 0, "indexed": 0, "chunks": 0})
        s["files"] += 1
        if f.get("indexed"):
            s["indexed"] += 1
            s["chunks"] += f.get("chunks") or 0
    out = []
    for k in d["kbs"]:
        st = stats.get(k["name"], {"files": 0, "indexed": 0, "chunks": 0})
        out.append({**k, **st, "groups": _groups_in_kb(d, k["name"])})
    return JSONResponse({"kbs": out, "total_files": len(files), "default_kb": DEFAULT_KB,
                         "ungrouped": UNGROUPED})


async def api_kb_tree(request):
    """某个知识库的分组树（含每个分组下的文件）。"""
    kb = request.query_params.get("kb") or DEFAULT_KB
    d = _ensure_default_kb(_load_kbmeta())
    files = _scan_docs()
    return JSONResponse({"kb": kb, "tree": _build_group_tree(d, files, kb)})


async def api_kb_op(request):
    """知识库操作：{action:"create"|"rename"|"delete"|"setdesc", name, newName, desc}"""
    body = await request.json()
    action = body.get("action")
    d = _ensure_default_kb(_load_kbmeta())
    name = _norm_name(body.get("name") or "")
    new = _norm_name(body.get("newName") or "")

    if action == "create":
        if not name:
            return JSONResponse({"error": "名称不能为空"}, status_code=400)
        if _find_kb(d, name):
            return JSONResponse({"error": "同名知识库已存在"}, status_code=400)
        d["kbs"].append({"name": name, "desc": (body.get("desc") or "").strip(),
                         "created": int(time.time())})
    elif action == "rename":
        k = _find_kb(d, name)
        if not k:
            return JSONResponse({"error": "知识库不存在"}, status_code=404)
        if not new:
            return JSONResponse({"error": "新名称不能为空"}, status_code=400)
        if _find_kb(d, new):
            return JSONResponse({"error": "新名称已存在"}, status_code=400)
        k["name"] = new
        for src, a in (d.get("assign") or {}).items():
            if (a.get("kb") or DEFAULT_KB) == name:
                a["kb"] = new
    elif action == "setdesc":
        k = _find_kb(d, name)
        if not k:
            return JSONResponse({"error": "知识库不存在"}, status_code=404)
        k["desc"] = (body.get("desc") or "").strip()
    elif action == "delete":
        if name == DEFAULT_KB:
            return JSONResponse({"error": "默认知识库不能删除"}, status_code=400)
        if not _find_kb(d, name):
            return JSONResponse({"error": "知识库不存在"}, status_code=404)
        d["kbs"] = [x for x in d["kbs"] if x["name"] != name]
        # 该库的文件移到默认库（不动文件本身）
        for src, a in (d.get("assign") or {}).items():
            if (a.get("kb") or DEFAULT_KB) == name:
                a["kb"] = DEFAULT_KB
    else:
        return JSONResponse({"error": f"未知 action: {action}"}, status_code=400)

    _save_kbmeta(d)
    _refresh_payloads(d=d)
    return JSONResponse({"ok": True, "kbs": d["kbs"]})


async def api_kb_assign(request):
    """批量归类：{sources:[...], kb, group}  —— 不动文件、不动向量。"""
    body = await request.json()
    sources = [str(x) for x in (body.get("sources") or [])]
    kb = _norm_name(body.get("kb") or "") or None
    grp = _norm_name(body.get("group") or "")
    if not sources:
        return JSONResponse({"error": "sources 为空"}, status_code=400)

    d = _ensure_default_kb(_load_kbmeta())
    if kb and not _find_kb(d, kb):
        return JSONResponse({"error": f"知识库不存在：{kb}"}, status_code=404)

    changed = []
    for src in sources:
        cur_kb, cur_grp = _meta_of(d, src)
        a = d["assign"].setdefault(src, {})
        a["kb"] = kb or cur_kb
        if grp:
            a["group"] = grp
        elif body.get("clearGroup"):
            a["group"] = UNGROUPED
        else:
            a["group"] = cur_grp
        if a.get("group") == UNGROUPED:
            a.pop("group", None)
        changed.append(src)

    # 归类后把新分组也声明出来（保证空组/父级可见）
    for src in changed:
        k2, g2 = _meta_of(d, src)
        if g2 and g2 != UNGROUPED:
            lst = list((d.get("groups") or {}).get(k2) or [])
            parts = g2.split(GROUP_SEP)
            for i in range(1, len(parts) + 1):
                pre = GROUP_SEP.join(parts[:i])
                if pre not in lst:
                    lst.append(pre)
            _set_groups(d, k2, lst)

    _save_kbmeta(d)
    # 归类结果下沉到向量 payload
    for src in changed:
        k2, g2 = _meta_of(d, src)
        try:
            refresh_file_meta(src, kb=k2, group=g2)
        except Exception:
            pass
    return JSONResponse({"ok": True, "changed": changed})


async def api_kb_group_op(request):
    """分组操作：{action:"rename"|"delete", kb, group, newName}"""
    body = await request.json()
    action = body.get("action")
    kb = _norm_name(body.get("kb") or "") or DEFAULT_KB
    grp = _norm_name(body.get("group") or "")
    new = _norm_name(body.get("newName") or "")
    if not grp:
        return JSONResponse({"error": "group 不能为空"}, status_code=400)

    d = _ensure_default_kb(_load_kbmeta())
    hit = []
    for src, a in (d.get("assign") or {}).items():
        if (a.get("kb") or DEFAULT_KB) == kb and (a.get("group") or UNGROUPED) == grp:
            hit.append((src, a))

    if action == "create":
        g = _norm_name(body.get("group") or "")
        if not g:
            return JSONResponse({"error": "分组名不能为空"}, status_code=400)
        lst = list((d.get("groups") or {}).get(kb) or [])
        if g not in lst:
            lst.append(g)
        # 同时把父级路径也声明出来，保证树能展开
        parts = g.split(GROUP_SEP)
        for i in range(1, len(parts)):
            pre = GROUP_SEP.join(parts[:i])
            if pre not in lst:
                lst.append(pre)
        _set_groups(d, kb, lst)
        _save_kbmeta(d)
        return JSONResponse({"ok": True, "group": g})

    if action == "rename":
        if not new:
            return JSONResponse({"error": "新分组名不能为空"}, status_code=400)
        if grp == new:
            return JSONResponse({"ok": True, "affected": 0})
        # 该组（及其子组）下的文件全部改到新名下
        pre = grp + GROUP_SEP
        n_hit = 0
        for src, a in (d.get("assign") or {}).items():
            if (a.get("kb") or DEFAULT_KB) != kb:
                continue
            g = a.get("group") or ""
            if g == grp:
                a["group"] = new; n_hit += 1
            elif g.startswith(pre):
                a["group"] = new + GROUP_SEP + g[len(pre):]; n_hit += 1
        # 显式列表同步改名
        lst = []
        for g in ((d.get("groups") or {}).get(kb) or []):
            if g == grp:
                lst.append(new)
            elif g.startswith(pre):
                lst.append(new + GROUP_SEP + g[len(pre):])
            else:
                lst.append(g)
        # 父级也补声明
        parts = new.split(GROUP_SEP)
        for i in range(1, len(parts)):
            pre2 = GROUP_SEP.join(parts[:i])
            if pre2 not in lst:
                lst.append(pre2)
        _set_groups(d, kb, lst)
        _save_kbmeta(d)
        _refresh_payloads(d=d)
        return JSONResponse({"ok": True, "affected": n_hit})
    elif action == "rename_prefix":
        if not new:
            return JSONResponse({"error": "新分组名不能为空"}, status_code=400)
        pre = grp + GROUP_SEP
        for src, a in (d.get("assign") or {}).items():
            g = a.get("group") or ""
            if (a.get("kb") or DEFAULT_KB) != kb:
                continue
            if g == grp:
                a["group"] = new
            elif g.startswith(pre):
                a["group"] = new + GROUP_SEP + g[len(pre):]
    elif action == "delete":
        # 取消归类（只清 group，文档保留在库里）+ 移除该分组（含其子分组的文件）
        pre = grp + GROUP_SEP
        for src, a in (d.get("assign") or {}).items():
            if (a.get("kb") or DEFAULT_KB) != kb:
                continue
            g = a.get("group") or ""
            if g == grp or g.startswith(pre):
                a.pop("group", None)
        lst = [g for g in ((d.get("groups") or {}).get(kb) or [])
               if g != grp and not g.startswith(pre)]
        _set_groups(d, kb, lst)
    else:
        return JSONResponse({"error": f"未知 action: {action}"}, status_code=400)

    _save_kbmeta(d)
    _refresh_payloads(d=d)
    return JSONResponse({"ok": True, "affected": len(hit)})


# ================================================================
# AI 候选（半自动归档）：生成 → 人在管理台审核 → 采纳才落库
# ================================================================

CAND_FILE = str(pathlib.Path(STATE_FILE).parent / "candidates.json")


def _load_cand():
    p = pathlib.Path(CAND_FILE)
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}
    return {}


def _save_cand(d):
    p = pathlib.Path(CAND_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


CJOB = {"running": False, "done": 0, "total": 0, "current": "", "log": [],
        "started_at": 0, "finished_at": 0, "ok": 0, "fail": 0}
CJOB_LOCK = threading.Lock()


def _cand_worker(sources, force):
    """后台线程：逐个文档调 LLM 生成候选。"""
    with CJOB_LOCK:
        CJOB.update(running=True, done=0, total=len(sources), current="",
                    log=[], started_at=time.time(), finished_at=0, ok=0, fail=0)
    cands = _load_cand()
    meta = _ensure_default_kb(_load_kbmeta())

    for i, src in enumerate(sources, 1):
        with CJOB_LOCK:
            CJOB["current"] = src
        try:
            pts, _ = list_chunks(src, limit=3)
            head = "\n".join([(x.payload or {}).get("text", "") for x in pts])
            kb_cur, grp_cur = _meta_of(meta, src)
            groups = _groups_in_kb(meta, kb_cur)
            kbs = [k["name"] for k in meta.get("kbs", [])]
            res = _classify.classify(src, head, groups, kbs)
            if res and (res.get("group") or res.get("kb")):
                res["suggest_kb"] = res.get("kb") or kb_cur
                res["created"] = int(time.time())
                res["source"] = src
                cands[src] = res
                with CJOB_LOCK:
                    CJOB["ok"] += 1
                    CJOB["log"].append(f"  ✔ {src.split('/')[-1][:46]}  →  {(res.get('group') or '不分类')}  [{','.join(res.get('tags') or [])}]")
            else:
                err = (res or {}).get("error") or "无返回"
                with CJOB_LOCK:
                    CJOB["fail"] += 1
                    CJOB["log"].append(f"  ✘ {src.split('/')[-1][:46]}  {err}")
        except Exception as e:
            with CJOB_LOCK:
                CJOB["fail"] += 1
                CJOB["log"].append(f"  ✘ {src.split('/')[-1][:46]}  {str(e)[:80]}")
        with CJOB_LOCK:
            CJOB["done"] = i
        _save_cand(cands)

    with CJOB_LOCK:
        CJOB.update(running=False, finished_at=time.time(), current="")


async def api_cand_list(request):
    cands = _load_cand()
    meta = _ensure_default_kb(_load_kbmeta())
    files = {f["name"]: f for f in _scan_docs()}
    out = []
    for src, c in cands.items():
        if src not in files:
            continue
        kb_cur, grp_cur = _meta_of(meta, src)
        out.append({
            "source": src,
            "fname": src.split("/")[-1],
            "size": files[src].get("size"),
            "chunks": files[src].get("chunks"),
            "cur_kb": kb_cur,
            "cur_group": grp_cur,
            "kb": c.get("suggest_kb") or c.get("kb") or kb_cur,
            "group": c.get("group"),
            "tags": c.get("tags") or [],
            "confidence": c.get("confidence"),
            "seconds": c.get("seconds"),
            "attempt": c.get("attempt"),
            "same_as_now": (c.get("suggest_kb") or kb_cur) == kb_cur and (c.get("group") or UNGROUPED) == grp_cur,
        })
    out.sort(key=lambda x: x["source"])
    return JSONResponse({"candidates": out, "total": len(out)})


async def api_cand_run(request):
    if request.method == "POST":
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        with CJOB_LOCK:
            if CJOB["running"]:
                return JSONResponse({"error": "分类任务正在运行中"}, status_code=409)
        meta = _ensure_default_kb(_load_kbmeta())
        allfiles = _scan_docs()
        want = body.get("sources")
        if want:
            todo = [s for s in want if any(f["name"] == s for f in allfiles)]
        else:
            # 默认只处理「未分组」或 未归到本知识库 的
            todo = []
            for f in allfiles:
                kb_, grp_ = _meta_of(meta, f["name"])
                if grp_ == UNGROUPED:
                    todo.append(f["name"])
            if body.get("force"):
                todo = [f["name"] for f in allfiles]
        if not todo:
            return JSONResponse({"started": True, "note": "没有需要分类的文档（都已归组）", "total": 0})
        threading.Thread(target=_cand_worker, args=(todo, bool(body.get("force"))), daemon=True).start()
        return JSONResponse({"started": True, "total": len(todo)})

    with CJOB_LOCK:
        return JSONResponse({
            "running": CJOB["running"], "done": CJOB["done"], "total": CJOB["total"],
            "current": CJOB["current"], "log": "\n".join(CJOB["log"][-60:]),
            "started_at": CJOB["started_at"], "finished_at": CJOB["finished_at"],
            "ok": CJOB["ok"], "fail": CJOB["fail"],
        })


async def api_cand_apply(request):
    """采纳候选：{sources:[...]} 或 {source, kb, group, tags} 手工调整后提交。"""
    body = await request.json()
    cands = _load_cand()
    meta = _ensure_default_kb(_load_kbmeta())
    applied = []

    single = body.get("source")
    items = []
    if single:
        items.append({"source": single, "kb": body.get("kb"), "group": body.get("group"),
                      "tags": body.get("tags")})
    else:
        for s in (body.get("sources") or []):
            items.append({"source": s})

    for it in items:
        src = it["source"]
        c = cands.get(src) or {}
        kb = _norm_name(it.get("kb") or c.get("suggest_kb") or c.get("kb") or DEFAULT_KB) or DEFAULT_KB
        grp = _norm_name(it.get("group") if it.get("group") is not None else (c.get("group") or ""))
        tags = it.get("tags") if it.get("tags") is not None else (c.get("tags") or [])

        if not _find_kb(meta, kb):
            meta["kbs"].append({"name": kb, "desc": "AI 建议创建", "created": int(time.time())})

        a = meta["assign"].setdefault(src, {})
        a["kb"] = kb
        if grp:
            a["group"] = grp
        else:
            a.pop("group", None)
        a["src"] = "ai"

        # 标签
        if tags:
            tg = _load_tags()
            cur = set(tg.get(src) or [])
            cur |= set(str(t).strip() for t in tags if str(t).strip())
            tg[src] = sorted(cur)
            _save_tags(tg)

        cands.pop(src, None)
        applied.append(src)

    # 声明用到的分组（保证空组也能显示）
    for src in applied:
        k2, g2 = _meta_of(meta, src)
        if g2 and g2 != UNGROUPED:
            lst = list((meta.get("groups") or {}).get(k2) or [])
            parts = g2.split(GROUP_SEP)
            for i in range(1, len(parts) + 1):
                pre = GROUP_SEP.join(parts[:i])
                if pre not in lst:
                    lst.append(pre)
            _set_groups(meta, k2, lst)

    _save_kbmeta(meta)
    _save_cand(cands)
    _refresh_payloads(d=meta)
    return JSONResponse({"ok": True, "applied": applied})


async def api_cand_dismiss(request):
    body = await request.json()
    srcs = body.get("sources") or ([body.get("source")] if body.get("source") else [])
    cands = _load_cand()
    n = 0
    for s in srcs:
        if s in cands:
            cands.pop(s)
            n += 1
    _save_cand(cands)
    return JSONResponse({"ok": True, "dismissed": n})


async def api_login(request):
    """网页登录：验证账号密码，成功后返回 KB_TOKEN。

    放行规则由 AuthMiddleware 处理（/api/login 不需要 token）。
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "请求格式错误"}, status_code=400)

    u = (body.get("username") or "").strip()
    pw = body.get("password") or ""

    if not KB_ADMIN_PASS:
        return JSONResponse(
            {"error": "服务端未设置登录密码：请在 .env 里配置 KB_ADMIN_PASS 后重启"},
            status_code=500)
    if u == KB_ADMIN_USER and pw == KB_ADMIN_PASS:
        return JSONResponse({"ok": True, "token": KB_TOKEN, "user": KB_ADMIN_USER})
    return JSONResponse({"error": "用户名或密码不对"}, status_code=401)


async def api_logininfo(request):
    """告诉前端：要不要登录、登录功能是否可用（不需要 token）。"""
    return JSONResponse({
        "login_enabled": bool(KB_ADMIN_PASS),
        "user_hint": KB_ADMIN_USER if KB_ADMIN_PASS else "",
    })


async def homepage(request):
    f = APP_DIR / "web" / "index.html"
    if not f.exists():
        return HTMLResponse("<h1>管理页缺失</h1>", status_code=500)
    # 禁止缓存：保证每次刷新都拿到最新的页面（否则改完页面用户看不到）
    return HTMLResponse(
        f.read_text(encoding="utf-8"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


async def api_status(request):
    try:
        ensure_collection()
        vectors = collection_count()
    except Exception as e:
        vectors = -1
        print(f"qdrant 异常: {e}")
    files = _scan_docs()
    state = _load_state()
    newest = 0
    if state:
        newest = max((v.get("indexed_at", 0) for v in state.values()), default=0)
    return JSONResponse({
        "collection": COLLECTION,
        "vectors": vectors,
        "docs_dir": DOCS_DIR,
        "files_total": len(files),
        "files_indexed": sum(1 for f in files if f["indexed"]),
        "files_pending": sum(1 for f in files if not f["indexed"]),
        "last_indexed": newest,
        "indexing": JOB["running"],
        "ocr_ready": bool(os.getenv("VLM_API_KEY")),
        "embed_model": os.getenv("EMBED_MODEL", ""),
        "vlm_model": os.getenv("VLM_MODEL", ""),
    })


async def api_files(request):
    d = _ensure_default_kb(_load_kbmeta())
    out = []
    for f in _scan_docs():
        kb, grp = _meta_of(d, f["name"])
        f["kb"] = kb
        f["group"] = grp              # 未归类时为「未分组」
        out.append(f)
    return JSONResponse({"files": out})


async def api_upload(request):
    form = await request.form()
    root = pathlib.Path(DOCS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    saved, skipped = [], []

    # 关键：必须用 multi_items()，不能只遍历 .values()
    #   FormData 继承 ImmutableMultiDict，.values() 是 Mapping 语义 —— 按 key 遍历，
    #   同名的 files 字段只会返回一次（最后那个），导致多文件上传只落 1 个。
    #   multi_items() 才会把所有重复键值对逐个吐出来。
    for _key, field in form.multi_items():
        fname = getattr(field, "filename", None)
        if not fname:
            continue
        name = pathlib.Path(fname).name
        if not name or name.startswith("."):
            continue
        if pathlib.Path(name).suffix.lower() not in SUPPORTED:
            skipped.append(name)
            continue
        try:
            data = await field.read()
            if not data:
                skipped.append(name)
                continue
            (root / name).write_bytes(data)
            saved.append(name)
        except Exception as e:
            skipped.append(f"{name}（{type(e).__name__}）")
    print(f"[上传] 收到字段 {len(form.multi_items())} 个，落盘 {len(saved)} 个，跳过 {len(skipped)} 个", flush=True)
    return JSONResponse({"saved": saved, "skipped": skipped})


async def api_delete(request):
    name = pathlib.Path(request.path_params["name"]).name
    p = pathlib.Path(DOCS_DIR) / name
    if p.exists() and p.is_file():
        p.unlink()
        return JSONResponse({"deleted": name})
    return JSONResponse({"error": "文件不存在"}, status_code=404)


def _read_progress_file():
    """读索引进程写的共享进度文件。"""
    p = pathlib.Path(PROGRESS_FILE)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


async def api_backends(request):
    """顶部连接状态指示灯：转发 embed-shim 的 /probe。

    只读探测，不解码任何文档、不消耗 Cloudflare neurons：
      local  -> GET {OLLAMA_URL}/api/version
      cf     -> GET /user/tokens/verify
    """
    base = (os.getenv("OLLAMA_URL") or "").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{base}/probe")
            return JSONResponse(r.json())
    except Exception as e:
        return JSONResponse({
            "local": {"ok": False, "ms": None, "detail": f"embed-shim 不可达：{e}"},
            "cloudflare": {"ok": False, "ms": None, "detail": f"embed-shim 不可达：{e}"},
            "error": str(e),
        })


async def api_index_progress(request):
    """统一的索引进度接口。

    数据源：共享文件 progress.json（cron 后台跑的也能读到），
    并与 server 自己 spawn 的 JOB 合并。
    超过 90 秒没有更新视为已结束（进程崩了 / 被 kill）。
    """
    j = _read_progress_file()
    if not j:
        j = dict(JOB.get("progress") or {})
        j["running"] = JOB.get("running", False)
        j["returncode"] = JOB.get("returncode")
        j["trigger"] = "manual"
        return JSONResponse(j)

    if JOB.get("running"):
        j["running"] = True
    rc = JOB.get("returncode")
    if rc is not None and not JOB.get("running"):
        j["returncode"] = rc

    age = time.time() - (j.get("updated_at") or 0)
    if j.get("running") and age > 90:
        j["running"] = False
        j["stale"] = True
        j["stale_seconds"] = int(age)
    j["age"] = round(age, 1)
    return JSONResponse(j)


async def api_index(request):
    if request.method == "POST":
        try:
            _body = await request.json()
        except Exception:
            _body = {}
        backend = (_body or {}).get("backend") or "auto"
        if backend not in ("auto", "local", "cloudflare"):
            backend = "auto"
        with JOB_LOCK:
            if JOB["running"]:
                return JSONResponse({"error": "索引正在运行中（本进程）"}, status_code=409)
        if _index_running_elsewhere():
            pf = _read_progress_file()
            who = pf.get("trigger") or "另一个进程"
            return JSONResponse({
                "error": f"索引正在运行中（{who} 触发的），请等它跑完再试",
                "running_elsewhere": True,
                "progress": pf,
            }, status_code=409)
        threading.Thread(target=_index_worker, args=(backend,), daemon=True).start()
        return JSONResponse({"started": True, "backend": backend})
    with JOB_LOCK:
        pf = _read_progress_file()
        return JSONResponse({
            "running": JOB["running"] or bool(pf.get("running")),
            "progress_file": pf,
            "trigger": pf.get("trigger"),
            "log": JOB["log"],
            "progress": JOB.get("progress") or {},
            "started_at": JOB["started_at"],
            "finished_at": JOB["finished_at"],
            "returncode": JOB["returncode"],
        })


async def api_search(request):
    body = await request.json()
    q = (body.get("query") or "").strip()
    k = int(body.get("top_k") or 6)
    sources = body.get("sources") or None          # 限定检索范围（文件列表）
    tags = body.get("tags") or None                # 限定标签范围
    if not q:
        return JSONResponse({"error": "query 为空"}, status_code=400)

    # 知识库 / 分组 / 标签：直接下沉到 payload 做向量级过滤
    # （v2 起不再需要先全量扫描文件算白名单）
    kb = (body.get("kb") or "").strip()
    group = (body.get("group") or "").strip()
    use_rerank = body.get("rerank")
    try:
        # 检索链路（ollama 嵌入 + qdrant 查询 + LLM 重排）全是同步阻塞调用，
        # 单次可能跑 30~400 秒。必须放到线程池，否则整个事件循环被占住，
        # 期间管理台、kb_stats、MCP 握手全部无响应（实测会整站假死）。
        hits = await run_in_threadpool(
            search_in,
            q,
            top_k=max(1, min(k, 15)),
            sources=sources,
            kb=kb or None,
            group=group or None,
            tags=tags or None,
            use_rerank=None if use_rerank is None else bool(use_rerank),
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    if not hits and (kb or group or tags):
        return JSONResponse({"hits": [], "note": "该范围下没有匹配的文档"})

    return JSONResponse({"hits": [
        {
            "source": (h.payload or {}).get("source"),
            "page": (h.payload or {}).get("page"),
            "chunk": (h.payload or {}).get("chunk"),
            "section": (h.payload or {}).get("section"),
            "context": (h.payload or {}).get("context"),
            "kb": (h.payload or {}).get("kb"),
            "group": (h.payload or {}).get("group"),
            "tags": (h.payload or {}).get("tags"),
            "score": getattr(h, "score", None),
            "text": (h.payload or {}).get("text", ""),
        }
        for h in hits
    ]})


# ---------------------------------------------------------------- 鉴权

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.url.path
        if path in ("/", "/api/login", "/api/logininfo") or path.startswith("/vendor/") or path.startswith("/favicon"):
            return await call_next(request)
        if KB_TOKEN:
            got = request.headers.get("authorization", "")
            if got != f"Bearer {KB_TOKEN}":
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def build_app():
    if hasattr(mcp, "streamable_http_app"):
        app = mcp.streamable_http_app()
    elif hasattr(mcp, "sse_app"):
        app = mcp.sse_app()
    else:
        raise RuntimeError("当前 mcp 版本不支持 streamable-http")

    routes = [
        Route("/", homepage),
        Route("/api/login", api_login, methods=["POST"]),
        Route("/api/logininfo", api_logininfo, methods=["GET"]),
        Route("/api/status", api_status),
        Route("/api/backends", api_backends),
        Route("/api/files", api_files),
        Route("/api/files/batch", api_files_batch, methods=["POST"]),
        Route("/api/files/{name}", api_delete, methods=["DELETE"]),
        Route("/api/upload", api_upload, methods=["POST"]),
        Route("/api/index", api_index, methods=["GET", "POST"]),
        Route("/api/index/progress", api_index_progress, methods=["GET"]),
        Route("/api/search", api_search, methods=["POST"]),
        Route("/api/tree", api_tree, methods=["GET"]),
        Route("/api/kb", api_kb_list, methods=["GET"]),
        Route("/api/kb/tree", api_kb_tree, methods=["GET"]),
        Route("/api/kb/op", api_kb_op, methods=["POST"]),
        Route("/api/kb/assign", api_kb_assign, methods=["POST"]),
        Route("/api/kb/group", api_kb_group_op, methods=["POST"]),
        Route("/api/candidates", api_cand_list, methods=["GET"]),
        Route("/api/candidates/run", api_cand_run, methods=["GET", "POST"]),
        Route("/api/candidates/apply", api_cand_apply, methods=["POST"]),
        Route("/api/candidates/dismiss", api_cand_dismiss, methods=["POST"]),
        Route("/api/tags", api_tags_get, methods=["GET"]),
        Route("/api/tags", api_tags_set, methods=["POST"]),
        Route("/api/chunks", api_chunks_list, methods=["GET"]),
        Route("/api/chunks", api_chunk_add, methods=["POST"]),
        Route("/api/chunks/{pid}", api_chunk_update, methods=["PUT"]),
        Route("/api/chunks/{pid}", api_chunk_delete, methods=["DELETE"]),
    ]
    for r in reversed(routes):
        app.routes.insert(0, r)

    vendor = APP_DIR / "web" / "vendor"
    if vendor.exists():
        app.mount("/vendor", StaticFiles(directory=str(vendor)), name="vendor")

    app.add_middleware(AuthMiddleware)
    return app


if __name__ == "__main__":
    application = build_app()
    print(f"MCP   : http://{MCP_HOST}:{MCP_PORT}/mcp")
    print(f"管理页: http://{MCP_HOST}:{MCP_PORT}/?token=<KB_TOKEN>")
    uvicorn.run(application, host=MCP_HOST, port=MCP_PORT, log_level="info")