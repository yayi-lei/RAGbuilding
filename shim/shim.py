#!/usr/bin/env python3
"""embed-shim —— 嵌入网关：本地 Ollama 优先，Cloudflare 兜底。

规则（v2）：
  * 默认顺序由 EMBED_PRIORITY 决定，当前部署为 local（本地优先），
    本地不可用时自动降级到 Cloudflare。
  * 不分场景：索引、检索、手动任务都走同一条规则。
  * 每个后端独立熔断，避免一个后端挂掉时逐个请求硬等超时。

路径路由（给手动模式预留，UI 下一步接）：
  /api/embed            → 按 EMBED_PRIORITY 规则
  /local/api/embed      → 强制本地
  /cf/api/embed         → 强制 Cloudflare

协议（三条都实现）：
  POST /api/embed        Ollama 批量
  POST /api/embeddings   Ollama 旧版单条
  POST /v1/embeddings    OpenAI 兼容
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

# ------------------------------------------------------------------ 配置

EMBED_PRIORITY = os.getenv("EMBED_PRIORITY", "local").strip().lower()

CF_ENABLED = os.getenv("CF_ENABLED", "1") not in ("0", "false", "False", "")
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "").strip()
CF_API_TOKEN = os.getenv("CF_API_TOKEN", "").strip()
CF_MODEL = os.getenv("CF_MODEL", "@cf/qwen/qwen3-embedding-0.6b")
CF_TIMEOUT = float(os.getenv("CF_TIMEOUT", "45"))
CF_PAYLOAD_KEY = os.getenv("CF_PAYLOAD_KEY", "text")
CF_MAX_BATCH = int(os.getenv("CF_MAX_BATCH", "32"))

OLLAMA_URL = os.getenv("OLLAMA_URL", "").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-embedding:0.6b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "240"))

EXPECT_DIM = int(os.getenv("EMBED_DIM", "1024"))

FAIL_THRESHOLD = int(os.getenv("EMBED_FAIL_THRESHOLD", "3"))
COOLDOWN = float(os.getenv("EMBED_COOLDOWN", "120"))

SHIM_HOST = os.getenv("SHIM_HOST", "0.0.0.0")
SHIM_PORT = int(os.getenv("SHIM_PORT", "8100"))
VERBOSE = os.getenv("SHIM_VERBOSE", "1") not in ("0", "false", "False")

CF_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{CF_MODEL}"
    if CF_ACCOUNT_ID
    else ""
)


def log(msg: str) -> None:
    if VERBOSE:
        print(f"[embed-shim] {msg}", flush=True)


# ------------------------------------------------------------------ 熔断

class Breaker:
    def __init__(self, name: str) -> None:
        self.name = name
        self.fails = 0
        self.open_until = 0.0
        self.last_error = ""
        self.ok = 0
        self.fail = 0

    @property
    def is_open(self) -> bool:
        return time.time() < self.open_until

    def success(self) -> None:
        self.fails = 0
        self.open_until = 0.0
        self.ok += 1

    def failure(self, err: str) -> None:
        self.fails += 1
        self.fail += 1
        self.last_error = err[:300]
        if self.fails >= FAIL_THRESHOLD:
            self.open_until = time.time() + COOLDOWN
            log(f"[{self.name}] 熔断 {COOLDOWN:.0f}s（连续失败 {self.fails} 次）：{self.last_error}")

    def snap(self) -> dict[str, Any]:
        return {
            "open": self.is_open,
            "remaining_s": max(0, round(self.open_until - time.time(), 1)),
            "consecutive_failures": self.fails,
            "last_error": self.last_error,
            "ok": self.ok,
            "fail": self.fail,
        }


BR_LOCAL = Breaker("local")
BR_CF = Breaker("cloudflare")
LAST_BACKEND = ""

_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(CF_TIMEOUT, connect=10.0))
    return _client


# ------------------------------------------------------------------ 后端

def _extract_cf_vectors(payload: dict[str, Any]) -> list[list[float]]:
    if not isinstance(payload, dict):
        raise ValueError("cf: 返回不是对象")
    if payload.get("success") is False:
        raise ValueError(f"cf 业务失败: {json.dumps(payload.get('errors'), ensure_ascii=False)[:200]}")
    res = payload.get("result", payload)
    if isinstance(res, dict):
        data = res.get("data")
        if isinstance(data, list) and data and isinstance(data[0], list):
            return data
        if isinstance(data, list) and data and isinstance(data[0], (int, float)):
            return [data]
        emb = res.get("embeddings") or res.get("embedding")
        if isinstance(emb, list):
            return emb if isinstance(emb[0], list) else [emb]
    if isinstance(res, list):
        if res and isinstance(res[0], list):
            return res
        if res and isinstance(res[0], (int, float)):
            return [res]
    raise ValueError(f"cf: 无法识别的返回结构 keys={list(payload)[:6]}")


def _guard_dim(vecs: list[list[float]], backend: str) -> list[list[float]]:
    if not vecs:
        raise ValueError(f"{backend}: 返回空向量")
    dim = len(vecs[0])
    if dim != EXPECT_DIM:
        raise ValueError(f"{backend}: 维度 {dim} != 期望 {EXPECT_DIM}")
    return vecs


async def embed_cloudflare(texts: list[str]) -> list[list[float]]:
    if not (CF_ENABLED and CF_ACCOUNT_ID and CF_API_TOKEN):
        raise RuntimeError("cf: 未配置")
    out: list[list[float]] = []
    c = client()
    for i in range(0, len(texts), CF_MAX_BATCH):
        chunk = texts[i:i + CF_MAX_BATCH]
        try:
            r = await c.post(
                CF_URL,
                headers={"Authorization": f"Bearer {CF_API_TOKEN}",
                         "Content-Type": "application/json"},
                json={CF_PAYLOAD_KEY: chunk},
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"cf: 网络异常 {type(e).__name__}: {e}") from e
        if r.status_code != 200:
            raise RuntimeError(f"cf: HTTP {r.status_code} {r.text[:200]}")
        try:
            payload = r.json()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"cf: 返回非 JSON {r.text[:160]}") from e
        out.extend(_extract_cf_vectors(payload))
    if len(out) != len(texts):
        raise RuntimeError(f"cf: 条数不符 {len(out)} != {len(texts)}")
    return _guard_dim(out, "cf")


async def embed_local(texts: list[str], model: str) -> list[list[float]]:
    c = client()
    try:
        r = await c.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": model or OLLAMA_MODEL, "input": texts},
            timeout=OLLAMA_TIMEOUT,
        )
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"local: 网络异常 {type(e).__name__}: {e}") from e
    if r.status_code != 200:
        raise RuntimeError(f"local: HTTP {r.status_code} {r.text[:200]}")
    try:
        vecs = (r.json() or {}).get("embeddings") or []
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"local: 返回非 JSON {e}") from e
    if not vecs:
        raise RuntimeError("local: 返回空")
    return _guard_dim(vecs, "local")


LOCAL = ("local", BR_LOCAL, embed_local)
CLOUDFLARE = ("cloudflare", BR_CF, embed_cloudflare)


def order_for(mode: str) -> list[tuple]:
    """mode: auto / local / cloudflare"""
    if mode == "local":
        return [LOCAL]
    if mode == "cloudflare":
        return [CLOUDFLARE]
    # auto：按 EMBED_PRIORITY 排，另一个做兜底
    return [LOCAL, CLOUDFLARE] if EMBED_PRIORITY != "cloudflare" else [CLOUDFLARE, LOCAL]


async def embed(texts: list[str], model: str, mode: str = "auto") -> tuple[list[list[float]], str]:
    global LAST_BACKEND
    errors: list[str] = []
    tried_primary = False

    for name, br, fn in order_for(mode):
        if br.is_open:
            errors.append(f"{name}: 熔断中（剩余 {max(0, br.open_until - time.time()):.0f}s）")
            continue
        try:
            vecs = await fn(texts, model) if name == "local" else await fn(texts)
            br.success()
            LAST_BACKEND = name
            if tried_primary:
                log(f"已降级到 {name}（主通道不可用）")
            return vecs, name
        except Exception as e:  # noqa: BLE001
            br.failure(str(e))
            errors.append(f"{name}: {e}")
            log(f"{name} 失败：{str(e)[:180]}")
            tried_primary = True

    raise RuntimeError("；".join(errors) or "没有可用后端")


# ------------------------------------------------------------------ 请求归一化

def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out = []
        for v in value:
            if not isinstance(v, str):
                raise ValueError("input 必须是字符串或字符串数组")
            out.append(v)
        return out
    raise ValueError("缺少 input")


def _err(status: int, msg: str) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=status)


# ------------------------------------------------------------------ 处理器

def make_embed_handler(mode: str):
    async def handler(request: Request) -> Response:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return _err(400, "invalid json")
        try:
            texts = _as_list(body.get("input"))
        except ValueError as e:
            return _err(400, str(e))
        model = body.get("model") or OLLAMA_MODEL
        t0 = time.time()
        try:
            vecs, backend = await embed(texts, model, mode)
        except Exception as e:  # noqa: BLE001
            log(f"/api/embed 全部后端失败：{e}")
            return _err(503, str(e))
        log(f"/api/embed mode={mode} n={len(texts)} backend={backend} {time.time()-t0:.1f}s")
        return JSONResponse({
            "model": model,
            "embeddings": vecs,
            "backend": backend,
            "total_duration": int((time.time() - t0) * 1e9),
        })
    return handler


def make_legacy_handler(mode: str):
    async def handler(request: Request) -> Response:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return _err(400, "invalid json")
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            return _err(400, "缺少 prompt")
        model = body.get("model") or OLLAMA_MODEL
        try:
            vecs, backend = await embed([prompt], model, mode)
        except Exception as e:  # noqa: BLE001
            return _err(503, str(e))
        return JSONResponse({"embedding": vecs[0], "backend": backend})
    return handler


def make_v1_handler(mode: str):
    async def handler(request: Request) -> Response:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return _err(400, "invalid json")
        try:
            texts = _as_list(body.get("input"))
        except ValueError as e:
            return _err(400, str(e))
        model = body.get("model") or OLLAMA_MODEL
        try:
            vecs, backend = await embed(texts, model, mode)
        except Exception as e:  # noqa: BLE001
            return _err(503, str(e))
        tokens = sum(max(1, len(t) // 2) for t in texts)
        return JSONResponse({
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": v}
                     for i, v in enumerate(vecs)],
            "model": model,
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            "backend": backend,
        })
    return handler


async def api_tags(request: Request) -> Response:
    try:
        r = await client().get(f"{OLLAMA_URL}/api/tags", timeout=20)
        return Response(r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))
    except Exception:  # noqa: BLE001
        return JSONResponse({"models": [{"name": OLLAMA_MODEL}]})


async def api_version(request: Request) -> Response:
    return JSONResponse({"version": "embed-shim/2.0", "priority": EMBED_PRIORITY})


async def probe(request: Request) -> Response:
    """连通性探测，给管理台顶部指示灯用。

    两个都刻意选了零成本的检查：
      本地   → GET {OLLAMA_URL}/api/version   （不加载模型、不算向量）
      CF     → GET /user/tokens/verify        （只校验 token，不消耗 neurons）
    """
    out: dict[str, Any] = {
        "local": {"ok": False, "ms": None, "detail": ""},
        "cloudflare": {"ok": False, "ms": None, "detail": ""},
        "priority": EMBED_PRIORITY,
        "last_backend": LAST_BACKEND,
        "breakers": {"local": BR_LOCAL.snap(), "cloudflare": BR_CF.snap()},
    }

    t0 = time.time()
    try:
        r = await client().get(f"{OLLAMA_URL}/api/version",
                               timeout=httpx.Timeout(6.0, connect=3.0))
        out["local"]["ms"] = int((time.time() - t0) * 1000)
        if r.status_code == 200:
            out["local"]["ok"] = True
            out["local"]["detail"] = str((r.json() or {}).get("version", ""))
        else:
            out["local"]["detail"] = f"HTTP {r.status_code}"
    except Exception as e:  # noqa: BLE001
        out["local"]["ms"] = int((time.time() - t0) * 1000)
        out["local"]["detail"] = f"{type(e).__name__}: {str(e)[:140]}"

    t0 = time.time()
    if not (CF_ENABLED and CF_ACCOUNT_ID and CF_API_TOKEN):
        out["cloudflare"]["detail"] = "未配置"
    else:
        try:
            r = await client().get(
                "https://api.cloudflare.com/client/v4/user/tokens/verify",
                headers={"Authorization": f"Bearer {CF_API_TOKEN}"},
                timeout=httpx.Timeout(8.0, connect=4.0),
            )
            out["cloudflare"]["ms"] = int((time.time() - t0) * 1000)
            ok = r.status_code == 200 and (r.json() or {}).get("success") is True
            out["cloudflare"]["ok"] = ok
            out["cloudflare"]["detail"] = "active" if ok else f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            out["cloudflare"]["ms"] = int((time.time() - t0) * 1000)
            out["cloudflare"]["detail"] = f"{type(e).__name__}: {str(e)[:140]}"

    return JSONResponse(out)


async def healthz(request: Request) -> Response:
    return JSONResponse({
        "ok": True,
        "priority": EMBED_PRIORITY,
        "order_auto": [n for n, _, _ in order_for("auto")],
        "cf_enabled": bool(CF_ENABLED and CF_ACCOUNT_ID and CF_API_TOKEN),
        "cf_model": CF_MODEL if CF_ACCOUNT_ID else None,
        "ollama_url": OLLAMA_URL,
        "ollama_model": OLLAMA_MODEL,
        "expected_dim": EXPECT_DIM,
        "last_backend": LAST_BACKEND,
        "local": BR_LOCAL.snap(),
        "cloudflare": BR_CF.snap(),
    })


app = Starlette(routes=[
    Route("/api/embed", make_embed_handler("auto"), methods=["POST"]),
    Route("/local/api/embed", make_embed_handler("local"), methods=["POST"]),
    Route("/cf/api/embed", make_embed_handler("cloudflare"), methods=["POST"]),

    Route("/api/embeddings", make_legacy_handler("auto"), methods=["POST"]),
    Route("/local/api/embeddings", make_legacy_handler("local"), methods=["POST"]),
    Route("/cf/api/embeddings", make_legacy_handler("cloudflare"), methods=["POST"]),

    Route("/v1/embeddings", make_v1_handler("auto"), methods=["POST"]),
    Route("/local/v1/embeddings", make_v1_handler("local"), methods=["POST"]),
    Route("/cf/v1/embeddings", make_v1_handler("cloudflare"), methods=["POST"]),

    Route("/api/tags", api_tags, methods=["GET"]),
    Route("/api/version", api_version, methods=["GET"]),
    Route("/probe", probe, methods=["GET"]),
    Route("/healthz", healthz, methods=["GET"]),
    Route("/", healthz, methods=["GET"]),
])


if __name__ == "__main__":
    log(f"listen {SHIM_HOST}:{SHIM_PORT} priority={EMBED_PRIORITY} "
        f"order={[n for n, _, _ in order_for('auto')]} "
        f"local={OLLAMA_URL} cf={CF_MODEL if CF_ACCOUNT_ID else 'off'} dim={EXPECT_DIM}")
    uvicorn.run(app, host=SHIM_HOST, port=SHIM_PORT, log_level="warning")
