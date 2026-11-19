"""AI 模型（归档建议 / 扫描件 OCR / 上下文 / 重排）运行时可改配置。

背景（踩过的坑）
    以前 VLM_BASE_URL / VLM_API_KEY / VLM_MODEL 只存在 docker env 里。
    网关那边一旦把模型名换掉（zhanlu/glm-5.3-flash -> moma/glm-5.3-flash），
    容器里的默认值就变成死值，AI 建议分组和扫描件 OCR 一起静默全失败，
    只能改 compose 重建容器。
    现在管理台「AI 建议设置」把值写进 /state/ai_settings.json，每次调用现读现用。

两个角色，共用一套网关（地址 + Key），模型各管各的：
    ai  → 归档建议 / 上下文 / 重排，纯文本对话即可
    ocr → 扫描件识别，必须选**多模态**模型（要能看图），检测时会真发一张图

优先级：ai_settings.json（管理台保存） > 环境变量 > 内置默认
"""
import base64
import json
import os
import pathlib
import struct
import threading
import time
import zlib

import httpx

STATE_DIR = pathlib.Path(os.getenv("STATE_FILE", "/state/index_state.json")).parent
SETTINGS_FILE = STATE_DIR / "ai_settings.json"
DEFAULT_MODEL = "moma/glm-5.3-flash"

# 这几项允许从管理台覆盖（其余仍走环境变量）
FIELDS = ("base_url", "api_key", "model", "enabled",
          "ocr_model", "ocr_enabled", "timeout", "max_tokens")

_LOCK = threading.RLock()
_CACHE = {"status": None, "at": 0.0}
STATUS_TTL = 60

ROLE_LABEL = {"ai": "AI 建议", "ocr": "扫描件 OCR"}


def _bool(v, default=True):
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("0", "false", "no", "off", "none", ""):
        return False
    if s in ("1", "true", "yes", "on"):
        return True
    return default


def _env():
    return {
        "base_url": (os.getenv("VLM_BASE_URL") or "").rstrip("/"),
        "api_key": os.getenv("VLM_API_KEY") or "",
        "model": (os.getenv("VLM_MODEL") or DEFAULT_MODEL).strip(),
        "enabled": _bool(os.getenv("CLASSIFY_ENABLED", "1")),
        "ocr_model": (os.getenv("OCR_MODEL") or os.getenv("VLM_MODEL") or DEFAULT_MODEL).strip(),
        "ocr_enabled": _bool(os.getenv("OCR_ENABLED", "1")),
        "timeout": int(os.getenv("CLASSIFY_TIMEOUT", "120") or 120),
        "max_tokens": int(os.getenv("CLASSIFY_MAX_TOKENS", "2000") or 2000),
    }


def _read():
    try:
        d = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _as_int(v, default):
    try:
        return int(v)
    except Exception:
        return default


def current():
    """当前生效配置（管理台保存的优先于环境变量）。"""
    c = _env()
    saved = _read()
    if saved.get("base_url"):
        c["base_url"] = str(saved["base_url"]).strip().rstrip("/")
    if saved.get("api_key"):
        c["api_key"] = str(saved["api_key"]).strip()
    if saved.get("model"):
        c["model"] = str(saved["model"]).strip()
    if "enabled" in saved:
        c["enabled"] = _bool(saved["enabled"])
    if saved.get("ocr_model"):
        c["ocr_model"] = str(saved["ocr_model"]).strip()
    else:
        # 没单独配 OCR 模型时，沿用 AI 建议那个（老部署就是这个行为）
        c["ocr_model"] = c["model"]
    if "ocr_enabled" in saved:
        c["ocr_enabled"] = _bool(saved["ocr_enabled"])
    if saved.get("timeout"):
        c["timeout"] = _as_int(saved["timeout"], c["timeout"])
    if saved.get("max_tokens"):
        c["max_tokens"] = _as_int(saved["max_tokens"], c["max_tokens"])
    return c


def model_of(cfg, kind="ai"):
    """这个角色当前用的模型名。"""
    cfg = cfg or current()
    if kind == "ocr":
        return (cfg.get("ocr_model") or cfg.get("model") or "").strip()
    return (cfg.get("model") or "").strip()


def enabled_of(cfg, kind="ai"):
    cfg = cfg or current()
    return _bool(cfg.get("ocr_enabled" if kind == "ocr" else "enabled"), True)


def _mask(k):
    if not k:
        return ""
    if len(k) <= 10:
        return k[:2] + "*" * max(0, len(k) - 2)
    return k[:6] + "*" * 8 + k[-4:]


def public():
    c = current()
    saved = _read()
    return {
        "base_url": c["base_url"],
        "model": c["model"],
        "enabled": c["enabled"],
        "ocr_model": c["ocr_model"],
        "ocr_enabled": c["ocr_enabled"],
        "timeout": c["timeout"],
        "max_tokens": c["max_tokens"],
        "api_key_set": bool(c["api_key"]),
        "api_key_masked": _mask(c["api_key"]),
        "source": "ui" if saved else "env",
        "env_model": _env()["model"],
        "env_ocr_model": _env()["ocr_model"],
        "file": str(SETTINGS_FILE),
    }


def save(patch):
    """保存设置；api_key 传空字符串表示保持不变。返回 public()。"""
    patch = patch or {}
    with _LOCK:
        saved = {k: v for k, v in _read().items() if k in FIELDS}
        for k in ("base_url", "model"):
            if patch.get(k) is not None and str(patch.get(k)).strip():
                saved[k] = str(patch[k]).strip()
        if "ocr_model" in patch:
            v = str(patch.get("ocr_model") or "").strip()
            if v:
                saved["ocr_model"] = v
            else:
                saved.pop("ocr_model", None)   # 清空 = 沿用「AI 建议」那个模型
        if patch.get("api_key"):
            saved["api_key"] = str(patch["api_key"]).strip()
        for k in ("enabled", "ocr_enabled"):
            if patch.get(k) is not None:
                saved[k] = _bool(patch[k])
        for k in ("timeout", "max_tokens"):
            if patch.get(k):
                saved[k] = _as_int(patch[k], _env()[k])
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            os.chmod(SETTINGS_FILE, 0o600)
        except Exception:
            pass
        _CACHE["status"] = None
        _CACHE["at"] = 0.0
        return public()


def _headers(key):
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _err_detail(r):
    t = (r.text or "")[:300]
    try:
        j = r.json()
        if isinstance(j, dict):
            e = j.get("error")
            if isinstance(e, dict) and e.get("message"):
                return str(e["message"])[:300]
            if isinstance(e, str):
                return e[:300]
    except Exception:
        pass
    return t


def list_models(cfg=None):
    """读网关 /v1/models，给设置页的下拉框用。"""
    cfg = cfg or current()
    out = {"models": [], "error": ""}
    if not (cfg["base_url"] and cfg["api_key"]):
        out["error"] = "未配置接口地址或 API Key"
        return out
    try:
        with httpx.Client(timeout=httpx.Timeout(20.0, connect=8.0)) as c:
            r = c.get(cfg["base_url"] + "/models", headers=_headers(cfg["api_key"]))
        if r.status_code != 200:
            out["error"] = f"HTTP {r.status_code}：{_err_detail(r)}"
            return out
        data = r.json() or {}
        out["models"] = [str(m.get("id")) for m in (data.get("data") or []) if m.get("id")]
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    return out


def _tiny_png(w=32, h=32):
    """生成一张极小的纯色 PNG，用来验证模型到底能不能看图。"""
    raw = b"".join(b"\x00" + b"\xff\xff\xff" * w for _ in range(h))

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    return base64.b64encode(png).decode()


_PNG_B64 = _tiny_png()


def _probe_body(kind, model):
    if kind == "ocr":
        return {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "这是一张 32x32 的白图，请只回复 ok。"},
                    {"type": "image_url",
                     "image_url": {"url": "data:image/png;base64," + _PNG_B64}},
                ],
            }],
            "max_tokens": 32,
            "temperature": 0,
        }
    return {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 64,
        "temperature": 0,
    }


def probe(cfg=None, kind="ai"):
    """真打一发 /chat/completions，判断这个角色的模型现在能不能用。

    kind="ocr" 时发一张真图片 —— 纯文本模型会在这里暴露出来（看图能力缺失）。
    """
    cfg = dict(cfg or current())
    model = model_of(cfg, kind)
    res = {
        "ok": False,
        "kind": kind,
        "label": ROLE_LABEL.get(kind, kind),
        "model": model,
        "base_url": cfg.get("base_url", ""),
        "latency_ms": None,
        "code": "",
        "error": "",
        "checked_at": int(time.time()),
        "models": [],
    }
    if not enabled_of(cfg, kind):
        res.update(code="disabled", error=f"{res['label']}已在设置里关闭")
        return res
    if not (cfg.get("base_url") and cfg.get("api_key")):
        res.update(code="unconfigured", error="未配置接口地址或 API Key")
        return res
    if not model:
        res.update(code="unconfigured", error="还没选模型")
        return res

    models = list_models(cfg)
    res["models"] = models.get("models") or []
    if models.get("error") and not res["models"]:
        msg = models["error"]
        low = msg.lower()
        code = "network"
        if "http 401" in low or "http 403" in low:
            code = "auth"
        elif "http 5" in low:
            code = "server_error"
        res.update(code=code, error=f"读取模型列表失败：{msg}")
        return res

    if res["models"] and model not in res["models"]:
        avail = "、".join(res["models"][:8])
        res.update(code="model_not_found",
                   error=f"模型 {model} 不在网关可用列表里（可用：{avail}）")
        return res

    body = _probe_body(kind, model)
    t0 = time.time()
    try:
        with httpx.Client(timeout=httpx.Timeout(float(cfg.get("timeout") or 60), connect=8.0)) as c:
            r = c.post(cfg["base_url"] + "/chat/completions",
                       headers=_headers(cfg["api_key"]), json=body)
        res["latency_ms"] = int((time.time() - t0) * 1000)
        if r.status_code != 200:
            detail = _err_detail(r)
            low = detail.lower()
            code = f"http_{r.status_code}"
            if ("model_not_found" in low
                    or "not supported by any configured account" in low
                    or "does not exist" in low
                    or "no such model" in low):
                code = "model_not_found"
            elif r.status_code in (401, 403):
                code = "auth"
            elif kind == "ocr" and any(k in low for k in
                                       ("image", "vision", "multimodal", "unsupported content",
                                        "image_url", "content type")):
                code = "not_vision"
                detail = "这个模型不支持图片输入（扫描件 OCR 需要多模态模型）：" + detail
            res.update(code=code, error=detail or f"HTTP {r.status_code}")
            return res
        j = r.json() or {}
        ch = (j.get("choices") or [{}])[0]
        if not ch:
            res.update(code="empty_response", error="接口返回 200 但没有 choices")
            return res
        res.update(ok=True, code="ok", error="", model=str(j.get("model") or model))
    except Exception as e:
        res.update(code="network", error=f"{type(e).__name__}: {str(e)[:160]}")
    return res


def status(force=False):
    """带 60 秒缓存的可用性检测（首页状态灯用），两个角色各测一次。"""
    now = time.time()
    with _LOCK:
        st, at = _CACHE["status"], _CACHE["at"]
    if not force and st is not None and (now - at) < STATUS_TTL:
        return st
    st = {"ai": probe(kind="ai"), "ocr": probe(kind="ocr")}
    with _LOCK:
        _CACHE["status"] = st
        _CACHE["at"] = time.time()
    return st


def status_cached():
    """只返回缓存结果（绝不阻塞请求线程）；没检测过时各角色为 None。"""
    with _LOCK:
        st = _CACHE["status"]
    if not st:
        return {"ai": None, "ocr": None}
    return st


def usable(kind="ai", check=False):
    """快速判断这个角色现在能不能跑，返回 (ok, message)。

    check=True：没有缓存时先真打一发探活。扫描件 OCR 开跑前值得花这几秒 ——
    否则要等上百页逐页失败完才知道模型名废了。
    """
    c = current()
    label = ROLE_LABEL.get(kind, kind)
    if not enabled_of(c, kind):
        return False, f"{label}已在设置里关闭（管理台 → AI 建议设置）"
    if not (c["base_url"] and c["api_key"]):
        return False, f"未配置接口地址 / API Key（管理台 → AI 建议设置）"
    st = (status_cached() or {}).get(kind)
    if st is None and check:
        st = probe(c, kind)
    if st and st.get("ok") is False and st.get("code") in (
            "model_not_found", "auth", "unconfigured", "disabled", "not_vision"):
        return False, f"{label}模型不可用：{st.get('error') or st.get('code')}"
    return True, ""
