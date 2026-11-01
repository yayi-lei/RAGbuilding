"""文档解析：把各种格式统一转成 Markdown 纯文本。

三级策略：
  1. 老格式 .doc/.xls/.ppt  → LibreOffice 无头转换 → 新格式 → markitdown
  2. 常规格式              → markitdown
  3. PDF 文本过少（扫描件） → pdftoppm 转 PNG → 多模态模型逐页识别（带磁盘缓存）
"""
import base64
import hashlib
import os
import pathlib
import subprocess

import httpx

OLD_FORMATS = {".doc": "docx", ".xls": "xlsx", ".ppt": "pptx"}

VLM_BASE_URL = os.getenv("VLM_BASE_URL", "").rstrip("/")
VLM_API_KEY = os.getenv("VLM_API_KEY", "")
VLM_MODEL = os.getenv("VLM_MODEL", "zhanlu/glm-5.3-flash")
OCR_MIN_CHARS = int(os.getenv("OCR_MIN_CHARS", "50"))
OCR_DPI = int(os.getenv("OCR_DPI", "150"))
OCR_CACHE = os.getenv("OCR_CACHE", "/ocr_cache")
TMP_DIR = os.getenv("TMP_DIR", "/tmp/ragtmp")

OCR_PROMPT = """请把这张扫描页的内容完整转写成 Markdown。要求：
1. 保留标题层级（# ## ###）
2. 表格必须用 Markdown 表格，保持行列对应，不要遗漏任何单元格
3. 条款编号、金额、日期、百分比、单位等数字必须逐字准确，不要臆测
4. 页眉、页脚、页码可以省略
5. 只输出 Markdown 正文，不要任何解释、前言或代码块围栏
"""


def human_error(e, ext=""):
    """把技术异常翻译成人能看懂的一句话。"""
    s = str(e)
    low = s.lower()
    if "not a zip file" in low or "badzipfile" in low:
        return "文件损坏或没复制完（不是完整的 Office 文件）"
    if "no /root object" in low or "cannot find root" in low:
        return "PDF 损坏或没复制完（读不出文档结构）"
    if "unrecognized shape" in low or "notimplementederror" in low:
        return "文件里有解析器不认识的图形元素"
    if "password" in low or "encrypted" in low:
        return "文件有密码保护，打不开"
    if "timed out" in low or "timeout" in low:
        return "处理超时（文件太大或太复杂）"
    if "no such file" in low:
        return "文件不存在（可能在拷贝过程中被移走了）"
    if "permission" in low:
        return "没有读取权限"
    if not s.strip():
        return "解析后没有内容（可能是空文件或纯图片）"
    return "解析失败：" + s[:80]


def _tmpdir():
    d = pathlib.Path(TMP_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run(cmd, timeout=600):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        print(f"    [命令异常] {' '.join(map(str, cmd))} -> {e}")
        return None


def _soffice(src, target_ext, outdir):
    """LibreOffice 无头转换：老格式 -> 新格式"""
    r = _run([
        "soffice", "--headless", "--norestore", "--nolockcheck",
        "-env:UserInstallation=file:///tmp/loprofile",
        "--convert-to", target_ext,
        "--outdir", str(outdir),
        str(src),
    ], timeout=300)
    out = pathlib.Path(outdir) / (pathlib.Path(src).stem + "." + target_ext)
    if out.exists():
        return out
    if r is not None and r.stderr:
        print(f"    [soffice stderr] {r.stderr.strip()[:300]}")
    return None


def _markitdown(path):
    from markitdown import MarkItDown
    md = MarkItDown()
    r = md.convert(str(path))
    return (r.text_content or "").strip()


def _pdf_pages(path):
    r = _run(["pdfinfo", str(path)], timeout=60)
    if r and r.stdout:
        for line in r.stdout.splitlines():
            if line.startswith("Pages:"):
                try:
                    return int(line.split()[-1])
                except Exception:
                    return 0
    return 0


def _pdf_to_images(path, outdir, dpi):
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    # 低性能平台上 pdftoppm 很慢（67 页 ≈ 数分钟），按页数给时间但要封顶，
    # 免得有问题的 PDF 把整批索引拖死（原来固定 900s，实测会超时）
    pages = _pdf_pages(path) or 1
    tmo = min(600, max(120, pages * 20))
    r = _run(["pdftoppm", "-r", str(dpi), "-png", str(path),
              str(pathlib.Path(outdir) / "page")], timeout=tmo)
    if r is None and pages > 0:
        raise RuntimeError(f"PDF 转图片超时（{pages} 页，限 {tmo} 秒）")
    return sorted(pathlib.Path(outdir).glob("page*.png"))


def _vlm_page(img_path):
    """把一页图片发给多模态模型，返回 Markdown。"""
    if not VLM_BASE_URL or not VLM_API_KEY:
        raise RuntimeError("未配置 VLM_BASE_URL / VLM_API_KEY")
    b64 = base64.b64encode(pathlib.Path(img_path).read_bytes()).decode()
    payload = {
        "model": VLM_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": OCR_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        "max_tokens": 4096,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {VLM_API_KEY}", "Content-Type": "application/json"}
    with httpx.Client(timeout=300) as c:
        r = c.post(f"{VLM_BASE_URL}/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    return (data["choices"][0]["message"]["content"] or "").strip()


def ocr_pdf(path, file_hash):
    """扫描件走多模态 OCR：逐页识别 + 磁盘缓存（重跑不重复花钱）。"""
    cache = pathlib.Path(OCR_CACHE) / file_hash
    cache.mkdir(parents=True, exist_ok=True)
    tmp = _tmpdir() / f"ocr_{file_hash[:12]}"

    imgs = _pdf_to_images(path, tmp, OCR_DPI)
    if not imgs:
        return "", 0

    parts, fresh = [], 0
    for i, img in enumerate(imgs, 1):
        cp = cache / f"p{i:04d}.md"
        if cp.exists():
            txt = cp.read_text(encoding="utf-8")
        else:
            try:
                txt = _vlm_page(img)
            except Exception as e:
                txt = f"（第 {i} 页识别失败：{human_error(e)}）"
            else:
                fresh += 1
            cp.write_text(txt, encoding="utf-8")
        parts.append(f"\n\n[第{i}页]\n\n{txt}")

    for f in imgs:
        try:
            f.unlink()
        except Exception:
            pass
    try:
        tmp.rmdir()
    except Exception:
        pass
    return "".join(parts).strip(), fresh


def _pptx_shapes(path):
    """逐 shape 抽 PPTX 文本，单个 shape 出错不影响整份。

    markitdown 的 PptxConverter 会对不认识的 shape 类型直接抛
    NotImplementedError，且没有 per-shape 容错反——一个怪图形就能让 280 多页的
    文档整份报废。这里只问「有没有 text_frame / table / chart」，
    不依赖 shape_type 判断，天然绕开那个坑。
    """
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    def walk(sh, out):
        try:
            if sh.has_text_frame and sh.text_frame.text.strip():
                out.append(sh.text_frame.text.strip())
        except Exception:
            pass
        try:
            if getattr(sh, "has_table", False) and sh.has_table:
                rows = []
                for row in sh.table.rows:
                    cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                    rows.append("| " + " | ".join(cells) + " |")
                if rows:
                    out.append("\n".join(rows))
        except Exception:
            pass
        try:
            if getattr(sh, "has_chart", False) and sh.has_chart:
                out.append(f"[图表: {sh.chart.chart_type}]")
        except Exception:
            pass
        try:
            if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
                for sub in sh.shapes:
                    walk(sub, out)
        except Exception:
            pass

    prs = Presentation(str(path))
    pages = []
    for i, slide in enumerate(prs.slides, 1):
        buf = []
        for sh in slide.shapes:
            walk(sh, buf)
        try:
            # 必须先判 has_notes_slide：直接读 .notes_slide 会给每页物化一个备注部件，
            # 280+ 页的 deck 会把内存顶爆（实测 1G 上限直接 OOM）。
            if getattr(slide, "has_notes_slide", False):
                notes = slide.notes_slide.notes_text_frame.text.strip()
            else:
                notes = ""
        except Exception:
            notes = ""
        if notes:
            buf.append("[备注] " + notes)
        if buf:
            pages.append(f"\n## 第 {i} 页\n\n" + "\n\n".join(buf))
    return "\n".join(pages).strip()


def extract_text(path):
    """统一入口。返回 (markdown 文本, 方式说明)。"""
    path = pathlib.Path(path)
    ext = path.suffix.lower()
    tmp = _tmpdir()

    # 1) 老格式：先转新格式
    if ext in OLD_FORMATS:
        conv = _soffice(path, OLD_FORMATS[ext], tmp)
        if conv is None:
            return "", "LibreOffice 转不出来（文件可能损坏或格式异常）"
        try:
            txt = _markitdown(conv)
        finally:
            try:
                conv.unlink()
            except Exception:
                pass
        return txt, f"libreoffice->{OLD_FORMATS[ext]}"

    # 2) 常规格式
    try:
        txt = _markitdown(path)
    except Exception as e:
        # 某些 Office 形状 markitdown 不认（实测：带特殊图形的 pptx 会抛
        # NotImplementedError: Shape instance of unrecognized shape type）
        # 降级①：PPTX 直接自己逐 shape 抽（快、无外部进程、单个 shape 坏了不影响整份）
        if ext == ".pptx":
            print(f"    [降级] markitdown 失败，改用 python-pptx 逐 shape 提取：{human_error(e, ext)}", flush=True)
            try:
                alt = _pptx_shapes(path)
            except Exception as e_pptx:
                alt = ""
                print(f"    [降级] python-pptx 也失败：{human_error(e_pptx, ext)}", flush=True)
            if len(alt.strip()) >= 30:
                return alt, "python-pptx 逐shape (降级)"
        # 降级②：用 LibreOffice 先转 PDF，再抽取文本
        if ext in (".pptx", ".ppt", ".docx", ".doc", ".xlsx", ".xls"):
            print(f"    [降级] 改走 LibreOffice 转 PDF：{human_error(e, ext)}", flush=True)
            conv = _soffice(path, "pdf", tmp)
            if conv is not None:
                try:
                    sub = _markitdown(conv)
                except Exception as e2:
                    sub = ""
                    print(f"    [降级] 转 PDF 后仍读不出文本：{human_error(e2, '.pdf')}", flush=True)
                finally:
                    try:
                        conv.unlink()
                    except Exception:
                        pass
                if len(sub.strip()) >= 30:
                    return sub, "libreoffice->pdf (降级)"
            return "", human_error(e, ext)
        return "", human_error(e, ext)

    # 3) PDF 文本过少 -> 判定扫描件，走多模态 OCR
    if ext == ".pdf":
        pages = _pdf_pages(path) or 1
        if len(txt) < OCR_MIN_CHARS * pages:
            h = hashlib.sha256(path.read_bytes()).hexdigest()[:32]
            ocr_txt, fresh = ocr_pdf(path, h)
            if ocr_txt:
                return ocr_txt, f"vlm-ocr ({pages}页, 本次新识别{fresh}页)"
            return txt, "扫描件，OCR 没能识别出文字"

    return txt, "markitdown"