FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    UV_HTTP_TIMEOUT=300

# 1) 换中科大源（实测 9.45MB/s；阿里云 debian 在本机被限到 112KB/s）
#    + 强制 IPv4 + 提高 apt 并发
RUN set -e; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i -e 's|deb.debian.org|mirrors.ustc.edu.cn|g' \
               -e 's|security.debian.org|mirrors.ustc.edu.cn/debian-security|g' \
               /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i -e 's|deb.debian.org|mirrors.ustc.edu.cn|g' \
               -e 's|security.debian.org|mirrors.ustc.edu.cn/debian-security|g' \
               /etc/apt/sources.list; \
    fi; \
    printf 'Acquire::ForceIPv4 "true";\nAcquire::http::Pipeline-Depth "10";\nAcquire::Retries "3";\nAcquire::http::Timeout "30";\n' > /etc/apt/apt.conf.d/99tune

# 2) 系统依赖
#    libmagic1     : markitdown 识别文件类型
#    poppler-utils : PDF → PNG（扫描件 OCR）
#    libreoffice-* : 老格式 .doc/.xls/.ppt → 新格式
#    curl          : 下载 uv
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagic1 \
        poppler-utils \
        libreoffice-writer \
        libreoffice-calc \
        libreoffice-impress \
        file \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 3) 装 uv —— 关键！pip 是 Python 写的，在低性能 ARM 平台上网络 I/O 极慢，
#    而 uv 是 Rust 写的，同一个包 15MB 只需 1.7 秒（约 8.8MB/s），实测装 numpy 快 39 倍。
RUN curl -LsS -o /tmp/uv.tgz \
        "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz" \
    && tar xzf /tmp/uv.tgz -C /usr/local/bin --strip-components=1 \
    && rm -f /tmp/uv.tgz \
    && uv --version

WORKDIR /app

# 4) Python 依赖：用 uv 装（独立层，改代码不用重装）
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt \
        --index-url https://mirrors.aliyun.com/pypi/web/simple

COPY . .

EXPOSE 8765
CMD ["python", "server.py"]