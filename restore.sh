#!/bin/bash
# ============================================================
#  rag-stack 一键重建脚本（换机器用）
#
#  用法：
#    1) 把整个 rag-stack-backup-YYYYMMDD 目录拷到新机器
#    2) cd 进去
#    3) sudo bash restore.sh
#
#  前提（脚本会检查）：
#    - Linux 系统（amd64 或 arm64 都行）
#    - Docker + docker-compose
#    - Ollama（如果没有，脚本会提示装）
# ============================================================
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET="${TARGET:-$HOME/rag-stack}"
TSIP="${TSIP:-}"

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; CYN=$'\033[36m'; RST=$'\033[0m'
ok()   { echo "  ${GRN}✅${RST} $*"; }
warn() { echo "  ${YEL}⚠️ ${RST} $*"; }
bad()  { echo "  ${RED}❌${RST} $*"; }
inf()  { echo "  ${CYN}→${RST} $*"; }

echo "============================================================"
echo "  rag-stack 一键重建"
echo "  源目录: $HERE"
echo "  目标:   $TARGET"
echo "============================================================"
echo

# ---------- 阶段 0：环境检查 ----------
echo "[0/8] 环境检查"
FAIL=0

if command -v docker >/dev/null 2>&1; then
  ok "docker: $(docker --version | head -1)"
else
  bad "没有 docker —— 请先装 Docker"
  FAIL=1
fi

if docker compose version >/dev/null 2>&1; then
  ok "docker compose（插件版）"
  DC="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  ok "docker-compose: $(docker-compose --version | head -1)"
  DC="docker-compose"
else
  bad "没有 docker compose / docker-compose"
  FAIL=1
fi

if command -v ollama >/dev/null 2>&1; then
  ok "ollama: $(ollama --version 2>/dev/null | head -1)"
  HAS_OLLAMA=1
else
  warn "没有 ollama —— 嵌入模型跑不起来"
  inf "安装：curl -fsSL https://ollama.com/install.sh | sh"
  HAS_OLLAMA=0
fi

if ! docker info >/dev/null 2>&1; then
  bad "docker 守护进程没运行（或没权限）—— 试试 sudo systemctl start docker"
  FAIL=1
fi

[ "$FAIL" = "1" ] && { echo; bad "环境不满足，先解决上面问题再跑"; exit 1; }
echo

# ---------- 阶段 1：确定绑定 IP ----------
echo "[1/8] 确定服务绑定 IP"
if [ -z "$TSIP" ]; then
  if command -v tailscale >/dev/null 2>&1; then
    TSIP=$(tailscale ip -4 2>/dev/null | head -1)
  fi
fi
if [ -z "$TSIP" ]; then
  warn "拿不到 Tailscale IP"
  inf "将使用 0.0.0.0（所有网卡）—— 安全性降低，但能用"
  TSIP="0.0.0.0"
fi
ok "绑定 IP: $TSIP"
echo

# ---------- 阶段 2：部署代码 ----------
echo "[2/8] 部署代码到 $TARGET"
mkdir -p "$TARGET"
for item in app shim compose.yml kb.sh README.md LICENSE .env.example; do
  if [ -e "$HERE/$item" ]; then
    cp -r "$HERE/$item" "$TARGET/" && inf "$item"
  fi
done
ok "代码部署完成"
echo

# ---------- 阶段 3：配置 .env ----------
echo "[3/8] 配置 .env"
if [ -f "$TARGET/.env" ]; then
  ok "已存在 .env，保留不动"
elif [ -f "$TARGET/.env.example" ]; then
  cp "$TARGET/.env.example" "$TARGET/.env"
  sed -i "s|^BIND_IP=.*|BIND_IP=$TSIP|" "$TARGET/.env"
  warn "已用模板生成 .env —— 请手工填写 QDRANT_API_KEY / KB_TOKEN / KB_ADMIN_PASS"
fi
inf "当前 .env："
grep -vE '^\s*#|^\s*$' "$TARGET/.env" 2>/dev/null | sed 's/^/      /'
echo

# ---------- 阶段 4：恢复元数据 ----------
echo "[4/8] 恢复元数据（分组/标签/耗时基线）"
mkdir -p "$TARGET/data/state"
for f in kbmeta.json tags.json timing.json candidates.json index_state.json; do
  if [ -s "$HERE/data/state/$f" ]; then
    cp "$HERE/data/state/$f" "$TARGET/data/state/$f"
    inf "$f ($(stat -c%s "$HERE/data/state/$f") 字节)"
  fi
done
ok "元数据恢复完成"
echo

# ---------- 阶段 5：恢复文档 ----------
echo "[5/8] 恢复文档"
if [ -d "$HERE/docs" ]; then
  mkdir -p "$TARGET/docs"
  cp -r "$HERE/docs/." "$TARGET/docs/"
  ok "$(find "$TARGET/docs" -type f | wc -l) 个文件，$(du -sh "$TARGET/docs" | cut -f1)"
else
  warn "备份里没有 docs —— 需要你自己拷文档进去"
fi
mkdir -p "$TARGET/data/ocr_cache" "$TARGET/data/tmp"
echo

# ---------- 阶段 6：准备 Ollama 模型 ----------
echo "[6/8] 准备嵌入模型"
if [ "$HAS_OLLAMA" = "1" ]; then
  MODEL=$(grep -oP '^EMBED_MODEL=\K.*' "$TARGET/.env" 2>/dev/null | tr -d '"' )
  MODEL=${MODEL:-qwen3-embedding:0.6b}
  if ollama list 2>/dev/null | grep -q "${MODEL%%:*}"; then
    ok "$MODEL 已在本地"
  else
    inf "拉取 $MODEL（约 640MB，需要联网）"
    ollama pull "$MODEL" || warn "拉取失败，稍后手工执行：ollama pull $MODEL"
  fi
  # 确保 ollama 在跑
  if ! curl -s --max-time 5 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    warn "ollama 服务没响应，试着启动"
    (sudo systemctl start ollama 2>/dev/null || nohup ollama serve >/tmp/ollama.log 2>&1 &) || true
    sleep 5
  fi
else
  warn "跳过（没有 ollama）"
fi
echo

# ---------- 阶段 7：构建并启动容器 ----------
echo "[7/8] 构建并启动容器"
cd "$TARGET"
inf "构建镜像（第一次要几分钟）"
$DC build 2>&1 | tail -3 | sed 's/^/      /'
inf "启动服务"
$DC up -d 2>&1 | tail -5 | sed 's/^/      /'
sleep 15
inf "容器状态："
$DC ps 2>/dev/null | sed 's/^/      /'
echo

# ---------- 阶段 8：恢复向量库快照 ----------
echo "[8/8] 恢复向量库"
SNAP="$HERE/qdrant/office_docs.snapshot"
if [ -s "$SNAP" ] && docker ps --format '{{.Names}}' | grep -q '^qdrant$'; then
  QK=$(grep -oP '^QDRANT_API_KEY=\K.*' "$TARGET/.env" 2>/dev/null | tr -d '"')
  COLL=$(grep -oP '^COLLECTION=\K.*' "$TARGET/.env" 2>/dev/null | tr -d '"')
  COLL=${COLL:-office_docs}
  BIND=$(grep -oP '^BIND_IP=\K.*' "$TARGET/.env" 2>/dev/null | tr -d '"')
  [ "$BIND" = "0.0.0.0" ] && BIND="127.0.0.1"
  B="http://$BIND:6333"

  inf "上传快照到 qdrant 容器"
  docker cp "$SNAP" qdrant:/tmp/restore.snapshot 2>&1 | sed 's/^/      /'

  inf "从快照恢复集合 $COLL"
  RESP=$(curl -s --max-time 600 -X POST \
    -H "api-key: $QK" -H "Content-Type: multipart/form-data" \
    -F "snapshot=@$SNAP" \
    "$B/collections/$COLL/snapshots/upload" 2>/dev/null)
  echo "$RESP" | head -c 300 | sed 's/^/      /'
  echo

  sleep 3
  inf "验证："
  curl -s --max-time 15 -H "api-key: $QK" "$B/collections/$COLL" 2>/dev/null | \
    python3 -c "
import sys,json
try:
    d=json.load(sys.stdin).get('result',{})
    print('      status: %s | vectors: %s' % (d.get('status'), d.get('points_count')))
except Exception as e:
    print('      读不到:', e)" 2>/dev/null
else
  warn "没有快照 —— 需要在新机器上重新索引"
  inf "执行：cd $TARGET && $DC --profile tools run --rm indexer"
  inf "（约 3~5 小时，夜里挂着就行）"
fi
echo

# ---------- 收尾 ----------
echo "============================================================"
echo "  重建完成"
echo "============================================================"
TSHOST=$( [ "$TSIP" = "0.0.0.0" ] && echo "127.0.0.1" || echo "$TSIP" )
KBT=$(grep -oP '^KB_TOKEN=\K.*' "$TARGET/.env" 2>/dev/null | tr -d '"')
KBA=$(grep -oP '^KB_ADMIN_USER=\K.*' "$TARGET/.env" 2>/dev/null | tr -d '"')
KBP=$(grep -oP '^KB_ADMIN_PASS=\K.*' "$TARGET/.env" 2>/dev/null | tr -d '"')

echo "  管理台:  http://$TSHOST:8765/"
echo "  登录:    ${KBA:-admin} / ${KBP:-<见 .env>}"
echo "  MCP:     http://$TSHOST:8765/mcp"
echo "  Qdrant:  http://$TSHOST:6333"
echo
echo "  接下来要做的："
echo "    1. 装 cron 自动索引："
echo "       (crontab -l 2>/dev/null; echo \"*/10 * * * * /usr/bin/flock -n $TARGET/data/state/index.lock bash -c 'cd $TARGET && $DC --profile tools run --rm indexer >> data/state/cron.log 2>&1'\") | crontab -"
echo "    2. 防休眠（如果是桌面系统）：sudo bash $TARGET/keepalive-fix.sh"
echo "    3. 浏览器打开管理台验证"
echo
