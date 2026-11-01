#!/usr/bin/env bash
# kb.sh —— 知识库日常操作封装（放在项目根目录）
set -e
cd "$(dirname "$0")"

if docker compose version >/dev/null 2>&1; then
  C="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  C="docker-compose"
else
  echo "需要 docker compose v2 或 docker-compose" >&2
  exit 1
fi

case "${1:-help}" in
  up)      $C up -d ;;
  down)    $C down ;;
  restart) $C restart ;;
  ps)      $C ps ;;
  logs)    $C logs -f --tail=100 "${2:-kb-mcp}" ;;
  index)   echo "== 开始增量索引 =="; $C --profile tools run --rm indexer ;;
  pull)    ${OLLAMA_HOST:-ollama} pull "${2:-qwen3-embedding:0.6b}" ;;
  models)  ${OLLAMA_HOST:-ollama} list ;;
  stats)   $C exec kb-mcp python -c "
from common import collection_count, COLLECTION
print('集合:', COLLECTION); print('向量数:', collection_count())" ;;
  search)  shift; $C exec kb-mcp python -c "
import sys
from common import search
for h in search(sys.argv[1], top_k=int(sys.argv[2]) if len(sys.argv)>2 else 5):
    p = h.payload or {}
    print('---')
    print('出处:', p.get('source'), '| 页:', p.get('page'), '| 块:', p.get('chunk'), '| 相似度:', round(getattr(h,'score',0),4))
    print((p.get('text') or '')[:300])
" "$@" ;;
  backup)  ts=$(date +%F-%H%M); tar czf "kb-backup-$ts.tar.gz" data/ docs/ 2>/dev/null; echo "已备份: kb-backup-$ts.tar.gz" ;;
  test)    curl -s -o /dev/null -w "管理页 HTTP %{http_code}\n" "http://127.0.0.1:8765/" ;;
  *) cat <<EOF
知识库操作台  kb.sh

  ./kb.sh up            启动服务
  ./kb.sh down          停止服务
  ./kb.sh restart       重启
  ./kb.sh ps            容器状态
  ./kb.sh logs [服务]   看日志（默认 kb-mcp）
  ./kb.sh index         跑一次增量索引
  ./kb.sh pull [模型]   拉取嵌入模型（走宿主机的 ollama 命令）
  ./kb.sh models        列出已装的 ollama 模型
  ./kb.sh stats         看库规模
  ./kb.sh search "关键词" [top_k]   命令行检索测试
  ./kb.sh backup        备份 data/ 和 docs/
  ./kb.sh test          自检管理页是否响应
EOF
  ;;
esac
