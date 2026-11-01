# RAGbuilding

一套**自托管的文档知识库**：把散落在硬盘里的 PDF / Word / PPT / Excel 变成可语义检索的知识库，并同时提供**网页管理台**和 **MCP 接口**，让 Claude、Cursor、ChatGPT 之类的 AI 客户端直接检索你的私有文档。

不上传云端、不需要 GPU、一台普通的 x86 小主机就能跑。

```
你放文件 → 自动解析/分块/嵌入 → 存入 Qdrant
                                        ↓
AI 客户端 ──MCP──┐                   混合检索（稠密向量 + BM25 + RRF + LLM 重排）
                 ├──→ kb-mcp ─────────────┘
浏览器 ───HTTP───┘
```

---

## 特性

- **结构化分块**：按标题层级、条款编号、表格边界断块，不是无脑按字数切
- **混合检索**：Qdrant 原生稠密向量 + BM25 全文索引，RRF 融合后接 LLM 重排
- **非对称嵌入**：查询侧加指令前缀、文档侧不加，符合 bge / qwen3-embedding 的设计
- **增量索引**：按文件 mtime + 内容指纹跳过未变化的文件，支持失败重试轮次
- **扫描件 OCR**：PDF 文本层太薄时自动走多模态模型转写（可选，不配则跳过）
- **嵌入主备兜底**：本地 Ollama 优先，挂了自动降级到 Cloudflare Workers AI，向量可混写
- **管理台**：文件浏览、上传、索引进度、分块可视化编辑、KB 分组、标签、AI 归档建议
- **MCP 原生**：自带 streamable-http MCP server，AI 客户端可直连

---

## 目录结构

```
.
├── app/                    # 主服务
│   ├── server.py           # Web API + 管理台 + MCP server
│   ├── index.py            # 索引器（解析→分块→上下文→嵌入→入库→归档建议）
│   ├── common.py           # Qdrant 连接、嵌入调用、混合检索
│   ├── chunking.py         # 结构化分块
│   ├── contextual.py       # 分块上下文补全（提升召回）
│   ├── extract.py          # 文档解析（含 LibreOffice / python-pptx 降级链）
│   ├── classify.py         # AI 归档建议
│   ├── rerank.py           # LLM 重排
│   ├── llm.py              # OpenAI 兼容 LLM 客户端
│   ├── reembed_all.py      # 全库重嵌工具（换嵌入模型时用）
│   ├── Dockerfile
│   └── web/index.html      # 管理台前端（Vue 3 + Element Plus，已本地化 vendor）
├── shim/
│   ├── shim.py             # 嵌入网关：Ollama 协议兼容，本地优先 / 云端兜底
│   └── check_consistency.py# 两种嵌入后端的一致性自检
├── compose.yml
├── .env.example
├── restore.sh              # 一键部署/迁移脚本
└── kb.sh                   # 日常操作快捷命令
```

---

## 快速开始

### 0. 前置要求

| 依赖 | 版本 | 说明 |
|---|---|---|
| Docker | 19.03+ | |
| docker compose | v2 | |
| Ollama | 任意 | 跑嵌入模型用，可以装在宿主机，也可以装在另一台机器 |
| 内存 | ≥ 8 GB | Qdrant + 应用 + 一次大文件解析 |

> 不需要 NVIDIA 显卡。嵌入模型用 CPU 跑即可。

### 1. 拉一个嵌入模型

```bash
ollama pull qwen3-embedding:0.6b     # 1024 维，中英文，约 640 MB
```

Ollama 默认只监听 `127.0.0.1`。容器要访问它，二选一：

- 让 Ollama 监听 `0.0.0.0`（`OLLAMA_HOST=0.0.0.0`），或
- 用 `compose.yml` 里已配好的 `host.docker.internal`（Linux 上需要 `extra_hosts` 映射 `host-gateway`，见注释）

### 2. 准备配置

```bash
git clone https://github.com/yayi-lei/RAGbuilding.git
cd RAGbuilding
cp .env.example .env
chmod 600 .env
```

编辑 `.env`，**至少**改这三项：

```bash
QDRANT_API_KEY=<随机长串>      # openssl rand -hex 32
KB_TOKEN=<随机长串>            # 给 AI 客户端用的 Bearer token
KB_ADMIN_PASS=<强密码>         # 网页登录密码
```

`OLLAMA_URL` 指向你的 Ollama；如果你就是本机装，默认值 `http://host.docker.internal:11434` 通常能用。

### 3. 放文档

```bash
mkdir -p docs
cp -r /path/to/your/documents/* docs/
```

支持的扩展名：`.pdf .docx .doc .xlsx .xls .pptx .ppt .md .txt .csv .html` 等（由 markitdown 提供）。

### 4. 起服务

```bash
docker compose up -d --build
```

三个常驻服务：

| 服务 | 作用 | 端口 |
|---|---|---|
| `qdrant` | 向量库 | `BIND_IP:6333` |
| `embed-shim` | 嵌入网关 | 仅容器网络内 8100 |
| `kb-mcp` | Web + API + MCP | `BIND_IP:8765` |

### 5. 建索引

```bash
# 首次全量（文档多的话挂后台跑）
docker compose --profile tools run --rm indexer

# 之后增量（也可以交给 cron，见下）
docker compose --profile tools run --rm indexer
```

索引器是**增量**的：已索引且未变化的文件会直接跳过。

### 6. 打开管理台

```
http://<你的IP>:8765/
```

用 `.env` 里的 `KB_ADMIN_USER` / `KB_ADMIN_PASS` 登录。

---

## 配置项

| 变量 | 默认 | 说明 |
|---|---|---|
| `BIND_IP` | `127.0.0.1` | 对外绑定地址。`0.0.0.0` 局域网可访问，`127.0.0.1` 只本机 |
| `QDRANT_API_KEY` | — | Qdrant 鉴权，**必须改** |
| `COLLECTION` | `office_docs` | 集合名 |
| `EMBED_MODEL` | `qwen3-embedding:0.6b` | 嵌入模型名，需与 Ollama 里的一致 |
| `EMBED_DIM` | `1024` | **必须**等于模型输出维度，改了要重建库 |
| `KB_ADMIN_USER` / `KB_ADMIN_PASS` | `admin` / 空 | 网页登录。密码为空时登录功能自动关闭 |
| `KB_TOKEN` | — | MCP / HTTP API 的 Bearer token，**必须改** |
| `EMBED_PRIORITY` | `local` | `local` = 本地优先；`cloudflare` = 云端优先 |
| `OLLAMA_URL` | `http://host.docker.internal:11434` | 本地 Ollama 地址 |
| `EMBED_FAIL_THRESHOLD` / `EMBED_COOLDOWN` | `3` / `120` | 连续失败几次熔断、熔断多少秒 |
| `CF_ENABLED` / `CF_ACCOUNT_ID` / `CF_API_TOKEN` / `CF_MODEL` | — | Cloudflare 兜底，不填则只用本地 |
| `VLM_BASE_URL` / `VLM_API_KEY` / `VLM_MODEL` | — | 扫描件 OCR 用的多模态网关，不填则跳过 OCR |
| `OCR_MIN_CHARS` / `OCR_DPI` | `50` / `150` | 判定扫描件的文本阈值、渲染 DPI |

---

## 嵌入后端：主备与手动切换

`embed-shim` 是一个 **Ollama 协议兼容**的网关，业务代码只认 `OLLAMA_URL`，换后端不用改一行代码。

```
          ┌─ /api/embed        → 按 EMBED_PRIORITY（默认本地优先，失败降级云端）
请求 ─────┼─ /local/api/embed   → 强制本地
          └─ /cf/api/embed      → 强制云端
```

- 每个后端**独立熔断**：连续失败 N 次后短路 `EMBED_COOLDOWN` 秒，避免每个请求都硬等超时
- 管理台顶部有两盏状态灯（本地 / Cloudflare），30 秒刷新一次，悬停可看延迟和失败原因
- 管理台的「增量索引」按钮旁可以**手动指定**本次索引用哪条通道；cron 定时任务永远走默认规则
- 云端兜底用的是 `@cf/qwen/qwen3-embedding-0.6b`，和本地同模型、同 1024 维，实测余弦相似度 **0.9994**，两边算的向量可以混写在同一个集合里

### 申请 Cloudflare Token（可选）

1. 打开 <https://dash.cloudflare.com/profile/api-tokens>
2. **Create Token** → 用 **Workers AI** 模板（或自定义）
3. Permissions 选 `Account` → `Workers AI` → `Read`
4. 复制 token 填进 `CF_API_TOKEN`
5. `account_id` 在控制台右侧栏，或地址栏 `dash.cloudflare.com/<这段>/...`

免费额度：**10,000 Neurons/天**，qwen3-embedding 约 **930 万 token/天**，个人知识库够用。

### 换嵌入模型 / 重嵌全库

```bash
# 改 .env 里的 EMBED_MODEL 与 EMBED_DIM 之后
docker compose exec kb-mcp python /app/reembed_all.py
```

它直接读 `payload.embed_text` 重算向量，**不重新解析文档、不重新 OCR**，几千块几分钟就能跑完。

---

## 使用方式

### 网页管理台

| 页面 | 能做什么 |
|---|---|
| 文件 | 浏览目录、上传、单文件/批量删除与重索引、查看分块并直接编辑 |
| 检索 | 语义+关键词混合检索，看命中原文、所属文档、章节定位 |
| 知识库 / 分组 / 标签 | 多 KB 隔离、分组整理、打标签；改了只动元数据，不触发重嵌 |
| 归档建议 | 索引完成后 AI 给出 KB/分组/标签建议，人工确认后应用 |
| 状态 | 向量总数、已索引/待索引文件数、最近索引时间、嵌入通道状态灯 |

### HTTP API

除首页外所有接口都要 `Authorization: Bearer <KB_TOKEN>`。

```bash
# 检索
curl -X POST http://localhost:8765/api/search \
  -H "Authorization: Bearer $KB_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query":"报销流程是怎样的","limit":5}'

# 触发增量索引（backend 可选 auto / local / cloudflare）
curl -X POST http://localhost:8765/api/index \
  -H "Authorization: Bearer $KB_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"backend":"auto"}'

# 看进度
curl http://localhost:8765/api/index/progress -H "Authorization: Bearer $KB_TOKEN"

# 看嵌入通道连通性
curl http://localhost:8765/api/backends -H "Authorization: Bearer $KB_TOKEN"
```

### MCP（给 AI 客户端用）

端点：`http://<host>:8765/mcp`（streamable-http，stateless）

```jsonc
// Cursor / Claude Desktop 类客户端的 mcp 配置示例
{
  "mcpServers": {
    "RAGbuilding": {
      "url": "http://127.0.0.1:8765/mcp",
      "headers": { "Authorization": "Bearer <你的 KB_TOKEN>" }
    }
  }
}
```

提供两个工具：

| 工具 | 作用 |
|---|---|
| `search_kb(query, top_k=6)` | 语义检索，返回最相关原文片段（含来源文档与章节） |
| `kb_stats()` | 查看向量总数、集合名、已索引文件数、最近索引时间 |

---

## 自动增量索引

```bash
# 每 10 分钟扫一次，用 flock 防止重叠
(crontab -l 2>/dev/null; \
 echo "*/10 * * * * /usr/bin/flock -n $PWD/data/state/index.lock \
  bash -c 'cd $PWD && /usr/bin/docker compose --profile tools run --rm indexer >> data/state/cron.log 2>&1'") | crontab -
```

没有新文件时索引器会直接跳过、一个嵌入请求都不发，所以空跑几乎零开销。

---

## 索引流水线

```
扫描文件 → 解析(markitdown) → 结构化分块(600字/120重叠)
        → 生成文档摘要与分块上下文（提升召回）
        → 嵌入（本地/云端）
        → 写入 Qdrant（稠密向量 + BM25 全文索引 + payload 元数据）
        → AI 归档建议（可选）
```

解析有多级降级，就是被真实文档揍出来的：

```
markitdown
  ├─ PPTX 失败 → python-pptx 逐 shape 抽取（跳过坏 shape，保住其余内容）
  └─ 再失败   → LibreOffice 转 PDF → 按 PDF 解析
PDF 文本过少 → 判定扫描件 → pdftoppm 渲染 → 多模态 OCR
```

---

## 已知限制

- **PPTX 解析吃内存**：`python-pptx` 加载 280 页的 deck 峰值约 1 GB，`kb-mcp` 的 `mem_limit` 已设 2 GB；再大的 PPT 需要按页拆分
- **LibreOffice 会静默失败**：某些 PPTX 转换返回码 0 但不产出文件，所以它只作为最后兜底，不作为主力
- **扫描件 OCR 依赖外部多模态网关**，没配就只对文本层 PDF 生效
- **单集合单向量**：换嵌入模型或改维度必须重建集合（`reembed_all.py` 或重新索引）
- **重排会调 LLM**，如果 `VLM_BASE_URL` 没配会自动退化为 RRF 顺序

---

## 安全须知

- `.env` 里有全部凭据，**永远不要提交**（`.gitignore` 已排除）
- `BIND_IP=0.0.0.0` 时请务必用强 `KB_TOKEN` 和 `KB_ADMIN_PASS`，否则等于把知识库裸奔在内网
- 若要暴露到公网，建议放在反向代理后面并加 HTTPS
- `KB_ADMIN_PASS` 留空会自动关闭网页登录（只剩 token 鉴权）

---

## License

MIT
