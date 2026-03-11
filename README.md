# Herb_AI：中医药多 Agent + MCP 服务系统

> 一个围绕“中药 / 方剂问答 + 拟人化展示 + 语音 + 图谱工具”的全栈项目。后端采用 **Multi-Agent 编排**，能力通过 **MCP 风格工具服务（FastAPI）** 统一暴露，前端通过页面路由实现中药与方剂两类场景的分流。

---

## 目录

- [1. 项目概览](#1-项目概览)
- [2. 系统架构](#2-系统架构)
- [3. 系统优点](#3-系统优点)
- [4. 目录结构](#4-目录结构)
- [5. 运行前准备](#5-运行前准备)
- [6. 启动与运行（实测流程）](#6-启动与运行实测流程)
- [7. 前后端如何衔接](#7-前后端如何衔接)
- [8. MCP 服务与工具说明（输入/输出/调用示例）](#8-mcp-服务与工具说明输入输出调用示例)
- [9. Multi-Agent 工作流详解](#9-multi-agent-工作流详解)
- [10. API 总览（业务网关）](#10-api-总览业务网关)
- [11. 常见问题与排查建议](#11-常见问题与排查建议)
- [12. 未来可改进方向](#12-未来可改进方向)

---

## 1. 项目概览

本项目主要解决四类问题：

1. **中药知识问答与展示**（卡片化结构 + 人格化叙述）；
2. **方剂结构化讲解**（基于 RAG 证据产出 JSON）；
3. **多模态输出**（角色图像生成 + 语音合成）；
4. **知识图谱查询与可视化探索**（节点、邻居、路径、子图、摘要）。

项目当前包含两层后端：

- **业务网关层（apps/api_server.py）**：对前端提供 `/api/plan`、`/api/herb_full`、`/api/formula_full`；
- **MCP 工具层（mcp_server/app.py）**：封装 Neo4j、TTS、图像生成等能力，供 Agent 调用。

---

## 2. 系统架构

### 2.1 总体架构图（逻辑）

```text
[Web 前端页面]
   ├─ index.html（入口 + 意图分流）
   ├─ herb_page.html（中药展示）
   ├─ formula_page.html（方剂展示页）
   └─ kg_graph_demo.html（图谱探索）
            │
            ▼ HTTP
[业务 API 服务: apps/api_server.py :8010]
            │
            ▼
[Orchestrator 多 Agent 编排]
   ├─ VectorStore(HNSW) 检索
   ├─ QwenClient 生成
   ├─ EntityExtractor 实体抽取
   ├─ HerbCardAgent / FormulaAgent
   ├─ FlavorStyleAgent / PersonaAgent
   ├─ TTSAgent / ImageAgent
            │
            ▼ HTTP
[MCP Server: mcp_server/app.py :8001]
   ├─ KG 工具（Neo4j）
   ├─ 图谱子图/路径/摘要工具
   ├─ Voice Design + Realtime TTS
   └─ wan2.6 文生图
            │
            ├─ Neo4j
            ├─ DashScope 音频/图像服务
            └─ assets/ 本地媒体落盘
```

### 2.2 核心设计点

- **检索增强（RAG）先行**：每次问答优先检索向量库，低相关时直接返回资料不足，降低幻觉风险。
- **能力拆分为 Agent**：卡片抽取、风格叙述、人格设定、TTS、文生图分别由独立 Agent 处理。
- **MCP 化工具调用**：图谱、音频、图像能力以 HTTP 工具接口统一，便于替换底层实现。
- **静态资源统一挂载**：`/assets` 同时服务图片和音频，前端直接可播可渲染。

---

## 3. 系统优点

1. **模块化强**：编排器和具体能力分层清晰，便于替换某个 Agent 或 MCP 工具。
2. **工程可扩展**：新增工具只需在 MCP Server 增加路由 + 客户端封装，不影响整体流程。
3. **结果结构稳定**：前端主要依赖 JSON 结构，不依赖 LLM 原始文本格式。
4. **多模态闭环**：同一查询可同时输出文本、语音、图像，体验更完整。
5. **图谱增强**：除单点问答，还支持路径、子图、摘要等解释能力。
6. **降级策略明确**：KG、TTS、图像异常时尽量返回可展示结果，不直接崩溃。

---

## 4. 目录结构

```text
Herb_AI/
├─ agents/                # 各类 Agent
├─ apps/                  # 业务 API 服务
├─ mcp_server/            # MCP 工具服务
├─ tools/                 # 客户端与通用工具（RAG、LLM、MCP 调用）
├─ data/rag_out_vec/      # 向量索引与分块数据
├─ web/                   # 前端页面
├─ assets/                # 图片/音频输出目录
├─ master_agent.py        # 命令行交互入口
└─ 启动说明.txt
```

---

## 5. 运行前准备

### 5.1 Python 依赖

```bash
pip install -r requirements.txt
```

### 5.2 环境变量

建议复制 `.env.example` 并按本机环境填写：

- `OPENAI_API_KEY` / `DASHSCOPE_API_KEY`
- `OPENAI_BASE_URL`（DashScope OpenAI 兼容地址）
- `OPENAI_MODEL`
- `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`
- `MCP_BASE_URL`

### 5.3 数据与模型注意事项

当前 `data/rag_out_vec/index_meta.json` 中 embedding 模型路径是 Windows 本地路径（`G:\models\bge-small-zh-v1.5`）。在 Linux/macOS 上若无同路径会导致 API 启动失败，需要：

- 改为可访问的本地模型目录，或
- 改为可下载的模型名（如 HuggingFace 名称）。

---

## 6. 启动与运行（实测流程）

项目提供的启动参考：

```text
uvicorn mcp_server.app:app --port 8001
python master_agent.py --vec_dir data\rag_out_vec --mcp http://127.0.0.1:8001
```

推荐完整本地启动顺序：

1. 启动 MCP 工具服务（8001）；
2. 启动业务 API（8010）；
3. 打开 `web/index.html`（或用静态服务托管 `web/`）；
4. 从首页输入中药/方剂，自动分流到对应页面。

---

## 7. 前后端如何衔接

### 7.1 入口分流（index -> herb/formula）

`web/index.html` 前端先调用：

- `POST http://127.0.0.1:8010/api/plan`，请求体：`{"query":"当归"}`

返回：

- `{"intent":"herb"}` -> 跳转 `herb_page.html?name=当归`
- `{"intent":"formula"}` -> 跳转 `formula_page.html?name=四君子汤`

### 7.2 中药页聚合渲染（herb_page）

`web/herb_page.html` 调用：

- `POST /api/herb_full`

业务 API 内部执行 Orchestrator，返回统一结果（节选）：

```json
{
  "ok": true,
  "type": "herb",
  "card": {...},
  "narration": {...},
  "persona": {...},
  "tts": {"audio_path": "assets/audio/..."},
  "image": {"image_path": "assets/images/..."}
}
```

前端将 `assets/...` 自动拼接为 `http://127.0.0.1:8010/assets/...` 播放与展示。

### 7.3 图谱页直连 MCP（kg_graph_demo）

`web/kg_graph_demo.html` 直接调用 MCP：

- `POST :8001/tools/kg_subgraph`
- `POST :8001/tools/kg_relation_paths`

即：图谱 Demo 并不经过业务 API，而是前端直连工具服务。

---

## 8. MCP 服务与工具说明（输入/输出/调用示例）

MCP 服务入口：`mcp_server/app.py`（默认 `http://127.0.0.1:8001`）

### 8.1 基础健康检查

#### `GET /health`
- **用途**：检查 Neo4j 连通性（内部执行 `RETURN 1`）。
- **输出**：
  - 成功：`{"ok": true}`
  - 失败：`{"ok": false, "error": "..."}`

---

### 8.2 知识图谱基础工具

#### 1) `POST /tools/kg_get_node`
- **输入**：
```json
{"name":"当归"}
```
- **输出（命中）**：
```json
{"found": true, "name": "当归", "props": {...}}
```
- **输出（未命中/异常）**：
```json
{"found": false, "name": "当归", "error": "..."}
```

#### 2) `POST /tools/kg_neighbors`
- **输入**：
```json
{
  "name":"当归",
  "limit":30,
  "rel_types":["HAS_EFFICACY"],
  "neighbor_labels":["Efficacy"]
}
```
- **输出**：
```json
{
  "found": true,
  "name":"当归",
  "neighbors":[
    {"rel":"HAS_EFFICACY","labels":["Efficacy"],"node":{...}}
  ]
}
```

#### 3) `POST /tools/kg_paths`
- **输入**：
```json
{"a":"当归","b":"川芎","k":3,"max_hops":3}
```
- **输出**：
```json
{
  "found": true,
  "a":"当归",
  "b":"川芎",
  "paths":[{"nodes":[...],"rels":[...]}]
}
```

---

### 8.3 图谱可视化增强工具

#### 4) `POST /tools/kg_subgraph`
- **输入**：
```json
{"name":"川芎","depth":1,"include_types":["Efficacy"],"max_nodes_per_type":20}
```
- **输出**：
```json
{"ok": true, "name":"川芎", "nodes":[...], "edges":[...], "stats": {...}}
```

#### 5) `POST /tools/kg_relation_paths`
- **输入**：
```json
{"source":"川芎","target":"当归","max_hops":3}
```
- **输出**：
```json
{"ok": true, "source":"川芎", "target":"当归", "paths":[...], "count":1}
```

#### 6) `POST /tools/kg_graph_summary`
- **输入**：
```json
{"name":"川芎"}
```
- **输出**：
```json
{"ok": true, "name":"川芎", "summary":"...", "highlights":[...]}
```

---

### 8.4 音频工具（Qwen Voice Design + Realtime TTS）

#### 7) `POST /tools/voice_design_create`
- **输入**：
```json
{
  "voice_prompt":"中文成年中性声音，语速适中，吐字清晰。",
  "preview_text":"你好，我来介绍这味中药。",
  "preferred_name":"danggui",
  "target_model":"qwen3-tts-vd-realtime-2026-01-15",
  "language":"zh",
  "sample_rate":24000,
  "response_format":"wav",
  "use_cache":true
}
```
- **输出**：
```json
{
  "ok": true,
  "voice": "...",
  "target_model":"...",
  "preferred_name":"...",
  "preview_audio_b64":"...",
  "cached": false
}
```

#### 8) `POST /tools/tts_vd_realtime_speak`
- **输入**：
```json
{
  "text":"我是当归，擅长补血活血。",
  "voice":"voice_xxx",
  "model":"qwen3-tts-vd-realtime-2026-01-15",
  "sample_rate":24000
}
```
- **输出（成功）**：
```json
{"ok": true, "model":"...", "voice":"...", "audio_path":"assets/audio/tts_xxx.wav", "bytes":12345}
```
- **输出（失败）**：
```json
{"ok": false, "error":"..."}
```

---

### 8.5 图像工具（wan2.6 文生图）

#### 9) `POST /tools/wan_text_to_image`
- **输入**：
```json
{
  "prompt":"中药拟人化角色，温和沉稳，汉服风格",
  "herb_name":"当归",
  "size":"1024*1024",
  "style_hint":"中国风中药拟人化角色设定图...",
  "watermark":false
}
```
- **输出**：
```json
{
  "ok": true,
  "model":"wan2.6-t2i",
  "image_url":"https://...",
  "image_urls":["https://..."],
  "image_path":"assets/images/当归_xxx.png",
  "usage":{},
  "request_id":"..."
}
```

---

## 9. Multi-Agent 工作流详解

### 9.1 入口编排（Orchestrator）

- 先判断是否方剂（后缀：汤/散/丸/饮/膏/丹/剂）；
- 方剂 -> `FormulaAgent.run(query)`；
- 否则 -> 中药流程：
  1. 代词与别名解析（`resolve_query`）；
  2. RAG 预检索，给实体抽取提供 hint；
  3. LLM 实体抽取（药名/方剂名）；
  4. 主检索并做低分阈值拦截；
  5. 调用 `kg_get_node`；
  6. 生成卡片 `HerbCardAgent`；
  7. 生成自述 `FlavorStyleAgent`；
  8. 生成人设 `PersonaAgent`；
  9. TTS 合成 `TTSAgent`；
  10. 图像生成 `ImageAgent`。

### 9.2 方剂流程

`FormulaAgent` 使用 RAG 证据构建 Prompt，要求 LLM 输出严格 JSON。若 JSON 解析失败，返回兜底结构 + 原始 evidence，保证前端可渲染。

---

## 10. API 总览（业务网关）

服务文件：`apps/api_server.py`（默认端口 `8010`）

### 10.1 `GET /health`
- 返回：`{"ok": true}`

### 10.2 `POST /api/plan`
- 入参：`{"query":"四君子汤"}`
- 出参：`{"intent":"formula"}` 或 `{"intent":"herb"}`

### 10.3 `POST /api/herb_full`
- 入参：`{"query":"当归"}`
- 出参：中药全链路聚合结果（卡片 + 叙述 + 人设 + 语音 + 图像）

### 10.4 `POST /api/formula_full`
- 入参：`{"query":"四君子汤"}`
- 出参：方剂结构化结果

---

## 11. 常见问题与排查建议

1. **API 服务启动即报错 `Path G:\models\... not found`**
   - 原因：向量索引元数据绑定了 Windows 本地模型路径。
   - 处理：修改 `data/rag_out_vec/index_meta.json` 的 `embedding_model/model` 到当前机器可用路径。

2. **MCP `/health` 返回 neo4j connection refused**
   - 原因：Neo4j 未启动或连接参数不对。
   - 处理：确认 `NEO4J_URI/USER/PASSWORD` 与服务状态。

3. **TTS / 文生图报 key 缺失**
   - 原因：`DASHSCOPE_API_KEY` 未配置。
   - 处理：补齐环境变量，或在不需要多模态时允许降级。

4. **前端能打开但无音频/图片**
   - 检查 `assets` 是否已挂载、返回路径是否 `assets/...`、跨域是否开启。

---

## 12. 未来可改进方向

- 把 `formula_page.html` 替换为真正的方剂专属渲染逻辑（当前与中药页几乎一致）。
- 增加统一配置中心（.env + pydantic settings），避免硬编码端口。
- 增加端到端测试（前端 -> API -> MCP）与契约测试（JSON schema）。
- 为 MCP 工具补充 OpenAPI 文档页与示例调用脚本。
- 将 RAG 索引构建脚本纳入标准化流水线（含模型可移植配置）。

---

如果你要快速验证“最小可用链路”，建议先启动 MCP（即便 Neo4j 暂时不可用），再修正向量模型路径后启动业务 API，用 `web/index.html` 走一次完整查询。
