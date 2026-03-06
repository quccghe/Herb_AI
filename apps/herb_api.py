# -*- coding: utf-8 -*-
import os
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.orchestrator import Orchestrator

app = FastAPI(title="HerbAI API", version="0.1.0")

# 允许前端直接访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态资源挂载（图片/音频）
if os.path.exists("assets"):
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")


# ====== 初始化 Orchestrator ======
# 这里默认读取你当前项目的配置
MCP_BASE_URL = os.environ.get("MCP_BASE_URL", "http://127.0.0.1:8001")
VEC_DIR = os.environ.get("VEC_DIR", "data/rag_out_vec")

orch = Orchestrator(
    vec_dir=VEC_DIR,
    mcp_url=MCP_BASE_URL,
    min_rag_score=0.28
)


# ====== 请求模型 ======
class HerbFullReq(BaseModel):
    query: str


class HealthResp(BaseModel):
    ok: bool
    message: str
    mcp_url: str
    vec_dir: str


# ====== 路由 ======
@app.get("/health", response_model=HealthResp)
def health():
    return HealthResp(
        ok=True,
        message="HerbAI API is running",
        mcp_url=MCP_BASE_URL,
        vec_dir=VEC_DIR
    )


@app.post("/api/herb_full")
def herb_full(req: HerbFullReq):
    """
    前端聚合接口：
    输入中药名，返回完整多Agent结果
    """
    try:
        q = (req.query or "").strip()
        if not q:
            return {"ok": False, "error": "empty query"}

        result = orch.run_once(q)
        return result

    except Exception as e:
        return {"ok": False, "error": str(e)}