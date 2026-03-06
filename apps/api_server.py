from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import os

from agents.orchestrator import Orchestrator


# ===============================
# 请求模型
# ===============================

class QueryReq(BaseModel):
    query: str


# ===============================
# 初始化系统
# ===============================

print("初始化 Orchestrator...")

orch = Orchestrator(
    vec_dir="data/rag_out_vec",
    mcp_url="http://127.0.0.1:8001"
)

print("系统初始化完成")


# ===============================
# FastAPI
# ===============================

app = FastAPI(title="TCM Multi-Agent API")


# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===============================
# 静态资源
# ===============================

if not os.path.exists("assets"):
    os.makedirs("assets")

app.mount("/assets", StaticFiles(directory="assets"), name="assets")


# ===============================
# 健康检查
# ===============================

@app.get("/health")
def health():
    return {"ok": True}


# ===============================
# Planner接口
# ===============================

FORMULA_SUFFIX = ["汤", "散", "丸", "饮", "膏", "丹", "剂"]


def is_formula(text: str):

    for s in FORMULA_SUFFIX:
        if text.endswith(s):
            return True

    return False


@app.post("/api/plan")
def api_plan(req: QueryReq):

    q = req.query.strip()

    if is_formula(q):
        return {"intent": "formula"}

    return {"intent": "herb"}


# ===============================
# 中药完整查询
# ===============================

@app.post("/api/herb_full")
def api_herb(req: QueryReq):

    try:

        result = orch.run_herb(req.query)

        return result

    except Exception as e:

        return {
            "ok": False,
            "error": str(e)
        }


# ===============================
# 方剂查询
# ===============================

@app.post("/api/formula_full")
def api_formula(req: QueryReq):

    try:

        result = orch.run_formula(req.query)

        return result

    except Exception as e:

        return {
            "ok": False,
            "error": str(e)
        }