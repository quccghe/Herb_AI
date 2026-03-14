from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

from agents.orchestrator import Orchestrator
from agents.planner_agent import PlannerAgent


class QueryReq(BaseModel):
    query: str


print("初始化 Orchestrator...")
orch = Orchestrator(
    vec_dir="data/rag_out_vec",
    mcp_url="http://127.0.0.1:8001"
)
planner = PlannerAgent(orch.llm)
print("系统初始化完成")

app = FastAPI(title="TCM Multi-Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists("assets"):
    os.makedirs("assets")

app.mount("/assets", StaticFiles(directory="assets"), name="assets")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/plan")
def api_plan(req: QueryReq):
    return planner.plan(req.query)


@app.post("/api/herb_full")
def api_herb(req: QueryReq):
    try:
        return orch.run_herb(req.query)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/formula_full")
def api_formula(req: QueryReq):
    try:
        # 这里是主流程：RAG -> LLM结构化 -> 写 JSON -> 返回结果
        return orch.run_formula(req.query)
    except Exception as e:
        return {"ok": False, "error": str(e)}