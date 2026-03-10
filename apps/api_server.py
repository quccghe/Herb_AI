from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import os
from typing import Optional, Tuple

from agents.orchestrator import Orchestrator
from agents.planner_agent import PlannerAgent
from tools.qwen_client import QwenClient


class QueryReq(BaseModel):
    query: str


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

MCP_BASE_URL = os.environ.get("MCP_BASE_URL", "http://127.0.0.1:8001")
VEC_DIR = os.environ.get("VEC_DIR", "data/rag_out_vec")

_orch: Optional[Orchestrator] = None
_orch_err: Optional[str] = None


def get_orchestrator() -> Tuple[Optional[Orchestrator], Optional[str]]:
    global _orch, _orch_err
    if _orch is not None:
        return _orch, None
    if _orch_err:
        return None, _orch_err

    try:
        print("初始化 Orchestrator...")
        _orch = Orchestrator(
            vec_dir=VEC_DIR,
            mcp_url=MCP_BASE_URL,
            min_rag_score=0.28,
        )
        print("系统初始化完成")
        return _orch, None
    except Exception as e:
        _orch_err = str(e)
        return None, _orch_err


# planner 支持 LLM + 规则回退
try:
    planner = PlannerAgent(llm=QwenClient())
except Exception:
    planner = PlannerAgent(llm=None)


@app.get("/health")
def health():
    orch, orch_err = get_orchestrator()
    return {
        "ok": True,
        "planner_ready": True,
        "orchestrator_ready": orch is not None,
        "orchestrator_error": orch_err,
        "mcp_url": MCP_BASE_URL,
        "vec_dir": VEC_DIR,
    }


@app.post("/api/plan")
def api_plan(req: QueryReq):
    q = (req.query or "").strip()
    if not q:
        return {
            "ok": False,
            "error": "empty query",
            "intent": "herb",
            "target_page": "herb_page.html",
        }

    plan = planner.plan(q)
    return {"ok": True, **plan}


@app.post("/api/herb_full")
def api_herb(req: QueryReq):
    orch, orch_err = get_orchestrator()
    if orch is None:
        return {"ok": False, "error": f"orchestrator init failed: {orch_err}"}

    try:
        return orch.run_herb(req.query)
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/formula_full")
def api_formula(req: QueryReq):
    orch, orch_err = get_orchestrator()
    if orch is None:
        return {"ok": False, "error": f"orchestrator init failed: {orch_err}"}

    try:
        return orch.run_formula(req.query)
    except Exception as e:
        return {"ok": False, "error": str(e)}
