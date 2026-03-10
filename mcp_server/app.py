# -*- coding: utf-8 -*-
"""MCP Server (FastAPI) for Knowledge Graph tools.

Tools:
- POST /tools/kg_get_node
- POST /tools/kg_neighbors
- POST /tools/kg_paths

Run:
  pip install -r requirements.txt
  uvicorn mcp_server.app:app --host 0.0.0.0 --port 8001
"""

# app.py 顶部 import
from mcp_server.tools.qwen_audio_tools import voice_design_create, tts_vd_realtime_speak
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
import os

from mcp_server.tools.kg_tools import (
    init_driver,
    tool_health,
    tool_kg_get_node,
    tool_kg_neighbors,
    tool_kg_paths,
)

from fastapi.staticfiles import StaticFiles
import os

from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from mcp_server.tools.kg_graph_tools import (
    tool_kg_subgraph,
    tool_kg_relation_paths,
    tool_kg_graph_summary,
)

from mcp_server.tools.wan_image_tools import tool_wan_text_to_image
from mcp_server.tools.formula_tools import tool_formula_fallback, tool_formula_story_refine

DEFAULT_TARGET_MODEL = "qwen3-tts-vd-realtime-2026-01-15"

app = FastAPI(title="HerbAI MCP Server", version="0.1.0")
# 挂载静态资源目录，供前端访问图片/音频
if os.path.exists("assets"):
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# init neo4j driver
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

init_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)


class KgGetNodeReq(BaseModel):
    name: str = Field(..., description="Medicine name (exact match preferred)")


class KgNeighborsReq(BaseModel):
    name: str
    limit: int = 30
    rel_types: Optional[List[str]] = None
    neighbor_labels: Optional[List[str]] = None


class KgPathsReq(BaseModel):
    a: str
    b: str
    k: int = 3
    max_hops: int = 3


@app.get("/health")
def health():
    return tool_health()


@app.post("/tools/kg_get_node")
def kg_get_node(req: KgGetNodeReq):
    return tool_kg_get_node(req.name)


@app.post("/tools/kg_neighbors")
def kg_neighbors(req: KgNeighborsReq):
    return tool_kg_neighbors(req.name, req.limit, req.rel_types, req.neighbor_labels)


@app.post("/tools/kg_paths")
def kg_paths(req: KgPathsReq):
    return tool_kg_paths(req.a, req.b, req.k, req.max_hops)


class VoiceDesignReq(BaseModel):
    voice_prompt: str
    preview_text: str
    preferred_name: str = "herb"
    target_model: str = DEFAULT_TARGET_MODEL
    language: str = "zh"
    sample_rate: int = 24000
    response_format: str = "wav"
    use_cache: bool = True

@app.post("/tools/voice_design_create")
def api_voice_design_create(req: VoiceDesignReq):
    return voice_design_create(
        voice_prompt=req.voice_prompt,
        preview_text=req.preview_text,
        preferred_name=req.preferred_name,
        target_model=req.target_model,
        language=req.language,
        sample_rate=req.sample_rate,
        response_format=req.response_format,
        use_cache=req.use_cache,
    )

class TTSReq(BaseModel):
    text: str
    voice: str
    model: str = DEFAULT_TARGET_MODEL
    sample_rate: int = 24000


@app.post("/tools/tts_vd_realtime_speak")
def api_tts_vd_realtime_speak(req: TTSReq):
    try:
        return tts_vd_realtime_speak(
            text=req.text,
            voice=req.voice,
            model=req.model,
            sample_rate=req.sample_rate,
            save_wav=True,
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}


class KgSubgraphReq(BaseModel):
    name: str
    depth: int = 1
    include_types: Optional[List[str]] = None
    max_nodes_per_type: int = 20


class KgRelationPathsReq(BaseModel):
    source: str
    target: str
    max_hops: int = 3


class KgGraphSummaryReq(BaseModel):
    name: str

@app.post("/tools/kg_subgraph")
def kg_subgraph(req: KgSubgraphReq):
    return tool_kg_subgraph(
        name=req.name,
        depth=req.depth,
        include_types=req.include_types,
        max_nodes_per_type=req.max_nodes_per_type,
    )


@app.post("/tools/kg_relation_paths")
def kg_relation_paths(req: KgRelationPathsReq):
    return tool_kg_relation_paths(
        source=req.source,
        target=req.target,
        max_hops=req.max_hops,
    )


@app.post("/tools/kg_graph_summary")
def kg_graph_summary(req: KgGraphSummaryReq):
    return tool_kg_graph_summary(name=req.name)

class WanImageReq(BaseModel):
    prompt: str
    herb_name: str = ""
    size: str = "1024*1024"
    style_hint: Optional[str] = None
    watermark: bool = False

@app.post("/tools/wan_text_to_image")
def wan_text_to_image(req: WanImageReq):
    try:
        return tool_wan_text_to_image(
            prompt=req.prompt,
            herb_name=req.herb_name,
            size=req.size,
            style_hint=req.style_hint,
            watermark=req.watermark,
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}



class FormulaFallbackReq(BaseModel):
    name: str


@app.post("/tools/formula_fallback")
def formula_fallback(req: FormulaFallbackReq):
    return tool_formula_fallback(req.name)


class FormulaStoryRefineReq(BaseModel):
    name: str
    composition_items: List[dict] = []
    efficacy_and_indications: str = ""
    applicable_syndromes: str = ""
    source: str = ""


@app.post("/tools/formula_story_refine")
def formula_story_refine(req: FormulaStoryRefineReq):
    return tool_formula_story_refine(
        name=req.name,
        composition_items=req.composition_items,
        efficacy_and_indications=req.efficacy_and_indications,
        applicable_syndromes=req.applicable_syndromes,
        source=req.source,
    )
