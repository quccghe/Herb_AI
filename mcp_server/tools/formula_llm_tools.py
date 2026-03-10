# -*- coding: utf-8 -*-
import json
import os
import re
from typing import Dict, Any, List, Optional

from tools.qwen_client import QwenClient
from tools.rag_hnsw import VectorStore, compose_evidence


def _safe_json_loads(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    t = re.sub(r"^```json\s*", "", t)
    t = re.sub(r"^```\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    m = re.search(r"\{.*\}", t, flags=re.S)
    if m:
        t = m.group(0)
    try:
        return json.loads(t)
    except Exception:
        return {}


def _default_card(name: str, evidence: str = "", reason: str = "") -> Dict[str, Any]:
    return {
        "ok": True,
        "type": "formula",
        "name": name,
        "summary": "",
        "source": "",
        "composition": [],
        "composition_items": [],
        "jun_chen_zuo_shi_analysis": "",
        "efficacy_and_indications": "",
        "applicable_syndromes": "",
        "modifications": "",
        "modern_research": "",
        "role_story": "",
        "formula_story": "",
        "evidence": evidence,
        "note": reason,
    }


def _template_story(name: str, items: List[Dict[str, Any]], efficacy: str, syndrome: str) -> Dict[str, str]:
    role_map = {"君": [], "臣": [], "佐": [], "使": []}
    for x in items:
        role = str(x.get("role") or "").strip()
        herb = str(x.get("name") or "").strip()
        if role in role_map and herb:
            role_map[role].append(herb)

    jun = "、".join(role_map["君"]) or "君药"
    chen = "、".join(role_map["臣"]) or "臣药"
    zuo = "、".join(role_map["佐"]) or "佐药"
    shi = "、".join(role_map["使"]) or "使药"

    return {
        "role_story": f"我是{name}，{jun}担任队长，{chen}做副将推进主攻，{zuo}在侧调和与补位，{shi}负责引经与统筹，整队配伍环环相扣。",
        "formula_story": f"{name}围绕“{efficacy or '调和阴阳、扶正祛邪'}”展开，通过多味药协同改善“{syndrome or '相关证候'}”，体现中医方剂整体观与辨证思路。",
    }


def tool_formula_story_llm(
    name: str,
    composition_items: Optional[List[Dict[str, Any]]] = None,
    efficacy_and_indications: str = "",
    applicable_syndromes: str = "",
    source: str = "",
) -> Dict[str, Any]:
    n = (name or "方剂").strip()
    items = composition_items or []
    efficacy = (efficacy_and_indications or "").strip()
    syndrome = (applicable_syndromes or "").strip()

    try:
        llm = QwenClient()
        item_txt = "；".join([f"{x.get('name','')}{x.get('dose','')}({x.get('role','')})" for x in items])
        prompt = f"""
请根据下列方剂信息，生成两个故事并输出严格JSON：
{{"role_story":"","formula_story":""}}

要求：
1) role_story: 100-180字，君臣佐使分工鲜明，语言生动。
2) formula_story: 100-180字，围绕功效、主治、配伍逻辑。
3) 不要空话，不要列表，不要Markdown。

方剂名：{n}
出处：{source}
组成与角色：{item_txt}
功效与主治：{efficacy}
适用证候：{syndrome}
""".strip()
        raw = llm.chat(
            [
                {"role": "system", "content": "你是严格JSON生成器，只输出JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.65,
            max_tokens=360,
            stream=False,
        )
        obj = _safe_json_loads(raw)
        rs = (obj.get("role_story") or "").strip()
        fs = (obj.get("formula_story") or "").strip()
        if rs and fs:
            return {"ok": True, "story_source": "llm", "role_story": rs, "formula_story": fs}
        raise RuntimeError("llm_empty_story")
    except Exception as e:
        tpl = _template_story(n, items, efficacy, syndrome)
        return {"ok": True, "story_source": "template", "error": str(e), **tpl}


def tool_formula_card_llm(
    name: str,
    vec_dir: Optional[str] = None,
    topk: int = 12,
    min_rag_score: float = 0.28,
) -> Dict[str, Any]:
    n = (name or "").strip()
    if not n:
        return {"ok": False, "error": "empty name"}

    vec = vec_dir or os.getenv("VEC_DIR", "data/rag_out_vec")

    try:
        vs = VectorStore(vec)
    except Exception as e:
        return {"ok": False, "error": f"vector_store_init_failed: {e}", "name": n}

    hits = vs.search(n, topk=max(6, int(topk)))
    best = hits[0][0] if hits else -1.0
    evidence = compose_evidence(hits, max_items=8, max_chars_each=420)

    if not hits or best < float(min_rag_score):
        return {"ok": False, "name": n, "reason": f"资料不足或相关性偏低(best_score={best:.3f})"}

    try:
        llm = QwenClient()
        prompt = f"""
你是中医方剂结构化助手。只能基于证据输出严格JSON：
{{
  "ok": true,
  "type": "formula",
  "name": "",
  "summary": "",
  "source": "",
  "composition": [],
  "composition_items": [{{"name":"","dose":"","role":"君|臣|佐|使|","weight":0}}],
  "jun_chen_zuo_shi_analysis": "",
  "efficacy_and_indications": "",
  "applicable_syndromes": "",
  "modifications": "",
  "modern_research": ""
}}

方剂名：{n}
证据：
{evidence}
""".strip()
        raw = llm.chat(
            [
                {"role": "system", "content": "你是严格JSON生成器，只输出JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=900,
            stream=False,
        )
        data = _safe_json_loads(raw)
        if not data.get("ok"):
            data = _default_card(n, evidence, "llm_json_parse_failed")
        data.setdefault("name", n)
        data.setdefault("type", "formula")
        data.setdefault("composition", [])
        data.setdefault("composition_items", [])
        data["evidence"] = evidence
        data["card_source"] = "llm_rag"

        story = tool_formula_story_llm(
            name=data.get("name", n),
            composition_items=data.get("composition_items", []),
            efficacy_and_indications=data.get("efficacy_and_indications", ""),
            applicable_syndromes=data.get("applicable_syndromes", ""),
            source=data.get("source", ""),
        )
        data["role_story"] = story.get("role_story", "")
        data["formula_story"] = story.get("formula_story", "")
        data["story_source"] = story.get("story_source", "template")
        return data
    except Exception as e:
        d = _default_card(n, evidence, f"llm_call_failed: {e}")
        d["ok"] = False
        return d
