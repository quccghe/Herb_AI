# -*- coding: utf-8 -*-
import json
import re
from typing import Dict, Any, List, Optional

from tools.rag_hnsw import compose_evidence
from tools.qwen_client import QwenClient


class FormulaAgent:
    """
    方剂讲解 Agent（仅使用 RAG，不依赖知识图谱）。

    流程：
    1) RAG 检索（提高 topk）
    2) LLM 结构化抽取
    3) 输出前端可视化需要的数据（配伍图、剂量比例、故事化讲解）
    """

    def __init__(self, vector_store, llm: Optional[QwenClient] = None, min_rag_score: float = 0.28):
        self.vs = vector_store
        self.llm = llm or QwenClient()
        self.min_rag_score = min_rag_score

    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except Exception:
            return {"ok": False, "error": "llm_json_parse_failed", "raw": text}

    def _build_prompt(self, query: str, evidence: str) -> List[Dict[str, str]]:
        system_prompt = """
你是一名中医方剂讲解助手。
你的任务：只能基于检索证据，输出“可直接用于前端展示与可视化”的严格 JSON。

硬性要求：
1. 只能依据证据，不可编造；证据不足字段用空字符串/空数组。
2. 输出必须是严格 JSON，不要 markdown，不要解释。
3. composition_items 要尽量结构化到“药名/剂量/君臣佐使/权重”。
4. role_story / formula_story 要故事化、口语化，100~180字，语气生动但严谨。
5. 如果证据里没有明确加减法，不要硬写。

JSON 格式：
{
  "ok": true,
  "type": "formula",
  "name": "",
  "summary": "",
  "source": "",
  "composition": [""],
  "composition_items": [
    {"name":"", "dose":"", "role":"君|臣|佐|使|", "weight":0}
  ],
  "jun_chen_zuo_shi_analysis": "",
  "efficacy_and_indications": "",
  "applicable_syndromes": "",
  "modifications": "",
  "modern_research": "",
  "dosage_usage": "",
  "cautions": "",
  "role_story": "",
  "formula_story": "",
  "raw_evidence_used": ""
}
""".strip()

        user_prompt = f"""
方剂名：{query}

检索证据如下：
{evidence}

请输出严格 JSON。
""".strip()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _normalize_composition_items(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        items = data.get("composition_items")
        if isinstance(items, list) and items:
            out = []
            for x in items:
                if not isinstance(x, dict):
                    continue
                name = str(x.get("name") or "").strip()
                if not name:
                    continue
                dose = str(x.get("dose") or "").strip()
                role = str(x.get("role") or "").strip()
                try:
                    weight = float(x.get("weight") or 0)
                except Exception:
                    weight = 0
                out.append({"name": name, "dose": dose, "role": role, "weight": weight})
            if out:
                return out

        # 兜底：由 composition 文本列表生成基础结构
        comp = data.get("composition") or []
        if isinstance(comp, list):
            out = []
            for c in comp:
                s = str(c or "").strip()
                if not s:
                    continue
                m = re.match(r"^([\u4e00-\u9fffA-Za-z0-9·]+)\s*([0-9\.两钱克gGmlML]*)?$", s)
                if m:
                    out.append({"name": m.group(1), "dose": (m.group(2) or "").strip(), "role": "", "weight": 0})
                else:
                    out.append({"name": s, "dose": "", "role": "", "weight": 0})
            return out
        return []

    def _with_default_fields(self, data: Dict[str, Any], query: str, evidence: str) -> Dict[str, Any]:
        default = {
            "ok": True,
            "type": "formula",
            "name": query,
            "summary": "",
            "source": "",
            "composition": [],
            "composition_items": [],
            "jun_chen_zuo_shi_analysis": "",
            "efficacy_and_indications": "",
            "applicable_syndromes": "",
            "modifications": "",
            "modern_research": "",
            "dosage_usage": "",
            "cautions": "",
            "role_story": "",
            "formula_story": "",
            "evidence": evidence,
        }
        default.update(data or {})
        default["composition_items"] = self._normalize_composition_items(default)
        default["herb_links"] = [x["name"] for x in default["composition_items"] if x.get("name")]
        return default

    def run(self, query: str) -> Dict[str, Any]:
        # topk 提高：增强方剂证据覆盖
        rag_hits = self.vs.search(query, topk=12)
        best = rag_hits[0][0] if rag_hits else -1.0
        evidence = compose_evidence(rag_hits, max_items=8, max_chars_each=420)

        if not rag_hits or best < self.min_rag_score:
            return {
                "ok": False,
                "type": "formula",
                "name": query,
                "reason": f"资料不足或相关性偏低(best_score={best:.3f})"
            }

        messages = self._build_prompt(query, evidence)
        llm_text = self.llm.chat(messages, temperature=0.2)
        data = self._safe_json_loads(llm_text)

        if not data.get("ok", False):
            return self._with_default_fields({"note": "LLM 结构化失败，已返回原始证据。"}, query, evidence)

        data["evidence"] = evidence
        return self._with_default_fields(data, query, evidence)
