# -*- coding: utf-8 -*-
import json
import re
from typing import Dict, Any, List, Optional

from tools.rag_hnsw import compose_evidence
from tools.qwen_client import QwenClient


class FormulaAgent:
    """
    方剂讲解 Agent
    做法：
    1. 用 RAG 检索证据
    2. 把证据喂给大模型
    3. 输出结构化讲解结果
    """

    def __init__(self, vector_store, llm: Optional[QwenClient] = None, min_rag_score: float = 0.28):
        self.vs = vector_store
        self.llm = llm or QwenClient()
        self.min_rag_score = min_rag_score

    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()

        # 去掉 markdown code fence
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)
        except Exception:
            return {
                "ok": False,
                "error": "llm_json_parse_failed",
                "raw": text
            }

    def _build_prompt(self, query: str, evidence: str) -> List[Dict[str, str]]:
        system_prompt = """
你是一名中医方剂讲解助手。
你的任务是：基于用户提供的检索证据，对方剂进行结构化总结。
要求：
1. 只能基于证据总结，不要凭空编造。
2. 若某字段证据不足，请写空字符串或空列表，不要乱填。
3. 输出必须是严格 JSON，不要输出解释性文字。
4. 尽量提炼方剂的组成、功效、主治、出处、证候要点、方义简析、用法用量、注意事项。
5. composition 输出列表；其他字段输出字符串。
6. summary 用一句话概括方剂核心作用。
JSON格式如下：
{
  "ok": true,
  "type": "formula",
  "name": "",
  "summary": "",
  "source": "",
  "composition": [],
  "efficacy": "",
  "indications": "",
  "syndrome_points": "",
  "compatibility_analysis": "",
  "dosage_usage": "",
  "modern_hint": "",
  "cautions": "",
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

    def run(self, query: str) -> Dict[str, Any]:
        rag_hits = self.vs.search(query, topk=6)
        best = rag_hits[0][0] if rag_hits else -1.0
        evidence = compose_evidence(rag_hits)

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
            # 兜底：即使 JSON 失败，也给前端可展示内容
            return {
                "ok": True,
                "type": "formula",
                "name": query,
                "summary": "",
                "source": "",
                "composition": [],
                "efficacy": "",
                "indications": "",
                "syndrome_points": "",
                "compatibility_analysis": "",
                "dosage_usage": "",
                "modern_hint": "",
                "cautions": "",
                "evidence": evidence,
                "note": "LLM 结构化失败，已返回原始证据。"
            }

        data["evidence"] = evidence
        return data