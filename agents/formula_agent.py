# -*- coding: utf-8 -*-
import json
import re
from typing import Dict, Any, List, Optional

from tools.rag_hnsw import compose_evidence
from tools.qwen_client import QwenClient
from tools.formula_evidence_cleaner import FormulaEvidenceCleaner


class FormulaAgent:
    def __init__(self, vector_store, llm: Optional[QwenClient] = None, min_rag_score: float = 0.28):
        self.vs = vector_store
        self.llm = llm or QwenClient()
        self.min_rag_score = min_rag_score
        self.cleaner = FormulaEvidenceCleaner(enable_rerank=True)

    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            text = m.group(0)

        try:
            return json.loads(text)
        except Exception:
            return {"ok": False, "error": "llm_json_parse_failed", "raw": text}

    def _build_prompt(self, query: str, evidence: str) -> List[Dict[str, str]]:
        system_prompt = """
你是一名中医方剂结构化抽取助手。
必须只基于证据抽取“目标方剂本身”的信息，不能混入加减方、相关方或近似方。

只输出严格 JSON：
{
  "ok": true,
  "type": "formula",
  "name": "",
  "summary": "",
  "source": "",
  "composition": [""],
  "composition_items": [
    {"name":"", "dose":"", "role":"", "weight":0}
  ],
  "jun_chen_zuo_shi_analysis": "",
  "efficacy_and_indications": "",
  "applicable_syndromes": "",
  "modifications": "",
  "modern_research": "",
  "dosage_usage": "",
  "cautions": "",
  "role_story": "",
  "formula_story": ""
}
""".strip()

        user_prompt = f"""
目标方剂：{query}

证据：
{evidence}

请只抽取【{query}】本方信息，输出严格 JSON。
""".strip()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_repair_prompt(self, query: str, bad_output: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": "你是 JSON 修复器。请把下面内容修正为严格 JSON，只输出 JSON。"},
            {"role": "user", "content": f"目标方剂：{query}\n请修正为严格 JSON：\n{bad_output}"}
        ]

    def _build_story_prompt(self, card: Dict[str, Any]) -> List[Dict[str, str]]:
        payload = json.dumps({
            "name": card.get("name", ""),
            "source": card.get("source", ""),
            "composition_items": card.get("composition_items", []),
            "efficacy_and_indications": card.get("efficacy_and_indications", ""),
            "applicable_syndromes": card.get("applicable_syndromes", ""),
            "jun_chen_zuo_shi_analysis": card.get("jun_chen_zuo_shi_analysis", "")
        }, ensure_ascii=False)

        system_prompt = """
你是一名中医方剂故事化讲解助手。
请基于给定的结构化 JSON，生成严格 JSON：
{
  "role_story": "",
  "formula_story": ""
}
要求：
1. 只基于给定 JSON，不可编造
2. 每段 80~150 字
3. role_story 更强调“配伍协同”
4. formula_story 更强调“适用证候与治疗思路”
5. 只输出 JSON
""".strip()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload}
        ]

    def _has_useful_fields(self, data: Dict[str, Any]) -> bool:
        keys = [
            "summary", "source", "efficacy_and_indications",
            "applicable_syndromes", "dosage_usage", "cautions"
        ]
        if any(str(data.get(k) or "").strip() for k in keys):
            return True
        if data.get("composition_items") or data.get("composition"):
            return True
        return False

    def _parse_composition_items(self, comp_text: str) -> List[Dict[str, Any]]:
        if not comp_text:
            return []

        s = comp_text.replace("（", "(").replace("）", ")")
        pattern = r"([\u4e00-\u9fffA-Za-z·]+?)\s*([一二三四五六七八九十百千万半\d\.]+(?:枚|两|钱|克|g|G|ml|ML))(?:\(([^)]+)\))?"
        matches = re.findall(pattern, s)

        items = []
        for herb, raw_dose, norm_dose in matches:
            herb = herb.strip()
            dose = (norm_dose or raw_dose or "").strip()
            if herb:
                items.append({
                    "name": herb,
                    "dose": dose,
                    "role": "",
                    "weight": 0
                })

        if not items:
            parts = re.split(r"[，、；;\s]+", comp_text)
            for p in parts:
                p = p.strip()
                if p:
                    items.append({"name": p, "dose": "", "role": "", "weight": 0})

        return items

    def _fallback_from_evidence(self, query: str, evidence: str) -> Dict[str, Any]:
        sec = self.cleaner.extract_sections(evidence)

        comp_items = self._parse_composition_items(sec["composition"])
        composition = [f"{x['name']}{(' ' + x['dose']) if x.get('dose') else ''}" for x in comp_items]

        summary = ""
        if sec["efficacy"] and sec["indications"]:
            summary = f"{query}主要用于{sec['indications']}，功效为{sec['efficacy']}。"
        elif sec["efficacy"]:
            summary = f"{query}的核心功效为：{sec['efficacy']}。"
        elif sec["indications"]:
            summary = f"{query}主要适用于：{sec['indications']}。"

        return {
            "ok": True,
            "type": "formula",
            "name": query,
            "summary": summary,
            "source": sec["source"],
            "composition": composition,
            "composition_items": comp_items,
            "jun_chen_zuo_shi_analysis": sec["analysis"],
            "efficacy_and_indications": f"{sec['efficacy']} {sec['indications']}".strip(),
            "applicable_syndromes": sec["indications"],
            "modifications": "",
            "modern_research": "",
            "dosage_usage": sec["dosage_usage"],
            "cautions": sec["cautions"],
            "role_story": "",
            "formula_story": "",
            "story_source": "",
            "note": "已使用 evidence 规则抽取兜底。"
        }

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
            "story_source": "",
            "evidence": evidence,
            "note": "",
        }
        default.update(data or {})

        if not default["composition"]:
            default["composition"] = [
                f"{x['name']}{(' ' + x['dose']) if x.get('dose') else ''}"
                for x in default.get("composition_items", [])
            ]

        default["herb_links"] = [x["name"] for x in default.get("composition_items", []) if x.get("name")]
        return default

    def _generate_story_from_card(self, card: Dict[str, Any]) -> Dict[str, str]:
        raw = self.llm.chat(self._build_story_prompt(card), temperature=0.3)
        data = self._safe_json_loads(raw)
        if data.get("role_story") or data.get("formula_story"):
            return {
                "role_story": data.get("role_story", ""),
                "formula_story": data.get("formula_story", ""),
                "story_source": "llm"
            }

        return {
            "role_story": f"{card.get('name','该方')}通过多味药物协同配伍，共同围绕核心病机发挥作用。",
            "formula_story": f"{card.get('name','该方')}体现了中医方剂整体调节、标本兼顾的治疗思路。",
            "story_source": "rule"
        }

    def run(self, query: str) -> Dict[str, Any]:
        rag_hits = self.vs.search(query, topk=12)
        best = rag_hits[0][0] if rag_hits else -1.0
        evidence_raw = compose_evidence(rag_hits, max_items=8, max_chars_each=420)

        print(f"[FormulaAgent] query={query}, hit_count={len(rag_hits)}, best_score={best:.3f}")

        if not rag_hits or best < self.min_rag_score:
            return {
                "ok": False,
                "type": "formula",
                "name": query,
                "reason": f"资料不足或相关性偏低(best_score={best:.3f})"
            }

        # 1. 清洗 + rerank
        evidence = self.cleaner.clean(query, evidence_raw, topk=4)

        # 2. 主 LLM 抽取
        llm_raw = self.llm.chat(self._build_prompt(query, evidence), temperature=0.15)
        llm_data = self._safe_json_loads(llm_raw)

        # 3. JSON 修复
        if not llm_data.get("ok", False):
            repaired = self.llm.chat(self._build_repair_prompt(query, llm_raw), temperature=0.0)
            llm_data = self._safe_json_loads(repaired)

        # 4. fallback 抽取
        if not llm_data.get("ok", False) or not self._has_useful_fields(llm_data):
            card = self._fallback_from_evidence(query, evidence)
            card = self._with_default_fields(card, query, evidence)

            # 关键：fallback 后也用 LLM 生成故事
            story = self._generate_story_from_card(card)
            card["role_story"] = story["role_story"]
            card["formula_story"] = story["formula_story"]
            card["story_source"] = story["story_source"]
            return card

        llm_data["evidence"] = evidence
        llm_data["note"] = "已使用大模型结构化生成方剂卡片。"
        llm_data = self._with_default_fields(llm_data, query, evidence)

        # 主卡片成功后，再基于结构化 JSON 生成故事
        story = self._generate_story_from_card(llm_data)
        llm_data["role_story"] = story["role_story"]
        llm_data["formula_story"] = story["formula_story"]
        llm_data["story_source"] = story["story_source"]

        return llm_data