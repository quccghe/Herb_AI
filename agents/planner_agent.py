# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from typing import Dict, Any, Optional


class PlannerAgent:
    """
    查询意图规划器：识别用户要查“中药”还是“方剂”。

    策略：
    1) 规则优先（后缀、关键词）
    2) 模糊句式清洗（如“介绍一下XX”）
    3) 可选 LLM 二次判定（有 llm 时）
    """

    FORMULA_SUFFIX = ("汤", "散", "丸", "饮", "膏", "丹", "剂", "颗粒", "胶囊", "片", "口服液")
    FORMULA_HINT_WORDS = ("方", "方剂", "处方", "中成药", "配伍", "君臣佐使", "组方")
    TRAILING_NOISE = ("是什么", "怎么样", "适合什么人", "适合谁", "有什么作用", "的功效", "功效与作用", "怎么用", "可以吗")

    def __init__(self, llm=None):
        self.llm = llm

    def _normalize_query(self, query: str) -> str:
        q = (query or "").strip()
        q = re.sub(r"^(请问|帮我|麻烦|想了解|介绍一下|请介绍一下|说说|科普一下)", "", q)
        for t in self.TRAILING_NOISE:
            if q.endswith(t):
                q = q[: -len(t)]
                break
        return q.strip(" ，。！？?、") or (query or "").strip()

    def _rule_plan(self, q: str) -> Dict[str, Any]:
        for s in self.FORMULA_SUFFIX:
            if q.endswith(s):
                return {"intent": "formula", "reason": f"suffix:{s}"}

        if any(w in q for w in self.FORMULA_HINT_WORDS):
            return {"intent": "formula", "reason": "hint_word"}

        # 句子中包含“XX汤/XX丸”等场景（例如：逍遥丸适合什么人）
        m = re.search(r"([\u4e00-\u9fffA-Za-z0-9]{1,12}(?:汤|散|丸|饮|膏|丹|剂|颗粒|胶囊|口服液))", q)
        if m:
            return {"intent": "formula", "reason": "embedded_formula_name"}

        return {"intent": "herb", "reason": "rule_default"}

    def _llm_plan(self, q: str) -> Optional[Dict[str, Any]]:
        if not self.llm:
            return None

        prompt = f"""
你是中医查询意图分类器。把输入分类为 herb（中药）或 formula（方剂/中成药）。
只输出严格 JSON：
{{"intent":"herb|formula","confidence":0~1,"normalized_query":""}}
输入：{q}
""".strip()
        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": "你是严格JSON分类器，只输出JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=120,
                stream=False,
            )
            m = re.search(r"\{.*\}", raw or "", flags=re.S)
            if not m:
                return None
            obj = json.loads(m.group(0))
            intent = (obj.get("intent") or "").strip()
            conf = float(obj.get("confidence") or 0.0)
            normalized_query = (obj.get("normalized_query") or q).strip() or q
            if intent in ("herb", "formula") and conf >= 0.55:
                return {
                    "intent": intent,
                    "reason": f"llm_conf:{conf:.2f}",
                    "normalized_query": normalized_query,
                    "confidence": conf,
                }
        except Exception:
            return None

        return None

    def plan(self, query: str) -> Dict[str, Any]:
        q = self._normalize_query(query)

        llm_result = self._llm_plan(q)
        if llm_result:
            return {
                "intent": llm_result["intent"],
                "query": q,
                "normalized_query": llm_result.get("normalized_query", q),
                "target_page": "formula_page.html" if llm_result["intent"] == "formula" else "herb_page.html",
                "source": "llm",
                "reason": llm_result.get("reason", "llm"),
                "confidence": llm_result.get("confidence", 0.0),
            }

        rule = self._rule_plan(q)
        intent = rule["intent"]
        return {
            "intent": intent,
            "query": q,
            "normalized_query": q,
            "target_page": "formula_page.html" if intent == "formula" else "herb_page.html",
            "source": "rule",
            "reason": rule.get("reason", "rule"),
        }
