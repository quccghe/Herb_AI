# -*- coding: utf-8 -*-
"""
LLM 实体抽取：从用户问题中抽取 1 个“主要实体”
支持：中药名、方剂/中成药名、炮制品名
"""
from typing import Optional, Dict, Any, List
import json
import re

from tools.qwen_client import QwenClient


def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    # 尝试截取第一个 {...}
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        return None


class LLMEntityExtractor:
    def __init__(self, llm: QwenClient):
        self.llm = llm

    def extract(self, user_input: str, rag_hint: Optional[str] = None) -> Optional[str]:
        """
        返回：实体字符串 或 None
        rag_hint：可选，把 RAG top 证据（或标题行）喂给 LLM 提升抽取成功率
        """
        user_input = (user_input or "").strip()
        hint = (rag_hint or "").strip()

        prompt = f"""
你是“中医药实体抽取器”。请从【用户输入】中抽取“一个最核心的实体”，用于查询知识图谱与检索。
实体类型仅限：
- 中药材名（如 甘草、川芎）
- 炮制品名（如 炙甘草）
- 方剂/中成药名（如 柏子养心丸）

输出必须是严格 JSON（不要 markdown），格式：
{{
  "entity": "实体名或空字符串",
  "entity_type": "herb|processed|formula|unknown",
  "confidence": 0.0-1.0
}}

抽取规则：
- 如果句子里包含“介绍/功效/禁忌/用量/现代研究”等，请忽略这些词，只抽实体名。
- 如果出现多个实体，只选“最主要那个”（通常是用户问的对象）。
- 如果无法确定，entity 置空，entity_type=unknown，confidence=0.0
- 不要输出多余文字。

【用户输入】
{user_input}

【可选提示：检索片段（可能含药典标题/拉丁名）】
{hint}
""".strip()

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "你是严格的JSON抽取器，只输出JSON。"},
            {"role": "user", "content": prompt},
        ]
        raw = self.llm.chat(messages, temperature=0.0, max_tokens=200, stream=False)
        obj = _safe_json_loads(raw)
        if not obj:
            return None

        entity = (obj.get("entity") or "").strip()
        conf = float(obj.get("confidence") or 0.0)

        if entity and conf >= 0.35:
            # 轻度清洗：去掉引号/空格
            entity = entity.strip().strip('"').strip()
            return entity
        return None