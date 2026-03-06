# -*- coding: utf-8 -*-
from typing import Dict, Any, Optional, List


def risk_radar(precaution: str) -> str:
    """
    根据注意事项粗略生成风险等级
    """
    p = (precaution or "").strip()
    score = 0

    if "孕" in p or "妊娠" in p:
        score += 2
    if "哺乳" in p:
        score += 1
    if "慎" in p:
        score += 1
    if "毒" in p:
        score += 2
    if "禁" in p:
        score += 2
    if "忌" in p:
        score += 1

    if score >= 3:
        return "🔴 高风险"
    if score >= 2:
        return "🟡 中风险"
    return "🟢 低风险"


def _ensure_list(val) -> List[str]:
    """
    把可能的字符串/列表统一成列表
    """
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        # 常见分隔符切分
        for sep in ["、", "，", ",", "；", ";", " "]:
            if sep in s:
                parts = [x.strip() for x in s.split(sep) if x.strip()]
                if parts:
                    return parts
        return [s]
    return [str(val).strip()]


class HerbCardAgent:
    """
    生成结构化中药卡片 JSON

    优先使用：
    - kg["efficacy"]
    - kg["diseases"]
    - kg["taste_nodes"]
    - kg["meridian_nodes"]
    - kg["categories"]

    如果没有这些字段，就从 kg["node"] 中兜底：
    - node["taste"]
    - node["meridian"]
    - node["dosage"]
    - node["precaution"]
    - node["category"]
    """

    def run(self, query: str, kg: Optional[Dict[str, Any]], evidence: str) -> Dict[str, Any]:
        # ===== 1. KG命中时，优先用图谱 =====
        if kg and kg.get("found"):
            node = kg.get("node") or {}
            name = kg.get("name") or node.get("name") or query

            # 先取结构化字段
            efficacy = _ensure_list(kg.get("efficacy"))
            diseases = _ensure_list(kg.get("diseases"))
            taste_nodes = _ensure_list(kg.get("taste_nodes"))
            meridian_nodes = _ensure_list(kg.get("meridian_nodes"))
            categories = _ensure_list(kg.get("categories"))

            # ===== 2. 从 node 里兜底 =====
            # 性味
            taste = (node.get("taste") or "").strip()
            if not taste and taste_nodes:
                taste = "、".join(taste_nodes)

            # 归经
            meridian = (node.get("meridian") or "").strip()
            if not meridian and meridian_nodes:
                meridian = "、".join(meridian_nodes)

            # 功效
            if not efficacy:
                # 某些图谱可能把功效存在 node 字段里
                efficacy = _ensure_list(node.get("efficacy"))

            # 主治/相关病症
            if not diseases:
                diseases = _ensure_list(node.get("diseases"))

            # 用量
            dosage = (node.get("dosage") or "").strip()

            # 注意事项
            precaution = (node.get("precaution") or "").strip()

            # 类别
            if not categories:
                categories = _ensure_list(node.get("category"))

            risk = risk_radar(precaution)

            return {
                "type": "herb_card",
                "name": name,
                "taste": taste,
                "meridian": meridian,
                "efficacy": efficacy,
                "diseases": diseases,
                "dosage": dosage,
                "precaution": precaution,
                "categories": categories,
                "risk_level": risk,
                "evidence": evidence,
            }

        # ===== 3. KG未命中：返回稳定骨架 =====
        return {
            "type": "herb_card",
            "name": query,
            "taste": "",
            "meridian": "",
            "efficacy": [],
            "diseases": [],
            "dosage": "",
            "precaution": "",
            "categories": [],
            "risk_level": "",
            "evidence": evidence,
            "note": "KG未命中：字段可能不全，本卡片可用于前端稳定渲染。",
        }