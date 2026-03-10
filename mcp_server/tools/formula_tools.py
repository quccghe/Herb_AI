# -*- coding: utf-8 -*-
import json
import re
from typing import Dict, Any, List

from tools.qwen_client import QwenClient


FORMULA_PRESETS = {
    "四君子汤": {
        "source": "《太平惠民和剂局方》",
        "composition_items": [
            {"name": "人参", "dose": "9g", "role": "君", "weight": 4},
            {"name": "白术", "dose": "9g", "role": "臣", "weight": 3},
            {"name": "茯苓", "dose": "9g", "role": "佐", "weight": 2},
            {"name": "炙甘草", "dose": "6g", "role": "使", "weight": 1},
        ],
        "efficacy_and_indications": "益气健脾。常用于脾胃气虚所致乏力、食少、便溏。",
        "applicable_syndromes": "脾胃气虚证。",
    },
    "柏子养心丸": {
        "source": "《体仁汇编》",
        "composition_items": [
            {"name": "柏子仁", "dose": "12g", "role": "君", "weight": 4},
            {"name": "党参", "dose": "9g", "role": "臣", "weight": 3},
            {"name": "黄芪", "dose": "9g", "role": "臣", "weight": 3},
            {"name": "炙甘草", "dose": "6g", "role": "使", "weight": 1},
        ],
        "efficacy_and_indications": "补气养血，宁心安神。用于心气不足、心血亏虚所致失眠健忘。",
        "applicable_syndromes": "心血不足、心神失养证。",
    },
}


def _default_items() -> List[Dict[str, Any]]:
    return [
        {"name": "君药", "dose": "", "role": "君", "weight": 4},
        {"name": "臣药", "dose": "", "role": "臣", "weight": 3},
        {"name": "佐药", "dose": "", "role": "佐", "weight": 2},
        {"name": "使药", "dose": "", "role": "使", "weight": 1},
    ]


def _safe_json_loads(s: str) -> Dict[str, Any]:
    text = (s or "").strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        text = m.group(0)
    try:
        return json.loads(text)
    except Exception:
        return {}


def _template_story(name: str, items: List[Dict[str, Any]], efficacy: str, syndrome: str) -> Dict[str, str]:
    role_map = {"君": [], "臣": [], "佐": [], "使": []}
    for x in items:
        role = str(x.get("role") or "").strip()
        n = str(x.get("name") or "").strip()
        if role in role_map and n:
            role_map[role].append(n)

    jun = "、".join(role_map["君"]) or "君药"
    chen = "、".join(role_map["臣"]) or "臣药"
    zuo = "、".join(role_map["佐"]) or "佐药"
    shi = "、".join(role_map["使"]) or "使药"

    role_story = f"我是{name}，由{jun}担任队长直指核心病机，{chen}协同扩大战果，{zuo}在侧调和偏性，{shi}负责统筹引导，让整方配伍紧密有序。"
    formula_story = f"{name}围绕“{efficacy or '调和脏腑、扶正祛邪'}”展开，多味药分工协作，共同改善“{syndrome or '相关证候'}”，体现中医方剂整体调节、标本兼顾的思路。"
    return {"role_story": role_story, "formula_story": formula_story}


def tool_formula_story_refine(
    name: str,
    composition_items: List[Dict[str, Any]],
    efficacy_and_indications: str,
    applicable_syndromes: str,
    source: str = "",
) -> Dict[str, Any]:
    n = (name or "方剂").strip()
    items = composition_items or _default_items()
    efficacy = (efficacy_and_indications or "").strip()
    syndrome = (applicable_syndromes or "").strip()

    # 先尝试大模型润色
    try:
        llm = QwenClient()
        item_txt = "；".join([f"{x.get('name','')}{x.get('dose','')}({x.get('role','')})" for x in items])
        prompt = f"""
你是中医科普写作者。请根据方剂信息生成两个故事，输出严格JSON：
{{"role_story":"","formula_story":""}}
要求：
1) role_story: 90-150字，突出君臣佐使分工，角色感强。
2) formula_story: 90-150字，突出功效主治与配伍逻辑。
3) 语言自然，避免空话和模板化。

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
            temperature=0.6,
            max_tokens=320,
            stream=False,
        )
        obj = _safe_json_loads(raw)
        rs = (obj.get("role_story") or "").strip()
        fs = (obj.get("formula_story") or "").strip()
        if rs and fs:
            return {"ok": True, "story_source": "llm", "role_story": rs, "formula_story": fs}
    except Exception as e:
        err = str(e)
    else:
        err = "llm_output_empty"

    # LLM 失败则模板兜底
    tpl = _template_story(n, items, efficacy, syndrome)
    return {"ok": True, "story_source": "template", "error": err, **tpl}


def tool_formula_fallback(name: str) -> Dict[str, Any]:
    n = (name or "").strip()
    preset = FORMULA_PRESETS.get(n, {})
    items = preset.get("composition_items") or _default_items()

    stories = tool_formula_story_refine(
        name=n or "方剂",
        composition_items=items,
        efficacy_and_indications=preset.get("efficacy_and_indications", ""),
        applicable_syndromes=preset.get("applicable_syndromes", ""),
        source=preset.get("source", ""),
    )

    return {
        "ok": True,
        "type": "formula",
        "name": n or "方剂",
        "summary": preset.get("efficacy_and_indications") or "基于降级服务返回的方剂信息。",
        "source": preset.get("source") or "资料来源待补充",
        "composition": [f"{x['name']}{(' ' + x['dose']) if x.get('dose') else ''}" for x in items],
        "composition_items": items,
        "jun_chen_zuo_shi_analysis": "君药主攻核心病机，臣药协同增效，佐药调和偏性，使药引经与统筹。",
        "efficacy_and_indications": preset.get("efficacy_and_indications") or "功效与主治待进一步补充。",
        "applicable_syndromes": preset.get("applicable_syndromes") or "适用证候待进一步补充。",
        "modifications": "请根据证候寒热虚实做临床加减。",
        "modern_research": "现代研究提示该类方剂在改善相关症状方面具有潜在价值。",
        "role_story": stories.get("role_story", ""),
        "formula_story": stories.get("formula_story", ""),
        "story_source": stories.get("story_source", "template"),
        "fallback": True,
    }
