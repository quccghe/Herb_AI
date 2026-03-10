# -*- coding: utf-8 -*-
from typing import Dict, Any, List


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


def tool_formula_fallback(name: str) -> Dict[str, Any]:
    n = (name or "").strip()
    preset = FORMULA_PRESETS.get(n, {})
    items = preset.get("composition_items") or _default_items()

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
        "role_story": f"我是{n or '这个方剂'}，君药领队，臣药助阵，佐使协同，让整体配伍更稳。",
        "formula_story": f"{n or '该方'}遵循整体调和思路，通过多味药协同实现标本兼顾。",
        "fallback": True,
    }
