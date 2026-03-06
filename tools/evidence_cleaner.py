# -*- coding: utf-8 -*-
"""
证据去污染：从拼接证据中提取更“像目标药条目”的片段，减少混入相邻药条目导致的错误迁移。
"""
from typing import Optional
import re


def clean_evidence_for_entity(evidence: str, entity: str, max_chars: int = 1400) -> str:
    """
    返回更干净的 evidence（仍保留引用结构，但内容更聚焦）
    """
    if not evidence or not entity:
        return evidence

    ev = evidence

    # 常见：药典条目会出现 “甘草 Gancao” 或 “GLYCYRRHIZAE RADIX ET RHIZOMA”
    # 1) 尝试在 evidence 中找 “{实体} ” 的标题行位置
    #    例如： "甘草 Gancao GLYCYRRHIZAE RADIX ET RHIZOMA"
    pattern_title = re.compile(rf"({re.escape(entity)}\s+Gancao|{re.escape(entity)}\s+.*RADIX|{re.escape(entity)}\s+.*RHIZOMA)", re.I)
    m = pattern_title.search(ev)
    if m:
        start = m.start()
        clipped = ev[start:start + max_chars]
        return clipped.strip()

    # 2) 退而求其次：找“性味与归经】 甘，平”附近（或“归…经”）
    pattern_taste = re.compile(r"(性味与归经】\s*.*?归.*?经)", re.I)
    m2 = pattern_taste.search(ev)
    if m2:
        start = max(0, m2.start() - 400)
        clipped = ev[start:start + max_chars]
        return clipped.strip()

    # 3) 再退一步：只保留包含实体名的证据块（按 [证据i] 分块过滤）
    blocks = re.split(r"\n\s*\n", ev)
    kept = []
    for b in blocks:
        if entity in b:
            kept.append(b)
    if kept:
        out = "\n\n".join(kept)
        return out[:max_chars].strip()

    return ev[:max_chars].strip()