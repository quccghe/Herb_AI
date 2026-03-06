import re
from typing import Optional, Dict, Set

PRONOUNS: Set[str] = {
    "它", "这个", "这个药", "这味药", "这种药",
    "这个方子", "这方", "这个成药", "这个丸", "这个汤", "这个散"
}

ALIAS_MAP: Dict[str, str] = {
    "川穹": "川芎",
}

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()

def resolve_query(user_q: str, topic: Optional[str]) -> str:
    q = normalize(user_q)
    if topic and q in PRONOUNS:
        return topic
    return ALIAS_MAP.get(q, q)