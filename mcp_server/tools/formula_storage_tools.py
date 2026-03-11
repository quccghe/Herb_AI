# -*- coding: utf-8 -*-
import json
import os
import re
from datetime import datetime
from typing import Any, Dict

FORMULA_CARD_ROOT = os.path.join("assets", "formula_cards")


def _safe_formula_name(name: str) -> str:
    raw = (name or "").strip() or "未命名方剂"
    safe = re.sub(r"[\\/:*?\"<>|]", "_", raw)
    safe = safe.replace("..", "_").strip()
    return safe[:120] or "未命名方剂"


def tool_formula_write_json(name: str, card_data: Dict[str, Any]) -> Dict[str, Any]:
    safe_name = _safe_formula_name(name)
    folder = os.path.join(FORMULA_CARD_ROOT, safe_name)
    os.makedirs(folder, exist_ok=True)

    payload = dict(card_data or {})
    payload.setdefault("name", name)
    payload["saved_at"] = datetime.now().isoformat(timespec="seconds")

    abs_path = os.path.join(folder, "card.json")
    with open(abs_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    web_path = f"/assets/formula_cards/{safe_name}/card.json"
    return {
        "ok": True,
        "type": "formula_json",
        "name": safe_name,
        "saved_path": abs_path,
        "web_path": web_path,
    }
