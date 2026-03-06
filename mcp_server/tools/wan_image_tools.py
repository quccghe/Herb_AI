# -*- coding: utf-8 -*-
import os
import time
import uuid
from typing import Dict, Any, Optional

import httpx


IMAGE_DIR = os.getenv("IMAGE_DIR", "assets/images")
WAN_MODEL = os.getenv("WAN_IMAGE_MODEL", "wan2.6-t2i")
DASHSCOPE_REGION = os.getenv("DASHSCOPE_REGION", "cn").lower().strip()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()


def _ensure_dir():
    os.makedirs(IMAGE_DIR, exist_ok=True)


def _get_base_url() -> str:
    if DASHSCOPE_REGION == "intl":
        return "https://dashscope-intl.aliyuncs.com/api/v1"
    if DASHSCOPE_REGION == "us":
        return "https://dashscope-us.aliyuncs.com/api/v1"
    return "https://dashscope.aliyuncs.com/api/v1"


def _safe_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "image"
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"):
            keep.append(ch)
        elif "\u4e00" <= ch <= "\u9fff":
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)[:40] or "image"


def tool_wan_text_to_image(
    prompt: str,
    herb_name: str = "",
    size: str = "1024*1024",
    style_hint: Optional[str] = None,
    watermark: bool = False,
) -> Dict[str, Any]:
    """
    使用 wan2.6-t2i 生成拟人化图像。
    返回本地保存路径 + 原始结果 URL。
    """
    if not DASHSCOPE_API_KEY:
        return {"ok": False, "error": "Missing DASHSCOPE_API_KEY"}

    prompt = (prompt or "").strip()
    if not prompt:
        return {"ok": False, "error": "empty prompt"}

    _ensure_dir()
    url = f"{_get_base_url()}/services/aigc/multimodal-generation/generation"

    full_prompt = prompt
    if style_hint:
        full_prompt = f"{prompt}。风格要求：{style_hint}"

    payload = {
        "model": WAN_MODEL,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": full_prompt}
                    ]
                }
            ]
        },
        "parameters": {
            "size": size,
            "watermark": watermark
        }
    }

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        with httpx.Client(timeout=180) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # 官方结果里 output.results[0].url 可下载，且 url 有效期 24h
        output = data.get("output", {}) or {}

        # 兼容两种返回结构：
        # 1) output.results[].url
        # 2) output.choices[].message.content[].image
        img_urls = []

        results = output.get("results", []) or []
        for item in results:
            url = item.get("url")
            if url:
                img_urls.append(url)

        choices = output.get("choices", []) or []
        for choice in choices:
            message = choice.get("message", {}) or {}
            content = message.get("content", []) or []
            for c in content:
                if c.get("type") == "image" and c.get("image"):
                    img_urls.append(c["image"])

        if not img_urls:
            return {"ok": False, "error": "No image url returned", "raw": data}

        # 默认取第一张图下载到本地；同时把全部URL返回
        img_url = img_urls[0]
        # 下载到本地
        file_stub = _safe_name(herb_name or "wan_image")
        filename = f"{file_stub}_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
        local_path = os.path.join(IMAGE_DIR, filename)

        with httpx.Client(timeout=180) as client:
            img_resp = client.get(img_url)
            img_resp.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(img_resp.content)

        return {
            "ok": True,
            "model": WAN_MODEL,
            "prompt": full_prompt,
            "size": size,
            "image_url": img_url,
            "image_urls": img_urls,
            "image_path": local_path.replace("\\", "/"),
            "usage": data.get("usage", {}),
            "request_id": data.get("request_id", "")
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}