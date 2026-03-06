# -*- coding: utf-8 -*-
from typing import Dict, Any

from tools.mcp_client import MCPClient


class ImageAgent:
    """
    根据 persona 输出的 image_prompt 调用 wan2.6-t2i 生成拟人化图像
    """

    def __init__(self, mcp: MCPClient):
        self.mcp = mcp

    def run(self, herb_name: str, persona: Dict[str, Any]) -> Dict[str, Any]:
        persona_obj = persona.get("persona") or {}
        image_prompt = (persona_obj.get("image_prompt") or "").strip()

        if not image_prompt:
            return {
                "type": "image",
                "ok": False,
                "error": "missing image_prompt"
            }

        # 统一风格模板，保证角色图更稳定
        style_hint = (
            "中国风中药拟人化角色设定图，角色立绘，半身或三分之二身，"
            "构图稳定，背景简洁，细节清晰，非海报排版，无文字，无水印，"
            "色彩柔和，适合科普系统展示"
        )

        resp = self.mcp.wan_text_to_image(
            prompt=image_prompt,
            herb_name=herb_name,
            size="1024*1024",
            style_hint=style_hint,
            watermark=False,
        )

        return {
            "type": "image",
            **resp
        }