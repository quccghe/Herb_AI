# -*- coding: utf-8 -*-

import os
import base64
import shutil
import uuid
import re
from typing import Dict, Any, List

from tools.mcp_client import MCPClient


def _safe_name(s: str) -> str:
    """
    防止中药名包含非法字符
    """
    s = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "_", (s or "").strip())
    return (s[:32] or "herb")


class TTSAgent:
    """
    TTSAgent

    功能：
    1. 根据 persona.voice_recommendation 设计音色
    2. 合成自我介绍语音
    3. 合成三条口头禅语音
    4. 保存到 assets/audio/中药名/
    """

    def __init__(self, mcp: MCPClient,
                 target_model: str = "qwen3-tts-vd-realtime-2026-01-15"):

        self.mcp = mcp
        self.target_model = target_model

    # ---------------------------------------------------
    # 提取音频路径
    # ---------------------------------------------------

    def _extract_audio_path(self, tts_resp: Dict[str, Any]) -> str:

        if not isinstance(tts_resp, dict):
            return ""

        if tts_resp.get("audio_path"):
            return tts_resp["audio_path"]

        tts = tts_resp.get("tts")

        if isinstance(tts, dict):
            return tts.get("audio_path", "")

        return ""

    # ---------------------------------------------------
    # base64音频保存
    # ---------------------------------------------------

    def _save_base64_audio(self, audio_base64: str, save_path: str):

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        audio_bytes = base64.b64decode(audio_base64)

        with open(save_path, "wb") as f:
            f.write(audio_bytes)

        return save_path

    # ---------------------------------------------------
    # TTS生成
    # ---------------------------------------------------

    def run(self, herb_name: str,
            narration: Dict[str, Any],
            persona: Dict[str, Any]) -> Dict[str, Any]:

        text = (narration.get("text") or "").strip()

        if not text:
            return {
                "type": "tts",
                "ok": False,
                "reason": "empty_text"
            }

        persona_obj = persona.get("persona") or {}

        voice_cfg = persona_obj.get("voice_recommendation") or {}

        voice_prompt = voice_cfg.get("voice") or ""

        if not voice_prompt:
            voice_prompt = "中文成年中性声音，语速适中，吐字清晰。"

        herb_name = _safe_name(herb_name)

        audio_dir = f"assets/audio/{herb_name}"

        os.makedirs(audio_dir, exist_ok=True)

        # ---------------------------------------------------
        # 1 设计音色
        # ---------------------------------------------------

        vd = self.mcp.voice_design_create(
            voice_prompt=voice_prompt,
            preview_text="你好，我来介绍这味中药。",
            preferred_name=herb_name,
            target_model=self.target_model,
            use_cache=True
        )

        voice = vd.get("voice")

        if not voice:
            return {
                "type": "tts",
                "ok": False,
                "reason": "voice_design_failed",
                "vd": vd
            }

        # ---------------------------------------------------
        # 2 生成自我介绍语音
        # ---------------------------------------------------

        intro_path = f"{audio_dir}/intro.wav"

        if not os.path.exists(intro_path):

            intro_tts = self.mcp.tts_vd_realtime_speak(
                text=text,
                voice=voice,
                model=self.target_model
            )

            audio_tmp = self._extract_audio_path(intro_tts)

            if audio_tmp and os.path.exists(audio_tmp):

                shutil.copy(audio_tmp, intro_path)

        # ---------------------------------------------------
        # 3 生成口头禅语音
        # ---------------------------------------------------

        catchphrases: List[str] = persona_obj.get("catchphrases") or []

        catchphrases = [
            str(x).strip()
            for x in catchphrases
            if str(x).strip()
        ][:3]

        catchphrase_audio = []
        catchphrase_items = []

        for idx, phrase in enumerate(catchphrases, start=1):

            final_path = f"{audio_dir}/catch_{idx}.wav"

            if not os.path.exists(final_path):

                phrase_tts = self.mcp.tts_vd_realtime_speak(
                    text=phrase,
                    voice=voice,
                    model=self.target_model
                )

                tmp_audio = self._extract_audio_path(phrase_tts)

                if tmp_audio and os.path.exists(tmp_audio):

                    shutil.copy(tmp_audio, final_path)

            catchphrase_audio.append(final_path)

            catchphrase_items.append({
                "index": idx,
                "text": phrase,
                "audio_path": final_path
            })

        # ---------------------------------------------------
        # 返回结果
        # ---------------------------------------------------

        return {

            "type": "tts",

            "ok": True,

            "voice": voice,

            "vd": vd,

            # 自我介绍
            "intro_audio": intro_path,

            # 兼容旧前端
            "audio_path": intro_path,

            # 口头禅
            "catchphrase_audio": catchphrase_audio,

            "catchphrase_items": catchphrase_items
        }