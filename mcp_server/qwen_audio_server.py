# -*- coding: utf-8 -*-
"""
MCP Server: Qwen Voice Design + Qwen3 TTS VD Realtime

提供两个工具：
1) qwen_voice_design_create  -> 通过文本描述创建音色（返回 voice + 预览音频base64）
2) qwen_tts_vd_speak         -> 使用指定 voice 合成一段语音（返回 wav base64）

文档要点：
- 声音设计模型固定为 qwen-voice-design，并用 /api/v1/services/audio/tts/customization
- target_model 要与后续语音合成使用的 model 一致，否则可能合成失败
- voice 可直接用于语音合成的 voice 参数
"""
from __future__ import annotations

import base64
import os
import re
import time
import wave
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# dashscope SDK（用于 realtime 合成）
import dashscope
from dashscope.audio.qwen_tts_realtime import QwenTtsRealtime, QwenTtsRealtimeCallback, AudioFormat

load_dotenv()

mcp = FastMCP("qwen-audio")

DEFAULT_TARGET_MODEL = "qwen3-tts-vd-realtime-2026-01-15"
DEFAULT_SAMPLE_RATE = 24000

def _get_region_host() -> str:
    region = (os.getenv("DASHSCOPE_REGION") or "cn").lower().strip()
    if region == "intl":
        return "dashscope-intl.aliyuncs.com"
    return "dashscope.aliyuncs.com"

def _get_api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少环境变量 DASHSCOPE_API_KEY")
    return api_key

def _safe_preferred_name(s: str) -> str:
    """
    文档要求 preferred_name：仅允许数字、英文字母和下划线，不超过16字符。:contentReference[oaicite:4]{index=4}
    """
    s = s.strip()
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
    s = s[:16] if s else "herb"
    if not re.match(r"^[0-9A-Za-z_]{1,16}$", s):
        s = "herb"
    return s

def _pcm_to_wav_bytes(pcm: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    """
    将 PCM_24000HZ_MONO_16BIT 包装成 wav
    """
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()

@mcp.tool()
def qwen_voice_design_create(
    voice_prompt: str,
    preview_text: str,
    preferred_name: str = "herb",
    target_model: str = DEFAULT_TARGET_MODEL,
    language: str = "zh",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    response_format: str = "wav",
) -> Dict[str, Any]:
    """
    通过文本描述生成一个“定制音色”(voice)，并返回预览音频（base64）。

    - model 固定为 qwen-voice-design
    - URL（中国内地）：POST https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization :contentReference[oaicite:5]{index=5}
    - input.action=create, input.target_model=你要后续使用的 TTS 模型（必须一致）:contentReference[oaicite:6]{index=6}
    """
    api_key = _get_api_key()
    host = _get_region_host()
    url = f"https://{host}/api/v1/services/audio/tts/customization"

    preferred_name = _safe_preferred_name(preferred_name)

    payload = {
        "model": "qwen-voice-design",
        "input": {
            "action": "create",
            "target_model": target_model,
            "voice_prompt": voice_prompt,
            "preview_text": preview_text,
            "preferred_name": preferred_name,
            "language": language,
        },
        "parameters": {
            "sample_rate": sample_rate,
            "response_format": response_format,
        },
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=90) as client:
        r = client.post(url, json=payload, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"voice_design_create失败: {r.status_code} {r.text}")

        data = r.json()

    # 文档示例里 output.voice + output.preview_audio.data(base64) :contentReference[oaicite:7]{index=7}
    voice = data["output"]["voice"]
    preview_audio_b64 = data["output"]["preview_audio"]["data"]

    return {
        "ok": True,
        "voice": voice,
        "target_model": target_model,
        "preview_audio_b64": preview_audio_b64,  # wav base64（按 response_format）
        "preferred_name": preferred_name,
    }

@dataclass
class _CollectCallback(QwenTtsRealtimeCallback):
    pcm_chunks: bytearray
    finished: bool = False
    error: Optional[str] = None

    def __init__(self):
        super().__init__()
        self.pcm_chunks = bytearray()
        self.finished = False
        self.error = None

    # SDK 的回调名不同版本可能略有差异；这里覆盖常见的两个：
    def on_audio_frame(self, audio: bytes, **kwargs):  # noqa
        if audio:
            self.pcm_chunks.extend(audio)

    def on_message(self, message: Any, **kwargs):  # noqa
        # 允许你后续打印调试
        pass

    def on_error(self, error: Any, **kwargs):  # noqa
        self.error = str(error)
        self.finished = True

    def on_close(self, **kwargs):  # noqa
        self.finished = True

@mcp.tool()
def qwen_tts_vd_speak(
    text: str,
    voice: str,
    model: str = DEFAULT_TARGET_MODEL,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    response_format: str = "wav",
    mode: str = "server_commit",
    language_type: str = "Auto",
) -> Dict[str, Any]:
    """
    用“声音设计(VD)”生成的 voice 进行语音合成，返回音频 base64。

    - Realtime WebSocket endpoint（中国内地）：wss://dashscope.aliyuncs.com/api-ws/v1/realtime :contentReference[oaicite:8]{index=8}
    - 会话里设置 voice、response_format、sample_rate（文档示例）:contentReference[oaicite:9]{index=9}
    """
    api_key = _get_api_key()
    dashscope.api_key = api_key

    # 采样格式：优先走 PCM_24000HZ_MONO_16BIT，再自己封装 wav（更通用）
    # 你也可以改为 SDK 直接输出 wav（取决于 SDK 支持情况）
    tts_format = AudioFormat.PCM_24000HZ_MONO_16BIT

    cb = _CollectCallback()
    host = _get_region_host()
    ws_url = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime" if host == "dashscope.aliyuncs.com" \
        else "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"

    q = QwenTtsRealtime(model=model, callback=cb, url=ws_url)
    q.connect()

    # 设置会话参数：voice / sample_rate / response_format 等（文档示例里就是这么配的）:contentReference[oaicite:10]{index=10}
    q.update_session(
        voice=voice,
        response_format=tts_format,
        mode=mode,
        language_type=language_type,
    )

    # 发送文本
    q.append_text(text)
    time.sleep(0.05)
    q.finish()

    # 等待结束（最多 30 秒）
    t0 = time.time()
    while not cb.finished and (time.time() - t0) < 30:
        time.sleep(0.05)

    q.close()

    if cb.error:
        raise RuntimeError(f"TTS 合成失败: {cb.error}")

    pcm = bytes(cb.pcm_chunks)
    if not pcm:
        raise RuntimeError("TTS 未返回音频数据（pcm为空）")

    if response_format.lower() == "wav":
        audio_bytes = _pcm_to_wav_bytes(pcm, sample_rate=sample_rate)
        mime = "audio/wav"
    else:
        audio_bytes = pcm
        mime = "audio/pcm"

    return {
        "ok": True,
        "model": model,
        "voice": voice,
        "mime": mime,
        "audio_b64": base64.b64encode(audio_bytes).decode("utf-8"),
        "bytes": len(audio_bytes),
    }

if __name__ == "__main__":
    # stdio 方式启动 MCP server（给 MCP Host/你的主程序连接）
    mcp.run()