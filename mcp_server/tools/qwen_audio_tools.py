# -*- coding: utf-8 -*-
import os
import re
import json
import time
import base64
from typing import Dict, Any, Optional

import httpx
import dashscope
from dashscope.audio.qwen_tts_realtime import QwenTtsRealtime, QwenTtsRealtimeCallback  # :contentReference[oaicite:4]{index=4}

AUDIO_DIR = os.getenv("AUDIO_DIR", "assets/audio")
VOICE_CACHE_PATH = os.getenv("VOICE_CACHE_PATH", "assets/personas/voice_cache.json")

DEFAULT_TARGET_MODEL = "qwen3-tts-vd-realtime-2026-01-15"
DEFAULT_SAMPLE_RATE = 24000


def _ensure_dirs():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(VOICE_CACHE_PATH), exist_ok=True)


def _get_api_key() -> str:
    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY")
    return api_key


def _get_host() -> str:
    # cn: dashscope.aliyuncs.com; intl: dashscope-intl.aliyuncs.com
    region = (os.getenv("DASHSCOPE_REGION") or "cn").lower().strip()
    return "dashscope-intl.aliyuncs.com" if region == "intl" else "dashscope.aliyuncs.com"


def _safe_preferred_name(s: str) -> str:
    """
    preferred_name: 仅允许数字、英文字母和下划线，不超过16字符。:contentReference[oaicite:5]{index=5}
    """
    s = (s or "").strip()
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
    s = s[:16] if s else "herb"
    if not re.fullmatch(r"[0-9A-Za-z_]{1,16}", s):
        return "herb"
    return s


def _load_voice_cache() -> Dict[str, Any]:
    _ensure_dirs()
    if not os.path.exists(VOICE_CACHE_PATH):
        return {}
    try:
        with open(VOICE_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_voice_cache(cache: Dict[str, Any]) -> None:
    _ensure_dirs()
    with open(VOICE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def voice_design_create(
    voice_prompt: str,
    preview_text: str,
    preferred_name: str,
    target_model: str = DEFAULT_TARGET_MODEL,
    language: str = "zh",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    response_format: str = "wav",
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    调用声音设计接口（qwen-voice-design）：/api/v1/services/audio/tts/customization :contentReference[oaicite:6]{index=6}
    返回 voice + preview_audio(base64)
    """
    _ensure_dirs()
    preferred_name = _safe_preferred_name(preferred_name)

    cache_key = f"{target_model}::{preferred_name}::{hash(voice_prompt)}"
    if use_cache:
        cache = _load_voice_cache()
        if cache_key in cache:
            return {"ok": True, "cached": True, **cache[cache_key]}

    api_key = _get_api_key()
    host = _get_host()
    url = f"https://{host}/api/v1/services/audio/tts/customization"  # :contentReference[oaicite:7]{index=7}

    payload = {
        "model": "qwen-voice-design",
        "input": {
            "action": "create",
            "target_model": target_model,          # 需与后续合成模型一致:contentReference[oaicite:8]{index=8}
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
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    with httpx.Client(timeout=120) as client:
        r = client.post(url, json=payload, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"voice_design_create failed: {r.status_code} {r.text}")
        data = r.json()

    out = {
        "voice": data["output"]["voice"],
        "target_model": target_model,
        "preferred_name": preferred_name,
        "preview_audio_b64": data["output"]["preview_audio"]["data"],  # base64
        "created_at": int(time.time()),
    }

    if use_cache:
        cache = _load_voice_cache()
        cache[cache_key] = out
        _save_voice_cache(cache)

    return {"ok": True, "cached": False, **out}


# mcp_server/tools/qwen_audio_tools.py
# mcp_server/tools/qwen_audio_tools.py (只贴需要替换的部分)
import os
import time
import base64
import threading
from typing import Optional, Dict, Any

import dashscope
from dashscope.audio.qwen_tts_realtime import QwenTtsRealtime, QwenTtsRealtimeCallback, AudioFormat

AUDIO_DIR = os.getenv("AUDIO_DIR", "assets/audio")

def _ensure_dirs():
    os.makedirs(AUDIO_DIR, exist_ok=True)

def _get_api_key() -> str:
    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY")
    return api_key

def _get_realtime_ws_url() -> str:
    # 官方示例：北京 vs 新加坡 URL 不同 :contentReference[oaicite:4]{index=4}
    region = (os.getenv("DASHSCOPE_REGION") or "cn").lower().strip()
    return "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime" if region == "intl" else "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"

class _Collector(QwenTtsRealtimeCallback):
    """
    按官方 SDK 示例：on_event 收到的是 dict，音频事件 type= response.audio.delta，
    base64 数据在 response['delta'] :contentReference[oaicite:5]{index=5}
    """
    def __init__(self):
        super().__init__()
        self.complete_event = threading.Event()
        self.audio_bytes = bytearray()
        self.error: Optional[str] = None

    def on_open(self) -> None:
        # 可选：print("tts ws opened")
        pass

    def on_close(self, close_status_code=None, close_msg=None) -> None:
        # 可选：print("tts ws closed", close_status_code, close_msg)
        # 连接关闭也视为结束
        self.complete_event.set()

    def on_event(self, response) -> None:
        try:
            # 注意：response 是 dict（官方示例如此） :contentReference[oaicite:6]{index=6}
            event_type = response.get("type") if isinstance(response, dict) else None
            if not event_type:
                return

            # 音频增量：base64 在 delta 字段 :contentReference[oaicite:7]{index=7}
            if event_type == "response.audio.delta":
                b64 = response.get("delta")
                if b64:
                    self.audio_bytes.extend(base64.b64decode(b64))
                return

            # done / finished
            if event_type in ("response.done", "session.finished"):
                self.complete_event.set()
                return

            # 可选：处理错误事件
            if event_type in ("error", "failed"):
                self.error = str(response)
                self.complete_event.set()

        except Exception as e:
            self.error = f"[CollectorError] {e}"
            self.complete_event.set()

    def wait_done(self, timeout: float = 30.0) -> bool:
        return self.complete_event.wait(timeout)

def _pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1, sampwidth: int = 2) -> bytes:
    """
    把 PCM16LE 封装成 WAV，前端 <audio> 更好播
    """
    import io, wave
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return bio.getvalue()

def tts_vd_realtime_speak(
    text: str,
    voice: str,
    model: str = "qwen3-tts-vd-realtime-2026-01-15",
    sample_rate: int = 24000,
    save_wav: bool = True,
) -> Dict[str, Any]:
    """
    使用 realtime TTS 合成音频（VD voice）。
    关键点：按官方事件类型 response.audio.delta 解析音频 :contentReference[oaicite:8]{index=8}
    """
    text = (text or "").strip()
    voice = (voice or "").strip()
    if not text:
        return {"ok": False, "error": "empty text"}
    if not voice:
        return {"ok": False, "error": "empty voice"}

    dashscope.api_key = _get_api_key()

    cb = _Collector()
    tts = QwenTtsRealtime(
        model=model,
        callback=cb,
        url=_get_realtime_ws_url(),  # 地域对应 URL :contentReference[oaicite:9]{index=9}
    )
    tts.connect()

    # 官方示例：update_session 指定 voice/response_format/mode :contentReference[oaicite:10]{index=10}
    tts.update_session(
        voice=voice,
        response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
        mode="server_commit",
    )

    # 这里一次性 append，也可以分片 append（官方示例分片） :contentReference[oaicite:11]{index=11}
    tts.append_text(text)
    tts.finish()

    done = cb.wait_done(timeout=30.0)
    tts.close()

    if cb.error:
        return {"ok": False, "error": cb.error}

    if not done:
        return {"ok": False, "error": "timeout waiting tts"}

    if not cb.audio_bytes:
        # 现在不会再抛异常，让上层稳定拿到 JSON
        return {
            "ok": False,
            "error": "TTS produced empty audio (no response.audio.delta received). "
                     "Check dashscope SDK version >= 1.25.11 and model/voice permissions."
        }

    pcm = bytes(cb.audio_bytes)
    _ensure_dirs()

    if save_wav:
        wav_bytes = _pcm_to_wav_bytes(pcm, sample_rate=sample_rate)
        fname = f"tts_{int(time.time())}.wav"
        out_path = os.path.join(AUDIO_DIR, fname)
        with open(out_path, "wb") as f:
            f.write(wav_bytes)
        return {"ok": True, "model": model, "voice": voice, "audio_path": out_path.replace("\\", "/"), "bytes": len(wav_bytes)}

    # 不封装 wav 就直接存 pcm
    fname = f"tts_{int(time.time())}.pcm"
    out_path = os.path.join(AUDIO_DIR, fname)
    with open(out_path, "wb") as f:
        f.write(pcm)
    return {"ok": True, "model": model, "voice": voice, "audio_path": out_path.replace("\\", "/"), "bytes": len(pcm)}

def preferred_file_safe(s: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s or "")
    return s[:24] if s else "voice"