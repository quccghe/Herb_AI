from typing import Any, Dict, Optional, List
import requests
import os
import base64
import uuid


def save_audio_file(audio_base64: str, save_path: str) -> str:
    """
    将 base64 音频保存为 wav 文件
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    audio_bytes = base64.b64decode(audio_base64)

    with open(save_path, "wb") as f:
        f.write(audio_bytes)

    return save_path


class MCPClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def health(self) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/health", timeout=10)
        try:
            return r.json()
        except Exception:
            return {"ok": False, "status_code": r.status_code, "raw": r.text[:500]}

    def kg_get_node(self, name: str) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/tools/kg_get_node",
            json={"name": name},
            timeout=15
        )
        try:
            return r.json()
        except Exception:
            return {"ok": False, "status_code": r.status_code, "raw": r.text[:500]}

    def kg_neighbors(
        self,
        name: str,
        limit: int = 30,
        rel_types: Optional[List[str]] = None,
        neighbor_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "name": name,
            "limit": limit,
            "rel_types": rel_types,
            "neighbor_labels": neighbor_labels
        }
        r = requests.post(f"{self.base_url}/tools/kg_neighbors", json=payload, timeout=15)
        try:
            return r.json()
        except Exception:
            return {"ok": False, "status_code": r.status_code, "raw": r.text[:500]}

    def kg_paths(self, a: str, b: str, k: int = 3, max_hops: int = 3) -> Dict[str, Any]:
        payload = {"a": a, "b": b, "k": k, "max_hops": max_hops}
        r = requests.post(f"{self.base_url}/tools/kg_paths", json=payload, timeout=20)
        try:
            return r.json()
        except Exception:
            return {"ok": False, "status_code": r.status_code, "raw": r.text[:500]}

    def voice_design_create(
        self,
        voice_prompt: str,
        preview_text: str,
        preferred_name: str = "herb",
        target_model: str = "qwen3-tts-vd-realtime-2026-01-15",
        language: str = "zh",
        sample_rate: int = 24000,
        response_format: str = "wav",
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        payload = {
            "voice_prompt": voice_prompt,
            "preview_text": preview_text,
            "preferred_name": preferred_name,
            "target_model": target_model,
            "language": language,
            "sample_rate": sample_rate,
            "response_format": response_format,
            "use_cache": use_cache,
        }
        r = requests.post(f"{self.base_url}/tools/voice_design_create", json=payload, timeout=120)
        try:
            return r.json()
        except Exception:
            return {"ok": False, "status_code": r.status_code, "raw": r.text[:500]}

    def tts_vd_realtime_speak(self, text: str, voice: str, model: Optional[str] = None) -> Dict[str, Any]:
        payload = {
            "text": text,
            "voice": voice,
            "model": model
        }

        r = requests.post(
            f"{self.base_url}/tools/tts_vd_realtime_speak",
            json=payload,
            timeout=120
        )

        try:
            data = r.json()
        except Exception:
            return {"ok": False, "status_code": r.status_code, "raw": r.text[:500]}

        if not data.get("ok"):
            return data

        audio_b64 = data.get("audio_base64")
        if not audio_b64:
            return data

        filename = f"tmp_{uuid.uuid4().hex}.wav"
        save_path = f"assets/audio/tmp/{filename}"

        path = save_audio_file(audio_b64, save_path)
        data["audio_path"] = path

        return data

    def kg_subgraph(
        self,
        name: str,
        depth: int = 1,
        include_types=None,
        max_nodes_per_type: int = 20,
    ) -> Dict[str, Any]:
        payload = {
            "name": name,
            "depth": depth,
            "include_types": include_types,
            "max_nodes_per_type": max_nodes_per_type,
        }
        r = requests.post(f"{self.base_url}/tools/kg_subgraph", json=payload, timeout=30)
        try:
            return r.json()
        except Exception:
            return {"ok": False, "status_code": r.status_code, "raw": r.text[:500]}

    def kg_relation_paths(self, source: str, target: str, max_hops: int = 3) -> Dict[str, Any]:
        payload = {
            "source": source,
            "target": target,
            "max_hops": max_hops,
        }
        r = requests.post(f"{self.base_url}/tools/kg_relation_paths", json=payload, timeout=30)
        try:
            return r.json()
        except Exception:
            return {"ok": False, "status_code": r.status_code, "raw": r.text[:500]}

    def kg_graph_summary(self, name: str) -> Dict[str, Any]:
        payload = {"name": name}
        r = requests.post(f"{self.base_url}/tools/kg_graph_summary", json=payload, timeout=30)
        try:
            return r.json()
        except Exception:
            return {"ok": False, "status_code": r.status_code, "raw": r.text[:500]}

    def wan_text_to_image(
        self,
        prompt: str,
        herb_name: str = "",
        size: str = "1024*1024",
        style_hint: Optional[str] = None,
        watermark: bool = False,
    ) -> Dict[str, Any]:
        payload = {
            "prompt": prompt,
            "herb_name": herb_name,
            "size": size,
            "style_hint": style_hint,
            "watermark": watermark,
        }
        r = requests.post(f"{self.base_url}/tools/wan_text_to_image", json=payload, timeout=240)
        try:
            return r.json()
        except Exception:
            return {"ok": False, "status_code": r.status_code, "raw": r.text[:500]}