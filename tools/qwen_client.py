import os
from typing import List, Dict
from openai import OpenAI

class QwenClient:
    """DashScope OpenAI-compatible client wrapper."""

    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY or DASHSCOPE_API_KEY")

        base_url = os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = os.environ.get("OPENAI_MODEL", "qwen-plus")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.2, max_tokens: int = 900, stream: bool = False) -> str:
        if stream:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            out = ""
            for chunk in resp:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    print(content, end="", flush=True)
                    out += content
            print()
            return out.strip()

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()