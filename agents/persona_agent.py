from typing import Dict, Any
import json
from tools.qwen_client import QwenClient

class PersonaAgent:
    """根据卡片+自述，生成拟人角色设定（严格JSON）。"""

    def __init__(self, llm: QwenClient):
        self.llm = llm

    def run(self, card: Dict[str, Any], narration: Dict[str, Any]) -> Dict[str, Any]:
        name = card.get("name", "")
        taste = card.get("taste", "")
        tone = narration.get("tone", "")
        narration_text = narration.get("text", "")

        prompt = f"""
请为中药“{name}”生成一个拟人化角色设定，用于后续生成形象与配音。

输入信息：
- 性味：{taste}
- 自述风格：{tone}
- 第一人称自述（供参考）：{narration_text}

输出必须是严格 JSON（不要 markdown），字段如下：
{{
  "role_title": "...",
  "age_style": "...",
  "appearance_tags": ["..."],
  "personality_tags": ["..."],
  "catchphrases": ["..."],
  "voice_recommendation": {{
      "voice": "...",
      "speed": 1.0,
      "pitch": 1.0,
      "emotion": "..."
  }},
  "image_prompt": "..."
}}

要求：
- 角色设定要稳定、克制、不低俗
- 不要涉及真实人物或公众人物
- voice 字段先写“音色描述”，后续可映射到具体阿里音色ID
""".strip()

        messages = [
            {"role": "system", "content": "你是角色设定师，擅长把中药抽象成稳定的人格形象。"},
            {"role": "user", "content": prompt},
        ]
        raw = self.llm.chat(messages, temperature=0.6, max_tokens=900, stream=False)

        try:
            obj = json.loads(raw)
            return {"type": "persona", "name": name, "persona": obj}
        except Exception:
            return {"type": "persona", "name": name, "persona_raw": raw}