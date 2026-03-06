from typing import Dict, Any
from tools.qwen_client import QwenClient

class FlavorStyleAgent:
    """根据性味控制语气，生成第一人称自述（200-300字）。"""

    def __init__(self, llm: QwenClient):
        self.llm = llm

    def _tone_from_taste(self, taste: str) -> str:
        t = taste or ""
        if "热" in t:
            return "语气略带热情与主动感（但仍严谨）"
        if "寒" in t:
            return "语气冷静克制，略带高冷感（但仍严谨）"
        if "温" in t:
            return "语气温和自然、亲切"
        if "凉" in t:
            return "语气理性平稳、清爽"
        return "语气客观中性"

    def run(self, card: Dict[str, Any]) -> Dict[str, Any]:
        name = card.get("name", "")
        taste = card.get("taste", "")
        tone = self._tone_from_taste(taste)
        evidence = card.get("evidence", "")

        prompt = f"""
你要写“中药第一人称自述”，让用户感觉这味药在自我介绍。
不要列表、不要表格。

主题：{name}
风格要求：{tone}
字数：200-300字

必须依据下面证据，不在证据中的细节不要硬编；不确定用“资料提示/常用于”等措辞：
{evidence}
""".strip()

        messages = [
            {"role": "system", "content": f"你是中药科普写作者，当前主题：{name}。"},
            {"role": "user", "content": prompt},
        ]
        text = self.llm.chat(messages, temperature=0.4, max_tokens=520, stream=False)
        return {"type": "flavor_narration", "name": name, "tone": tone, "text": text}