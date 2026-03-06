from openai import OpenAI

client = OpenAI(
    api_key="sk-731eb815e22f4eb498c81d28f6e95668",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

resp = client.chat.completions.create(
    model="qwen-plus-2025-07-28",
    messages=[{"role": "user", "content": "你是谁"}]
)

print(resp.choices[0].message.content)
