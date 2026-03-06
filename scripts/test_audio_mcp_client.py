# -*- coding: utf-8 -*-
import asyncio
import base64
import os

from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession


SERVER_CMD = ["python", "-u", "mcp_servers/qwen_audio_server.py"]

async def main():
    async with stdio_client(SERVER_CMD) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1) 先做 Voice Design：用音色描述生成 voice
            voice_prompt = "中低频饱满、语速均匀的成年男声，声线温润沉稳，吐字清晰，情绪平和持重。"
            preview_text = "大家好，我是甘草。今天我来用温和的方式介绍自己。"

            r1 = await session.call_tool(
                "qwen_voice_design_create",
                {
                    "voice_prompt": voice_prompt,
                    "preview_text": preview_text,
                    "preferred_name": "gancao",
                    "target_model": "qwen3-tts-vd-realtime-2026-01-15",
                    "language": "zh",
                    "sample_rate": 24000,
                    "response_format": "wav",
                },
            )
            data1 = r1.content[0].text  # FastMCP 默认把 dict 转成 text/json
            print("voice_design result:", data1)

            # 你也可以在这里解析 JSON；简化起见，直接再次调用 query/list 也行
            # 我们这里只演示：把 preview_audio_b64 写文件（你自己 parse 出来）
            # 建议你把 mcp_server 返回值直接做成严格 JSON 字符串，客户端用 json.loads 解析。

            # 2) 合成：你需要拿到 voice 名称（从返回里解析出来）
            # 这里偷懒：让你手动把 voice 复制出来更直观
            voice = input("把上一步输出中的 voice 粘贴到这里：").strip()

            r2 = await session.call_tool(
                "qwen_tts_vd_speak",
                {
                    "text": "我守中，故能调众味。烈性非我所拒，唯需明其所向。",
                    "voice": voice,
                    "model": "qwen3-tts-vd-realtime-2026-01-15",
                    "response_format": "wav",
                    "sample_rate": 24000,
                },
            )
            data2 = r2.content[0].text
            print("tts result:", data2)

            # 同样建议你 json.loads 后取 audio_b64 写文件；
            # 这里示例：手动粘贴 audio_b64（为了不写解析逻辑）
            audio_b64 = input("把上一步输出中的 audio_b64 粘贴到这里：").strip()
            wav = base64.b64decode(audio_b64)
            with open("out.wav", "wb") as f:
                f.write(wav)
            print("已写出 out.wav")

if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    asyncio.run(main())