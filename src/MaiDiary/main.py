"""
MaiSaka - 程序入口
使用方法:
    python main.py

环境变量 (可通过 .env 文件设置):
    OPENAI_API_KEY   - API 密钥
    OPENAI_BASE_URL  - API 基地址 (可选, 默认 https://api.openai.com/v1)
    OPENAI_MODEL     - 模型名称 (可选, 默认 gpt-4o)
    ENABLE_THINKING  - 是否启用思考模式 (可选, true/false, 不设置则不发送该参数)
"""

import asyncio

from config import console
from cli import BufferCLI


def main():
    cli = BufferCLI()
    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        console.print("\n[muted]程序已终止[/muted]")
    finally:
        cli._debug_viewer.close()


if __name__ == "__main__":
    main()
