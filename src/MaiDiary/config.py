"""
MaiSaka - 全局配置
环境变量加载、Rich Console 实例、主题定义。
"""

import os
from dotenv import load_dotenv
from rich.console import Console
from rich.theme import Theme

# ──────────────────── 加载 .env ────────────────────

load_dotenv()

# ──────────────────── 模块开关配置 ────────────────────

ENABLE_EMOTION_MODULE = os.getenv("ENABLE_EMOTION_MODULE", "true").strip().lower() == "true"
ENABLE_COGNITION_MODULE = os.getenv("ENABLE_COGNITION_MODULE", "true").strip().lower() == "true"
# Timing 模块已包含自我反思功能
ENABLE_TIMING_MODULE = os.getenv("ENABLE_TIMING_MODULE", "true").strip().lower() == "true"
ENABLE_KNOWLEDGE_MODULE = os.getenv("ENABLE_KNOWLEDGE_MODULE", "true").strip().lower() == "true"
ENABLE_MCP = os.getenv("ENABLE_MCP", "true").strip().lower() == "true"
ENABLE_WRITE_FILE = os.getenv("ENABLE_WRITE_FILE", "true").strip().lower() == "true"
ENABLE_READ_FILE = os.getenv("ENABLE_READ_FILE", "true").strip().lower() == "true"
ENABLE_LIST_FILES = os.getenv("ENABLE_LIST_FILES", "true").strip().lower() == "true"

# ──────────────────── QQ 工具配置 ────────────────────

ENABLE_QQ_TOOLS = os.getenv("ENABLE_QQ_TOOLS", "false").strip().lower() == "true"
QQ_API_BASE_URL = os.getenv("QQ_API_BASE_URL", "").strip()
QQ_API_KEY = os.getenv("QQ_API_KEY", "").strip()

# ──────────────────── Rich 主题 & Console ────────────────────

custom_theme = Theme(
    {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "bold red",
        "muted": "dim",
        "accent": "bold magenta",
    }
)

console = Console(theme=custom_theme)
