"""
MaiSaka - 全局配置
从主项目配置系统读取配置、Rich Console 实例、主题定义。
"""

from pathlib import Path
import sys

from rich.console import Console
from rich.theme import Theme

from src.config.config import global_config

# 添加项目根目录到路径以导入主配置
_root = Path(__file__).parent.parent.parent.absolute()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# ──────────────────── 模块开关配置 ────────────────────
ENABLE_EMOTION_MODULE = global_config.maisaka.enable_emotion_module
ENABLE_COGNITION_MODULE = global_config.maisaka.enable_cognition_module
ENABLE_KNOWLEDGE_MODULE = global_config.maisaka.enable_knowledge_module
ENABLE_MCP = global_config.maisaka.enable_mcp
ENABLE_WRITE_FILE = global_config.maisaka.enable_write_file
ENABLE_READ_FILE = global_config.maisaka.enable_read_file
ENABLE_LIST_FILES = global_config.maisaka.enable_list_files
SHOW_ANALYZE_COGNITION_PROMPT = global_config.maisaka.show_analyze_cognition_prompt
SHOW_THINKING = global_config.maisaka.show_thinking
USER_NAME = global_config.maisaka.user_name.strip() or "用户"
DIRECT_IMAGE_INPUT = global_config.maisaka.direct_image_input
MERGE_USER_MESSAGES = global_config.maisaka.merge_user_messages
TERMINAL_IMAGE_PREVIEW = global_config.maisaka.terminal_image_preview
TERMINAL_IMAGE_PREVIEW_WIDTH = global_config.maisaka.terminal_image_preview_width
TAKE_OVER_HFC = global_config.maisaka.take_over_hfc


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
