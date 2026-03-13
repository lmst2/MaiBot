"""
MaiSaka - 全局配置
从主项目配置系统读取配置、Rich Console 实例、主题定义。
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.theme import Theme

# 添加项目根目录到路径以导入主配置
_root = Path(__file__).parent.parent.parent.absolute()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# ──────────────────── 从主配置读取 ────────────────────


def _get_maisaka_config():
    """获取 MaiSaka 配置"""
    try:
        from src.config.config import config_manager

        return config_manager.config.maisaka
    except Exception:
        # 如果配置加载失败，返回默认值
        from src.config.official_configs import MaiSakaConfig

        return MaiSakaConfig()


_maisaka_config = _get_maisaka_config()

# ──────────────────── 模块开关配置 ────────────────────

ENABLE_EMOTION_MODULE = _maisaka_config.enable_emotion_module
ENABLE_COGNITION_MODULE = _maisaka_config.enable_cognition_module
# Timing 模块已包含自我反思功能
ENABLE_TIMING_MODULE = _maisaka_config.enable_timing_module
ENABLE_KNOWLEDGE_MODULE = _maisaka_config.enable_knowledge_module
ENABLE_MCP = _maisaka_config.enable_mcp
ENABLE_WRITE_FILE = _maisaka_config.enable_write_file
ENABLE_READ_FILE = _maisaka_config.enable_read_file
ENABLE_LIST_FILES = _maisaka_config.enable_list_files

# ──────────────────── QQ 工具配置 ────────────────────

ENABLE_QQ_TOOLS = _maisaka_config.enable_qq_tools
QQ_API_BASE_URL = _maisaka_config.qq_api_base_url
QQ_API_KEY = _maisaka_config.qq_api_key

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
