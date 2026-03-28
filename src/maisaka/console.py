"""MaiSaka 终端输出组件。"""

from rich.console import Console
from rich.theme import Theme

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
