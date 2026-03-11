"""
MaiSaka - Prompt 加载器
支持从 .prompt 文件加载模板，并进行变量替换。
"""

import os
from pathlib import Path
from typing import Any


class PromptLoader:
    """Prompt 模板加载器"""

    def __init__(self, prompts_dir: str | None = None):
        """
        初始化加载器。

        Args:
            prompts_dir: prompts 目录路径，默认为项目根目录下的 prompts/
        """
        if prompts_dir is None:
            # 默认为项目根目录下的 prompts/
            project_root = Path(__file__).parent
            prompts_dir = project_root / "prompts"

        self.prompts_dir = Path(prompts_dir)
        self._cache: dict[str, str] = {}

    def load(self, name: str, **kwargs: Any) -> str:
        """
        加载并渲染 prompt 模板。

        Args:
            name: 模板文件名（不含 .prompt 后缀）
            **kwargs: 模板变量

        Returns:
            渲染后的 prompt 文本
        """
        # 从缓存读取
        if name not in self._cache:
            template_path = self.prompts_dir / f"{name}.prompt"
            if not template_path.exists():
                raise FileNotFoundError(f"Prompt template not found: {template_path}")

            self._cache[name] = template_path.read_text(encoding="utf-8")

        template = self._cache[name]

        # 变量替换
        if kwargs:
            try:
                return template.format(**kwargs)
            except KeyError as e:
                raise ValueError(f"Missing template variable: {e}") from e

        return template

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()


# 全局单例
_loader = PromptLoader()


def load_prompt(name: str, **kwargs: Any) -> str:
    """
    加载并渲染 prompt 模板（全局函数）。

    Args:
        name: 模板文件名（不含 .prompt 后缀）
        **kwargs: 模板变量

    Returns:
        渲染后的 prompt 文本
    """
    return _loader.load(name, **kwargs)


def reload_prompts():
    """重新加载所有 prompt（清空缓存）"""
    _loader.clear_cache()
