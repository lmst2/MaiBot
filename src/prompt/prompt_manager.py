from collections.abc import Callable, Coroutine
from typing import Any, Optional
from string import Formatter
from pathlib import Path
import inspect

from src.common.logger import get_logger


logger = get_logger("Prompt")

_LEFT_BRACE = "\ufde9"
_RIGHT_BRACE = "\ufdea"

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute().resolve()
PROMPTS_DIR = PROJECT_ROOT / "prompts"
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
SUFFIX_PROMPT = ".prompt"


class Prompt:
    prompt_name: str
    template: str
    prompt_render_context: dict[str, Callable[[str], str | Coroutine[Any, Any, str]]] = {}

    def __init__(self, prompt_name: str, template: str) -> None:
        self.prompt_name = prompt_name
        self.template = template
        self.__post_init__()

    def add_context(self, name: str, func: Callable[[str], str | Coroutine[Any, Any, str]]) -> None:
        if name in self.prompt_render_context:
            raise KeyError(f"Context function name '{name}' 已存在于 Prompt '{self.prompt_name}' 中")
        self.prompt_render_context[name] = func

    def __post_init__(self):
        if not self.prompt_name:
            raise ValueError("prompt_name 不能为空")
        if not self.template:
            raise ValueError("template 不能为空")
        tmp = self.template.replace("{{", _LEFT_BRACE).replace("}}", _RIGHT_BRACE)
        if "{}" in tmp:
            raise ValueError(r"模板中不允许使用未命名的占位符 '{}'")


class PromptManager:
    def __init__(self) -> None:
        PROMPTS_DIR.mkdir(parents=True, exist_ok=True)  # 确保提示词目录存在
        self.prompts: dict[str, Prompt] = {}
        """存储 Prompt 实例"""
        self._context_construct_functions: dict[str, tuple[Callable[[str], str | Coroutine[Any, Any, str]], str]] = {}
        """存储上下文构造函数及其所属模块"""
        self._formatter = Formatter()  # 仅用来解析模板
        """模板解析器"""
        self._prompt_to_save: set[str] = set()
        """需要保存的 Prompt 名称集合"""

    def add_prompt(self, prompt: Prompt, need_save: bool = False) -> None:
        if prompt.prompt_name in self.prompts or prompt.prompt_name in self._context_construct_functions:
            raise KeyError(f"Prompt name '{prompt.prompt_name}' 已存在")
        self.prompts[prompt.prompt_name] = prompt
        if need_save:
            self._prompt_to_save.add(prompt.prompt_name)

    def add_context_construct_function(self, name: str, func: Callable[[str], str | Coroutine[Any, Any, str]]) -> None:
        if name in self._context_construct_functions or name in self.prompts:
            raise KeyError(f"Construct function name '{name}' 已存在")
        # 获取调用栈
        frame = inspect.currentframe()
        if not frame:
            # 不应该出现的情况
            raise RuntimeError("无法获取调用栈")
        caller_frame = frame.f_back
        if not caller_frame:
            # 不应该出现的情况
            raise RuntimeError("无法获取调用栈的上一级")
        caller_module = caller_frame.f_globals.get("__name__", "unknown")
        if caller_module == "unknown":
            logger.warning("无法获取调用函数的模块名，使用 'unknown' 作为默认值")

        self._context_construct_functions[name] = func, caller_module

    def get_prompt(self, prompt_name: str) -> Prompt:
        if prompt_name not in self.prompts:
            raise KeyError(f"Prompt name '{prompt_name}' 不存在")
        return self.prompts[prompt_name]

    async def render_prompt(self, prompt: Prompt) -> str:
        return await self._render(prompt)

    async def _render(self, prompt: Prompt, recursive_level: int = 0) -> str:
        prompt.template = prompt.template.replace("{{", _LEFT_BRACE).replace("}}", _RIGHT_BRACE)
        if recursive_level > 10:
            raise RecursionError("递归层级过深，可能存在循环引用")
        field_block = {field_name for _, field_name, _, _ in self._formatter.parse(prompt.template) if field_name}
        rendered_fields: dict[str, str] = {}
        for field_name in field_block:
            if field_name in self.prompts:
                rendered_fields[field_name] = await self._render(self.prompts[field_name], recursive_level + 1)
            elif field_name in prompt.prompt_render_context:
                func = prompt.prompt_render_context[field_name]
                rendered_fields[field_name] = await self._get_function_result(
                    func, prompt.prompt_name, field_name, is_prompt_context=True
                )
            elif field_name in self._context_construct_functions:
                func, module = self._context_construct_functions[field_name]
                rendered_fields[field_name] = await self._get_function_result(
                    func, prompt.prompt_name, field_name, is_prompt_context=False, module=module
                )
            else:
                raise KeyError(f"Prompt '{prompt.prompt_name}' 中缺少必要的内容块或构建函数: '{field_name}'")
        rendered_template = prompt.template.format(**rendered_fields)
        return rendered_template.replace(_LEFT_BRACE, "{").replace(_RIGHT_BRACE, "}")

    def save_prompts(self) -> None:
        for prompt_name in self._prompt_to_save:
            prompt = self.prompts[prompt_name]
            file_path = PROMPTS_DIR / f"{prompt_name}{SUFFIX_PROMPT}"
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(prompt.template)
            except Exception as e:
                logger.error(f"保存 Prompt '{prompt_name}' 时出错，文件路径: '{file_path}'，错误信息: {e}")
                raise e

    def load_prompts(self) -> None:
        for prompt_file in PROMPTS_DIR.glob(f"*{SUFFIX_PROMPT}"):
            try:
                with open(prompt_file, "r", encoding="utf-8") as f:
                    template = f.read()
                self.add_prompt(Prompt(prompt_name=prompt_file.stem, template=template), need_save=True)
            except Exception as e:
                logger.error(f"加载 Prompt 文件 '{prompt_file}' 时出错，错误信息: {e}")
                raise e

    async def _get_function_result(
        self,
        func: Callable[[str], str | Coroutine[Any, Any, str]],
        prompt_name: str,
        field_name: str,
        is_prompt_context: bool,
        module: Optional[str] = None,
    ) -> str:
        try:
            res = func(prompt_name)
            if isinstance(res, Coroutine):
                res = await res
            return res
        except Exception as e:
            if is_prompt_context:
                logger.error(f"调用 Prompt '{prompt_name}' 内部上下文构造函数 '{field_name}' 时出错，错误信息: {e}")
            else:
                logger.error(f"调用上下文构造函数 '{field_name}' 时出错，所属模块: '{module}'，错误信息: {e}")
            raise e


prompt_manager = PromptManager()
