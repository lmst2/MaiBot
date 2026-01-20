# File: tests/test_prompt_manager.py

import asyncio
import inspect
from pathlib import Path
from typing import Any
import sys

import pytest

PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.absolute().resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "config"))

from src.prompt.prompt_manager import SUFFIX_PROMPT, Prompt, PromptManager, prompt_manager # noqa


@pytest.mark.parametrize(
    "prompt_name, template",
    [
        pytest.param("simple", "Hello {name}", id="simple-template-with-field"),
        pytest.param("no-fields", "Just a static template", id="template-without-fields"),
        pytest.param("brace-escaping", "Use {{ and }} around {field}", id="template-with-escaped-braces"),
    ],
)
def test_prompt_init_happy_paths(prompt_name: str, template: str):
    # Act
    prompt = Prompt(prompt_name=prompt_name, template=template)

    # Assert
    assert prompt.prompt_name == prompt_name
    assert prompt.template == template


@pytest.mark.parametrize(
    "prompt_name, template, expected_exception, expected_msg_substring",
    [
        pytest.param("", "Hello {name}", ValueError, "prompt_name 不能为空", id="empty-prompt-name"),
        pytest.param("valid-name", "", ValueError, "template 不能为空", id="empty-template"),
        pytest.param(
            "unnamed-placeholder",
            "Hello {}",
            ValueError,
            "模板中不允许使用未命名的占位符",
            id="unnamed-placeholder-not-allowed",
        ),
        pytest.param(
            "unnamed-placeholder-with-escaped-brace",
            "Value {{}} and {}",
            ValueError,
            "模板中不允许使用未命名的占位符",
            id="unnamed-placeholder-mixed-with-escaped",
        ),
    ],
)
def test_prompt_init_error_cases(prompt_name, template, expected_exception, expected_msg_substring):
    # Act / Assert
    with pytest.raises(expected_exception) as exc_info:
        Prompt(prompt_name=prompt_name, template=template)

    # Assert
    assert expected_msg_substring in str(exc_info.value)


@pytest.mark.parametrize(
    "initial_context, name, func, expected_value, expected_exception, expected_msg_substring, case_id",
    [
        (
            {},
            "const_str",
            "constant",
            "constant",
            None,
            None,
            "add-context-from-string-creates-wrapper",
        ),
        (
            {},
            "callable_str",
            lambda prompt_name: f"hello-{prompt_name}",
            "hello-my_prompt",
            None,
            None,
            "add-context-from-callable",
        ),
        (
            {"dup": lambda _: "x"},
            "dup",
            "y",
            None,
            KeyError,
            "Context function name 'dup' 已存在于 Prompt 'my_prompt' 中",
            "add-context-duplicate-key-error",
        ),
    ],
)
def test_prompt_add_context(
    initial_context,
    name,
    func,
    expected_value,
    expected_exception,
    expected_msg_substring,
    case_id,
):
    # Arrange
    prompt = Prompt(prompt_name="my_prompt", template="template")
    prompt.prompt_render_context = dict(initial_context)

    # Act
    if expected_exception:
        with pytest.raises(expected_exception) as exc_info:
            prompt.add_context(name, func)

        # Assert
        assert expected_msg_substring in str(exc_info.value)
    else:
        prompt.add_context(name, func)

        # Assert
        assert name in prompt.prompt_render_context
        result = prompt.prompt_render_context[name]("my_prompt")
        assert result == expected_value


def test_prompt_manager_add_prompt_happy_and_error():
    # Arrange
    manager = PromptManager()
    prompt1 = Prompt(prompt_name="p1", template="T1")
    manager.add_prompt(prompt1, need_save=True)

    # Act
    prompt2 = Prompt(prompt_name="p2", template="T2")
    manager.add_prompt(prompt2, need_save=False)

    # Assert
    assert "p1" in manager._prompt_to_save
    assert "p2" not in manager._prompt_to_save

    # Arrange
    prompt_dup = Prompt(prompt_name="p1", template="T-dup")

    # Act / Assert
    with pytest.raises(KeyError) as exc_info:
        manager.add_prompt(prompt_dup)

    # Assert
    assert "Prompt name 'p1' 已存在" in str(exc_info.value)

def test_prompt_manager_get_prompt_is_copy():
    # Arrange
    manager = PromptManager()
    prompt = Prompt(prompt_name="original", template="T")
    manager.add_prompt(prompt)

    # Act
    retrieved_prompt = manager.get_prompt("original")

    # Assert
    assert retrieved_prompt is not prompt
    assert retrieved_prompt.prompt_name == prompt.prompt_name
    assert retrieved_prompt.template == prompt.template
    assert retrieved_prompt.prompt_render_context == prompt.prompt_render_context

def test_prompt_manager_add_prompt_conflict_with_context_name():
    # Arrange
    manager = PromptManager()
    manager.add_context_construct_function("ctx_name", lambda _: "value")
    prompt_conflict = Prompt(prompt_name="ctx_name", template="T")

    # Act / Assert
    with pytest.raises(KeyError) as exc_info:
        manager.add_prompt(prompt_conflict)

    # Assert
    assert "Prompt name 'ctx_name' 已存在" in str(exc_info.value)


def test_prompt_manager_add_context_construct_function_happy():
    # Arrange
    manager = PromptManager()

    def ctx_func(prompt_name: str) -> str:
        return f"ctx-{prompt_name}"

    # Act
    manager.add_context_construct_function("ctx", ctx_func)

    # Assert
    assert "ctx" in manager._context_construct_functions
    stored_func, module = manager._context_construct_functions["ctx"]
    assert stored_func is ctx_func
    assert module == __name__


def test_prompt_manager_add_context_construct_function_duplicate():
    # Arrange
    manager = PromptManager()

    def f(_):
        return "x"

    manager.add_context_construct_function("dup", f)
    manager.add_prompt(Prompt(prompt_name="dup_prompt", template="T"))

    # Act / Assert
    with pytest.raises(KeyError) as exc_info1:
        manager.add_context_construct_function("dup", f)

    # Assert
    assert "Construct function name 'dup' 已存在" in str(exc_info1.value)

    # Act / Assert
    with pytest.raises(KeyError) as exc_info2:
        manager.add_context_construct_function("dup_prompt", f)

    # Assert
    assert "Construct function name 'dup_prompt' 已存在" in str(exc_info2.value)


def test_prompt_manager_get_prompt_not_exist():
    # Arrange
    manager = PromptManager()

    # Act / Assert
    with pytest.raises(KeyError) as exc_info:
        manager.get_prompt("no_such_prompt")

    # Assert
    assert "Prompt name 'no_such_prompt' 不存在" in str(exc_info.value)


@pytest.mark.parametrize(
    "template, inner_context, global_context, expected, case_id",
    [
        pytest.param(
            "Hello {name}",
            {"name": lambda p: f"name-for-{p}"},
            {},
            "Hello name-for-main",
            "render-with-inner-context",
        ),
        pytest.param(
            "Global {block}",
            {},
            {"block": lambda p: f"block-{p}"},
            "Global block-main",
            "render-with-global-context",
        ),
        pytest.param(
            "Mix {inner} and {global}",
            {"inner": lambda p: f"inner-{p}"},
            {"global": lambda p: f"global-{p}"},
            "Mix inner-main and global-main",
            "render-with-inner-and-global-context",
        ),
        pytest.param(
            "Escaped {{ and }} and {field}",
            {"field": lambda _: "X"},
            {},
            "Escaped { and } and X",
            "render-with-escaped-braces",
        ),
    ],
)
@pytest.mark.asyncio
async def test_prompt_manager_render_contexts(template, inner_context, global_context, expected, case_id):
    # Arrange
    manager = PromptManager()
    tmp_prompt = Prompt(prompt_name="main", template=template)
    manager.add_prompt(tmp_prompt)
    prompt = manager.get_prompt("main")
    for name, fn in inner_context.items():
        prompt.add_context(name, fn)
    for name, fn in global_context.items():
        manager.add_context_construct_function(name, fn)
    

    # Act
    rendered = await manager.render_prompt(prompt)

    # Assert
    assert rendered == expected


@pytest.mark.asyncio
async def test_prompt_manager_render_nested_prompts():
    # Arrange
    manager = PromptManager()
    p1 = Prompt(prompt_name="p1", template="P1-{x}")
    p2 = Prompt(prompt_name="p2", template="P2-{p1}")
    p3_tmp = Prompt(prompt_name="p3", template="{p2}-end")
    manager.add_prompt(p1)
    manager.add_prompt(p2)
    manager.add_prompt(p3_tmp)
    p3 = manager.get_prompt("p3")
    p3.add_context("x", lambda _: "X")

    # Act
    rendered = await manager.render_prompt(p3)

    # Assert
    assert rendered == "P2-P1-X-end"


@pytest.mark.asyncio
async def test_prompt_manager_render_recursive_limit():
    # Arrange
    manager = PromptManager()
    p1_tmp = Prompt(prompt_name="p1", template="{p2}")
    p2_tmp = Prompt(prompt_name="p2", template="{p1}")
    manager.add_prompt(p1_tmp)
    manager.add_prompt(p2_tmp)
    p1 = manager.get_prompt("p1")

    # Act / Assert
    with pytest.raises(RecursionError) as exc_info:
        await manager.render_prompt(p1)

    # Assert
    assert "递归层级过深" in str(exc_info.value)


@pytest.mark.asyncio
async def test_prompt_manager_render_missing_field_error():
    # Arrange
    manager = PromptManager()
    tmp_prompt = Prompt(prompt_name="main", template="Hello {missing}")
    manager.add_prompt(tmp_prompt)
    prompt = manager.get_prompt("main")

    # Act / Assert
    with pytest.raises(KeyError) as exc_info:
        await manager.render_prompt(prompt)

    # Assert
    assert "Prompt 'main' 中缺少必要的内容块或构建函数: 'missing'" in str(exc_info.value)


@pytest.mark.asyncio
async def test_prompt_manager_render_prefers_inner_context_over_global():
    # Arrange
    manager = PromptManager()
    tmp_prompt = Prompt(prompt_name="main", template="{field}")
    manager.add_context_construct_function("field", lambda _: "global")
    manager.add_prompt(tmp_prompt)
    prompt = manager.get_prompt("main")
    prompt.add_context("field", lambda _: "inner")

    # Act
    rendered = await manager.render_prompt(prompt)

    # Assert
    assert rendered == "inner"


@pytest.mark.asyncio
async def test_prompt_manager_render_with_coroutine_context_function():
    # Arrange
    manager = PromptManager()

    async def async_inner(prompt_name: str) -> str:
        await asyncio.sleep(0)
        return f"async-{prompt_name}"

    tmp_prompt = Prompt(prompt_name="main", template="{inner}")
    manager.add_prompt(tmp_prompt)
    prompt = manager.get_prompt("main")
    prompt.add_context("inner", async_inner)

    # Act
    rendered = await manager.render_prompt(prompt)

    # Assert
    assert rendered == "async-main"


@pytest.mark.asyncio
async def test_prompt_manager_render_with_coroutine_global_context_function():
    # Arrange
    manager = PromptManager()

    async def async_global(prompt_name: str) -> str:
        await asyncio.sleep(0)
        return f"g-{prompt_name}"

    tmp_prompt = Prompt(prompt_name="main", template="{g}")
    manager.add_context_construct_function("g", async_global)
    manager.add_prompt(tmp_prompt)
    prompt = manager.get_prompt("main")

    # Act
    rendered = await manager.render_prompt(prompt)

    # Assert
    assert rendered == "g-main"


@pytest.mark.parametrize(
    "is_prompt_context, use_coroutine, case_id",
    [
        pytest.param(True, False, "prompt-context-sync-error"),
        pytest.param(False, False, "global-context-sync-error"),
        pytest.param(True, True, "prompt-context-async-error"),
        pytest.param(False, True, "global-context-async-error"),
    ],
)
@pytest.mark.asyncio
async def test_prompt_manager_get_function_result_error_logging(monkeypatch, is_prompt_context, use_coroutine, case_id):
    # Arrange
    manager = PromptManager()

    class DummyError(Exception):
        pass

    def sync_func(_name: str) -> str:
        raise DummyError("sync-error")

    async def async_func(_name: str) -> str:
        await asyncio.sleep(0)
        raise DummyError("async-error")

    func = async_func if use_coroutine else sync_func
    logged_messages: list[str] = []

    def fake_error(msg: Any) -> None:
        logged_messages.append(str(msg))

    fake_logger = type("FakeLogger", (), {"error": staticmethod(fake_error)})

    monkeypatch.setattr("src.prompt.prompt_manager.logger", fake_logger)

    # Act / Assert
    with pytest.raises(DummyError):
        await manager._get_function_result(
            func=func,
            prompt_name="P",
            field_name="field",
            is_prompt_context=is_prompt_context,
            module="mod",
        )

    # Assert
    assert logged_messages
    log = logged_messages[0]
    if is_prompt_context:
        assert "调用 Prompt 'P' 内部上下文构造函数 'field' 时出错" in log
    else:
        assert "调用上下文构造函数 'field' 时出错，所属模块: 'mod'" in log


def test_prompt_manager_add_context_construct_function_unknown_frame(monkeypatch):
    # Arrange
    manager = PromptManager()

    def fake_currentframe() -> None:
        return None

    monkeypatch.setattr("inspect.currentframe", fake_currentframe)

    def f(_):
        return "x"

    # Act / Assert
    with pytest.raises(RuntimeError) as exc_info:
        manager.add_context_construct_function("x", f)

    # Assert
    assert "无法获取调用栈" in str(exc_info.value)


def test_prompt_manager_add_context_construct_function_unknown_caller_frame(monkeypatch):
    # Arrange
    manager = PromptManager()
    real_currentframe = inspect.currentframe

    class FakeFrame:
        f_back = None

    def fake_currentframe():
        return FakeFrame()

    monkeypatch.setattr("inspect.currentframe", fake_currentframe)

    def f(_):
        return "x"

    # Act / Assert
    with pytest.raises(RuntimeError) as exc_info:
        manager.add_context_construct_function("x", f)

    # Assert
    assert "无法获取调用栈的上一级" in str(exc_info.value)

    # Cleanup
    monkeypatch.setattr("inspect.currentframe", real_currentframe)


def test_prompt_manager_save_and_load_prompts(tmp_path, monkeypatch):
    # Arrange
    test_dir = tmp_path / "prompts_dir"
    test_dir.mkdir()

    monkeypatch.setattr("src.prompt.prompt_manager.PROMPTS_DIR", test_dir, raising=False)

    manager = PromptManager()
    p1 = Prompt(prompt_name="save_me", template="Template {x}")
    p1.add_context("x", "X")
    manager.add_prompt(p1, need_save=True)

    # Act
    manager.save_prompts()

    # Assert
    saved_file = test_dir / f"save_me{SUFFIX_PROMPT}"
    assert saved_file.exists()
    assert saved_file.read_text(encoding="utf-8") == "Template {x}"

    # Arrange
    new_manager = PromptManager()

    # Act
    new_manager.load_prompts()

    # Assert
    loaded = new_manager.get_prompt("save_me")
    assert loaded.template == "Template {x}"
    assert "save_me" in new_manager._prompt_to_save


def test_prompt_manager_save_prompts_io_error(tmp_path, monkeypatch):
    # Arrange
    test_dir = tmp_path / "prompts_dir"
    test_dir.mkdir()
    monkeypatch.setattr("src.prompt.prompt_manager.PROMPTS_DIR", test_dir, raising=False)
    manager = PromptManager()
    p1 = Prompt(prompt_name="save_error", template="T")
    manager.add_prompt(p1, need_save=True)

    class FakeFile:
        def __enter__(self):
            raise OSError("disk error")

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_open(*_args, **_kwargs):
        return FakeFile()

    monkeypatch.setattr("builtins.open", fake_open)

    # Act / Assert
    with pytest.raises(OSError) as exc_info:
        manager.save_prompts()

    # Assert
    assert "disk error" in str(exc_info.value)


def test_prompt_manager_load_prompts_io_error(tmp_path, monkeypatch):
    # Arrange
    test_dir = tmp_path / "prompts_dir"
    test_dir.mkdir()
    monkeypatch.setattr("src.prompt.prompt_manager.PROMPTS_DIR", test_dir, raising=False)
    prompt_file = test_dir / f"bad{SUFFIX_PROMPT}"
    prompt_file.write_text("content", encoding="utf-8")

    class FakeFile:
        def __enter__(self):
            raise OSError("read error")

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_open(*_args, **_kwargs):
        return FakeFile()

    monkeypatch.setattr("builtins.open", fake_open)
    manager = PromptManager()

    # Act / Assert
    with pytest.raises(OSError) as exc_info:
        manager.load_prompts()

    # Assert
    assert "read error" in str(exc_info.value)


def test_prompt_manager_global_instance_access():
    # Act
    pm = prompt_manager

    # Assert
    assert isinstance(pm, PromptManager)


def test_formatter_parsing_named_fields_only():
    # Arrange
    manager = PromptManager()
    prompt = Prompt(prompt_name="main", template="A {x} B {y} C")
    manager.add_prompt(prompt)

    # Act
    fields = {field_name for _, field_name, _, _ in manager._formatter.parse(prompt.template) if field_name}

    # Assert
    assert fields == {"x", "y"}
