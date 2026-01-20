import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import pytest

PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.absolute().resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "config"))

from src.prompt.prompt_manager import PromptManager

# --- Minimal stubs / constants matching the production module ---

# These imports/definitions are here only to make the tests self‑contained
# In the real project, they already exist in `prompt_manager.py`'s module.
# We mirror them here to control behavior via monkeypatch.


class Prompt:
    def __init__(self, prompt_name: str, template: str, prompt_render_context: Optional[dict[str, Callable]] = None):
        self.prompt_name = prompt_name
        self.template = template
        self.prompt_render_context = prompt_render_context or {}


class DummyLogger:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)


# --- Fixtures to patch module-level objects in prompt_manager ---


@pytest.fixture
def dummy_logger(monkeypatch):
    from src.prompt import prompt_manager as pm

    logger = DummyLogger()
    monkeypatch.setattr(pm, "logger", logger, raising=False)
    return logger


@pytest.fixture
def temp_prompts_dir(tmp_path, monkeypatch):
    from src.prompt import prompt_manager as pm

    prompts_dir = tmp_path / "prompts"
    monkeypatch.setattr(pm, "PROMPTS_DIR", prompts_dir, raising=False)
    return prompts_dir


@pytest.fixture
def brace_constants(monkeypatch):
    from src.prompt import prompt_manager as pm

    # emulate the placeholders used in the manager
    monkeypatch.setattr(pm, "_LEFT_BRACE", "__LEFT__", raising=False)
    monkeypatch.setattr(pm, "_RIGHT_BRACE", "__RIGHT__", raising=False)


@pytest.fixture
def suffix_prompt(monkeypatch):
    from src.prompt import prompt_manager as pm

    monkeypatch.setattr(pm, "SUFFIX_PROMPT", ".prompt", raising=False)


@pytest.fixture
def manager(temp_prompts_dir, brace_constants, suffix_prompt):
    # PromptManager.__init__ uses patched PROMPTS_DIR
    return PromptManager()


# --- Helper to run async methods in tests (for non-async pytest) ---


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# --- add_prompt tests --------------------------------------------------------


@pytest.mark.parametrize(
    "existing_prompts, existing_funcs, name_to_add, need_save, expect_in_save",
    [
        pytest.param(
            {},
            {},
            "greeting",
            False,
            False,
            id="add_prompt_simple_not_saved",
        ),
        pytest.param(
            {},
            {},
            "system",
            True,
            True,
            id="add_prompt_marked_for_save",
        ),
        pytest.param(
            {"existing": Prompt("existing", "tmpl")},
            {},
            "new",
            True,
            True,
            id="add_prompt_with_existing_other_prompt",
        ),
    ],
)
def test_add_prompt_happy_path(manager, existing_prompts, existing_funcs, name_to_add, need_save, expect_in_save):
    # Arrange

    manager.prompts.update(existing_prompts)
    manager._context_construct_functions.update(existing_funcs)
    prompt = Prompt(name_to_add, "template")

    # Act

    manager.add_prompt(prompt, need_save=need_save)

    # Assert

    assert manager.prompts[name_to_add] is prompt
    assert (name_to_add in manager._prompt_to_save) is expect_in_save


@pytest.mark.parametrize(
    "existing_prompts, existing_funcs, new_name, conflict_type",
    [
        pytest.param(
            {"dup": Prompt("dup", "tmpl")},
            {},
            "dup",
            "prompt_conflict",
            id="add_prompt_conflict_with_existing_prompt",
        ),
        pytest.param(
            {},
            {"dup": (lambda x: x, "mod")},
            "dup",
            "func_conflict",
            id="add_prompt_conflict_with_existing_context_function",
        ),
    ],
)
def test_add_prompt_conflict_raises_key_error(manager, existing_prompts, existing_funcs, new_name, conflict_type):
    # Arrange

    manager.prompts.update(existing_prompts)
    manager._context_construct_functions.update(existing_funcs)
    prompt = Prompt(new_name, "template")

    # Act / Assert

    with pytest.raises(KeyError) as exc:
        manager.add_prompt(prompt)

    assert new_name in str(exc.value)


# --- add_context_construct_function tests -----------------------------------


def test_add_context_construct_function_happy_path(manager):
    # Arrange

    def builder(prompt_name: str) -> str:
        return f"ctx_for_{prompt_name}"

    # Act

    manager.add_context_construct_function("ctx", builder)

    # Assert

    assert "ctx" in manager._context_construct_functions
    stored_func, module = manager._context_construct_functions["ctx"]
    assert stored_func is builder
    # module is caller's module name
    assert isinstance(module, str)
    assert module != ""


def test_add_context_construct_function_logs_unknown_module(manager, dummy_logger, monkeypatch):
    # Arrange

    def builder(prompt_name: str) -> str:
        return f"v_{prompt_name}"

    def fake_currentframe():
        class FakeCallerFrame:
            f_globals = {"__name__": "unknown"}

        class FakeFrame:
            f_back = FakeCallerFrame()

        return FakeFrame()

    from src.prompt import prompt_manager as pm

    monkeypatch.setattr(pm.inspect, "currentframe", fake_currentframe)

    # Act
    manager.add_context_construct_function("unknown_ctx", builder)

    # Assert

    assert any("无法获取调用函数的模块名" in msg for msg in dummy_logger.warnings)
    assert "unknown_ctx" in manager._context_construct_functions


@pytest.mark.parametrize(
    "existing_prompts, existing_funcs, name_to_add",
    [
        pytest.param(
            {"p": Prompt("p", "tmpl")},
            {},
            "p",
            id="add_context_construct_function_conflict_with_prompt",
        ),
        pytest.param(
            {},
            {"f": (lambda x: x, "mod")},
            "f",
            id="add_context_construct_function_conflict_with_existing_func",
        ),
    ],
)
def test_add_context_construct_function_conflict_raises_key_error(
    manager, existing_prompts, existing_funcs, name_to_add
):
    # Arrange

    manager.prompts.update(existing_prompts)
    manager._context_construct_functions.update(existing_funcs)

    def func(prompt_name: str) -> str:
        return "x"

    # Act / Assert

    with pytest.raises(KeyError) as exc:
        manager.add_context_construct_function(name_to_add, func)

    assert name_to_add in str(exc.value)


def test_add_context_construct_function_no_frame_raises_runtime_error(manager, monkeypatch):
    # Arrange

    from src.prompt import prompt_manager as pm

    monkeypatch.setattr(pm.inspect, "currentframe", lambda: None)

    def func(prompt_name: str) -> str:
        return "x"

    # Act / Assert

    with pytest.raises(RuntimeError) as exc:
        manager.add_context_construct_function("ctx", func)

    assert "无法获取调用栈" in str(exc.value)


def test_add_context_construct_function_no_caller_frame_raises_runtime_error(manager, monkeypatch):
    # Arrange

    from src.prompt import prompt_manager as pm

    class FakeFrame:
        f_back = None

    monkeypatch.setattr(pm.inspect, "currentframe", lambda: FakeFrame())

    def func(prompt_name: str) -> str:
        return "x"

    # Act / Assert

    with pytest.raises(RuntimeError) as exc:
        manager.add_context_construct_function("ctx", func)

    assert "无法获取调用栈的上一级" in str(exc.value)


# --- get_prompt tests --------------------------------------------------------


@pytest.mark.parametrize(
    "existing_name, requested_name, should_raise",
    [
        pytest.param("p1", "p1", False, id="get_existing_prompt"),
        pytest.param("p1", "missing", True, id="get_missing_prompt_raises"),
    ],
)
def test_get_prompt(manager, existing_name, requested_name, should_raise):
    # Arrange

    manager.prompts[existing_name] = Prompt(existing_name, "tmpl")

    # Act / Assert

    if should_raise:
        with pytest.raises(KeyError) as exc:
            manager.get_prompt(requested_name)
        assert requested_name in str(exc.value)
    else:
        prompt = manager.get_prompt(requested_name)
        assert prompt.prompt_name == existing_name


# --- render_prompt and _render tests ----------------------------------------


@pytest.mark.parametrize(
    "template, prompts_setup, ctx_funcs_setup, prompt_ctx, expected",
    [
        pytest.param(
            "Hello {name}",
            {},
            {},
            {"name": lambda p: "World"},
            "Hello World",
            id="render_with_prompt_context_sync",
        ),
        pytest.param(
            "Hello {name}",
            {},
            {},
            {
                "name": lambda p: asyncio.sleep(0, result=f"Async-{p}"),
            },
            "Hello Async-main",
            id="render_with_prompt_context_async",
        ),
        pytest.param(
            "Outer {inner}",
            {
                "inner": Prompt("inner", "Inner {value}", {"value": lambda p: "42"}),
            },
            {},
            {},
            "Outer Inner 42",
            id="render_with_nested_prompt_reference",
        ),
        pytest.param(
            "Module says {ext}",
            {},
            {
                "ext": (lambda p: f"external-{p}", "test_module"),
            },
            {},
            "Module says external-main",
            id="render_with_external_context_function_sync",
        ),
        pytest.param(
            "Module async {ext}",
            {},
            {
                "ext": (lambda p: asyncio.sleep(0, result=f"ext_async-{p}"), "test_module"),
            },
            {},
            "Module async ext_async-main",
            id="render_with_external_context_function_async",
        ),
        pytest.param(
            "Escaped {{ and }} literal plus {value}",
            {},
            {},
            {"value": lambda p: "X"},
            "Escaped { and } literal plus X",
            id="render_with_escaped_braces",
        ),
    ],
)
def test_render_prompt_happy_path(
    manager,
    template,
    prompts_setup,
    ctx_funcs_setup,
    prompt_ctx,
    expected,
):
    # Arrange

    main_prompt = Prompt("main", template, prompt_ctx)
    manager.add_prompt(main_prompt)
    for name, prompt in prompts_setup.items():
        manager.add_prompt(prompt)
    manager._context_construct_functions.update(ctx_funcs_setup)

    # Act

    rendered = run(manager.render_prompt(main_prompt))

    # Assert

    assert rendered == expected


def test_render_prompt_missing_field_raises_key_error(manager):
    # Arrange

    prompt = Prompt("main", "Hello {missing}")
    manager.add_prompt(prompt)

    # Act / Assert

    with pytest.raises(KeyError) as exc:
        run(manager.render_prompt(prompt))

    assert "缺少必要的内容块或构建函数" in str(exc.value)
    assert "missing" in str(exc.value)


def test_render_prompt_recursion_limit_exceeded(manager):
    # Arrange

    # Create mutual recursion between two prompts
    p1 = Prompt("p1", "P1 uses {p2}")
    p2 = Prompt("p2", "P2 uses {p1}")
    manager.add_prompt(p1)
    manager.add_prompt(p2)

    # Act / Assert

    with pytest.raises(RecursionError):
        run(manager.render_prompt(p1))


# --- _get_function_result tests ---------------------------------------------


@pytest.mark.parametrize(
    "func, is_prompt_context, expect_async",
    [
        pytest.param(
            lambda p: f"sync_{p}",
            True,
            False,
            id="get_function_result_sync_prompt_context",
        ),
        pytest.param(
            lambda p: asyncio.sleep(0, result=f"async_{p}"),
            False,
            True,
            id="get_function_result_async_external_context",
        ),
    ],
)
def test_get_function_result_happy_path(manager, dummy_logger, func, is_prompt_context, expect_async):
    # Act

    res = run(
        manager._get_function_result(
            func=func,
            prompt_name="prompt",
            field_name="f",
            is_prompt_context=is_prompt_context,
            module="mod",
        )
    )

    # Assert

    assert res in {"sync_prompt", "async_prompt"}


@pytest.mark.parametrize(
    "is_prompt_context, expected_message_part",
    [
        pytest.param(True, "内部上下文构造函数", id="get_function_result_error_prompt_context_logs_internal_msg"),
        pytest.param(False, "上下文构造函数", id="get_function_result_error_external_logs_external_msg"),
    ],
)
def test_get_function_result_error_logging(manager, dummy_logger, is_prompt_context, expected_message_part):
    # Arrange

    def bad_func(prompt_name: str) -> str:
        raise ValueError("bad")

    # Act / Assert

    with pytest.raises(ValueError):
        run(
            manager._get_function_result(
                func=bad_func,
                prompt_name="promptX",
                field_name="fieldX",
                is_prompt_context=is_prompt_context,
                module="modX",
            )
        )

    # Assert

    assert any(expected_message_part in msg for msg in dummy_logger.errors)
    assert any("promptX" in msg for msg in dummy_logger.errors) ^ (not is_prompt_context)
    assert any("modX" in msg for msg in dummy_logger.errors) ^ is_prompt_context
    assert any("fieldX" in msg for msg in dummy_logger.errors)


# --- save_prompts tests ------------------------------------------------------


def test_save_prompts_happy_path(manager, temp_prompts_dir):
    # Arrange

    p1 = Prompt("p1", "Hello {{name}}")
    p2 = Prompt("p2", "Bye {{value}}")
    manager.add_prompt(p1, need_save=True)
    manager.add_prompt(p2, need_save=True)

    # Act

    manager.save_prompts()

    # Assert

    files = sorted(temp_prompts_dir.glob("*.prompt"))
    assert len(files) == 2
    contents = {f.stem: f.read_text(encoding="utf-8") for f in files}
    assert contents["p1"] == "Hello {{name}}"
    assert contents["p2"] == "Bye {{value}}"


def test_save_prompts_io_error(manager, temp_prompts_dir, dummy_logger, monkeypatch):
    # Arrange

    prompt = Prompt("p1", "Hi")
    manager.add_prompt(prompt, need_save=True)

    def bad_open(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", bad_open)

    # Act / Assert

    with pytest.raises(OSError):
        manager.save_prompts()

    # Assert

    assert any("保存 Prompt 'p1' 时出错" in msg for msg in dummy_logger.errors)


# --- load_prompts tests ------------------------------------------------------


def test_load_prompts_happy_path(manager, temp_prompts_dir):
    # Arrange

    file1 = temp_prompts_dir / "greet.prompt"
    file2 = temp_prompts_dir / "farewell.prompt"
    temp_prompts_dir.mkdir(parents=True, exist_ok=True)
    file1.write_text("Hello {{name}}", encoding="utf-8")
    file2.write_text("Bye {{name}}", encoding="utf-8")

    # Act

    manager.load_prompts()

    # Assert

    assert "greet" in manager.prompts
    assert "farewell" in manager.prompts
    assert "greet" in manager._prompt_to_save
    assert "farewell" in manager._prompt_to_save
    assert manager.prompts["greet"].template == "Hello {{name}}"
    assert manager.prompts["farewell"].template == "Bye {{name}}"


def test_load_prompts_error(manager, temp_prompts_dir, dummy_logger, monkeypatch):
    # Arrange

    file1 = temp_prompts_dir / "broken.prompt"
    temp_prompts_dir.mkdir(parents=True, exist_ok=True)
    file1.write_text("whatever", encoding="utf-8")

    def bad_open(*args, **kwargs):
        raise OSError("cannot read")

    monkeypatch.setattr("builtins.open", bad_open)

    # Act / Assert

    with pytest.raises(OSError):
        manager.load_prompts()

    # Assert

    assert any("加载 Prompt 文件" in msg for msg in dummy_logger.errors)
