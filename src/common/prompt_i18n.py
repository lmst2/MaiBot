from __future__ import annotations

from pathlib import Path
from string import Formatter

import logging
import os
import re
import threading

from .i18n import get_locale, t
from .i18n.loaders import DEFAULT_LOCALE, normalize_locale

logger = logging.getLogger("maibot.prompt_i18n")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_ROOT = (PROJECT_ROOT / "prompts").resolve()
PROMPT_EXTENSIONS = (".prompt")
FORMATTER = Formatter()
SAFE_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
STRICT_ENV_KEYS = ("MAIBOT_PROMPT_I18N_STRICT", "MAIBOT_I18N_STRICT")

_prompt_cache: dict[Path, str] = {}
_cache_lock = threading.RLock()


def extract_prompt_placeholders(template: str) -> set[str]:
    placeholders: set[str] = set()
    for _, field_name, _, _ in FORMATTER.parse(template):
        if not field_name:
            continue
        placeholders.add(field_name.split(".", maxsplit=1)[0].split("[", maxsplit=1)[0])
    return placeholders


def get_prompts_root(prompts_root: Path | None = None) -> Path:
    return (prompts_root or PROMPTS_ROOT).resolve()


def normalize_prompt_name(name: str) -> str:
    candidate_name = name.strip()
    for suffix in PROMPT_EXTENSIONS:
        if candidate_name.endswith(suffix):
            candidate_name = candidate_name[: -len(suffix)]
            break

    if candidate_name in {".", ".."} or not candidate_name or not SAFE_SEGMENT_PATTERN.fullmatch(candidate_name):
        raise ValueError(t("prompt.invalid_name", name=name))
    return candidate_name


def normalize_prompt_category(category: str | None) -> str | None:
    if category is None:
        return None

    category_parts = [part for part in category.strip().split("/") if part]
    if not category_parts:
        raise ValueError(t("prompt.invalid_category", category=category))

    for part in category_parts:
        if part in {".", ".."} or not SAFE_SEGMENT_PATTERN.fullmatch(part):
            raise ValueError(t("prompt.invalid_category", category=category))
    return "/".join(category_parts)


def is_strict_prompt_i18n_mode() -> bool:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return True

    return any(os.getenv(env_key, "").strip().lower() in {"1", "true", "yes", "on"} for env_key in STRICT_ENV_KEYS)


def _supported_prompt_files(directory: Path) -> list[Path]:
    matched_files: list[Path] = []
    for suffix in PROMPT_EXTENSIONS:
        matched_files.extend(path for path in directory.rglob(f"*{suffix}") if path.is_file())
    return sorted(set(matched_files))


def _supported_prompt_files_non_recursive(directory: Path) -> list[Path]:
    matched_files: list[Path] = []
    for suffix in PROMPT_EXTENSIONS:
        matched_files.extend(path for path in directory.glob(f"*{suffix}") if path.is_file())
    return sorted(set(matched_files))


def _scan_prompt_directory(directory: Path, prompts_root: Path) -> dict[str, Path]:
    prompt_paths: dict[str, Path] = {}
    if not directory.exists():
        return prompt_paths

    for prompt_path in _supported_prompt_files(directory):
        prompt_name = prompt_path.stem
        if prompt_name in prompt_paths:
            raise ValueError(
                t(
                    "prompt.duplicate_template_name",
                    name=prompt_name,
                    path_a=prompt_paths[prompt_name].relative_to(prompts_root),
                    path_b=prompt_path.relative_to(prompts_root),
                )
            )
        prompt_paths[prompt_name] = prompt_path
    return prompt_paths


def _scan_legacy_prompt_directory(directory: Path) -> dict[str, Path]:
    prompt_paths: dict[str, Path] = {}
    if not directory.exists():
        return prompt_paths

    for prompt_path in _supported_prompt_files_non_recursive(directory):
        prompt_name = prompt_path.stem
        if prompt_name in prompt_paths:
            raise ValueError(
                t(
                    "prompt.duplicate_template_name",
                    name=prompt_name,
                    path_a=prompt_paths[prompt_name].relative_to(get_prompts_root(directory)),
                    path_b=prompt_path.relative_to(get_prompts_root(directory)),
                )
            )
        prompt_paths[prompt_name] = prompt_path
    return prompt_paths


def list_prompt_templates(locale: str | None = None, prompts_root: Path | None = None) -> dict[str, Path]:
    resolved_prompts_root = get_prompts_root(prompts_root)
    requested_locale = normalize_locale(locale or get_locale())

    prompt_paths = _scan_legacy_prompt_directory(resolved_prompts_root)
    prompt_paths.update(_scan_prompt_directory(resolved_prompts_root / DEFAULT_LOCALE, resolved_prompts_root))

    if requested_locale != DEFAULT_LOCALE:
        prompt_paths.update(_scan_prompt_directory(resolved_prompts_root / requested_locale, resolved_prompts_root))

    return prompt_paths


def resolve_prompt_path(name: str, locale: str | None = None, category: str | None = None, prompts_root: Path | None = None) -> Path:
    resolved_prompts_root = get_prompts_root(prompts_root)
    normalized_name = normalize_prompt_name(name)
    normalized_category = normalize_prompt_category(category)
    requested_locale = normalize_locale(locale or get_locale())

    locale_candidates: list[str | None] = [requested_locale]
    if requested_locale != DEFAULT_LOCALE:
        locale_candidates.append(DEFAULT_LOCALE)
    locale_candidates.append(None)

    if normalized_category is not None:
        for locale_candidate in locale_candidates:
            base_dir = resolved_prompts_root if locale_candidate is None else resolved_prompts_root / locale_candidate
            for suffix in PROMPT_EXTENSIONS:
                candidate_paths = [(base_dir / normalized_category / f"{normalized_name}{suffix}").resolve()]
                # 允许带 category 的调用在旧版平铺目录或未迁移完的 locale 目录中继续工作。
                candidate_paths.append((base_dir / f"{normalized_name}{suffix}").resolve())
                for candidate_path in candidate_paths:
                    if candidate_path.is_file():
                        return candidate_path
    else:
        prompt_paths = list_prompt_templates(locale=requested_locale, prompts_root=resolved_prompts_root)
        if normalized_name in prompt_paths:
            return prompt_paths[normalized_name]

    raise FileNotFoundError(t("prompt.template_not_found", locale=requested_locale, name=normalized_name))


def load_prompt(
    name: str,
    locale: str | None = None,
    category: str | None = None,
    prompts_root: Path | None = None,
    **kwargs: object,
) -> str:
    prompt_path = resolve_prompt_path(name=name, locale=locale, category=category, prompts_root=prompts_root)
    with _cache_lock:
        template = _prompt_cache.get(prompt_path)
        if template is None:
            with open(prompt_path, "r", encoding="utf-8") as prompt_file:
                template = prompt_file.read()
            _prompt_cache[prompt_path] = template

    if not kwargs:
        return template

    try:
        return template.format(**kwargs)
    except KeyError as exc:
        missing_placeholder = exc.args[0]
        error = KeyError(
            t(
                "prompt.missing_placeholder",
                name=normalize_prompt_name(name),
                placeholder=missing_placeholder,
            )
        )
        if is_strict_prompt_i18n_mode():
            raise error from exc
        logger.error("%s", error)
        return template
    except Exception as exc:
        logger.error(t("prompt.format_failed", name=normalize_prompt_name(name), error=exc))
        if is_strict_prompt_i18n_mode():
            raise
        return template


def clear_prompt_cache() -> None:
    with _cache_lock:
        _prompt_cache.clear()
