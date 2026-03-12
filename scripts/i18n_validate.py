from __future__ import annotations

from pathlib import Path

import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.i18n.loaders import (  # noqa: E402
    DEFAULT_LOCALE,
    TranslationValue,
    discover_locales,
    get_locales_root,
    load_locale_catalog,
)
from src.common.i18n.loaders import extract_placeholders  # noqa: E402
from src.common.prompt_i18n import (  # noqa: E402
    PROMPT_EXTENSIONS,
    extract_prompt_placeholders,
    get_prompts_root,
)

HAN_CHARACTER_PATTERN = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")


def contains_han_characters(text: str) -> bool:
    return HAN_CHARACTER_PATTERN.search(text) is not None


def iter_translation_strings(value: TranslationValue) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [value[category] for category in sorted(value.keys())]


def locale_requires_latin_only_validation(locale: str) -> bool:
    normalized_locale = locale.lower()
    return normalized_locale == "en" or normalized_locale.startswith("en-")


def validate_locale_content(
    key: str,
    source_value: TranslationValue,
    target_value: TranslationValue,
    locale: str,
    errors: list[str],
) -> None:
    source_texts = iter_translation_strings(source_value)
    target_texts = iter_translation_strings(target_value)

    if any(
        source_text == target_text and contains_han_characters(source_text)
        for source_text, target_text in zip(source_texts, target_texts, strict=False)
    ):
        errors.append(f"[{locale}] key '{key}' 直接保留了包含中文字符的 source 文案（仓库级校验策略），请提供目标语言翻译")

    if locale_requires_latin_only_validation(locale) and any(contains_han_characters(text) for text in target_texts):
        errors.append(f"[{locale}] key '{key}' 仍包含中文字符，请移除源语言残留后再提交")


def validate_translation_pair(
    key: str,
    source_value: TranslationValue,
    target_value: TranslationValue,
    locale: str,
    errors: list[str],
) -> None:
    if isinstance(source_value, str):
        if not isinstance(target_value, str):
            errors.append(f"[{locale}] key '{key}' 与 source 的类型不一致：source=string, target=plural")
            return
        if extract_placeholders(source_value) != extract_placeholders(target_value):
            errors.append(f"[{locale}] key '{key}' 的占位符集合与 source 不一致")
        return

    if not isinstance(target_value, dict):
        errors.append(f"[{locale}] key '{key}' 与 source 的类型不一致：source=plural, target=string")
        return

    source_categories = set(source_value.keys())
    target_categories = set(target_value.keys())
    if source_categories != target_categories:
        errors.append(
            f"[{locale}] key '{key}' 的 plural category 不一致："
            f"source={sorted(source_categories)}, target={sorted(target_categories)}"
        )

    for category in sorted(source_categories & target_categories):
        source_placeholders = extract_placeholders(source_value[category])
        target_placeholders = extract_placeholders(target_value[category])
        if source_placeholders != target_placeholders:
            errors.append(f"[{locale}] key '{key}' 的 plural category '{category}' 占位符集合与 source 不一致")


def validate_json_locales(locales_root: Path | None = None) -> list[str]:
    resolved_locales_root = get_locales_root(locales_root)
    locales = discover_locales(resolved_locales_root)
    errors: list[str] = []

    if DEFAULT_LOCALE not in locales:
        errors.append(f"缺少默认 locale 目录: {DEFAULT_LOCALE}")
        return errors

    catalogs: dict[str, dict[str, TranslationValue]] = {}
    for locale in locales:
        try:
            catalogs[locale] = load_locale_catalog(locale, resolved_locales_root)
        except Exception as exc:
            errors.append(f"[{locale}] 加载失败: {exc}")

    source_catalog = catalogs.get(DEFAULT_LOCALE)
    if source_catalog is None:
        return errors

    source_keys = set(source_catalog.keys())
    for locale, catalog in catalogs.items():
        if locale == DEFAULT_LOCALE:
            continue

        locale_keys = set(catalog.keys())
        missing_keys = sorted(source_keys - locale_keys)
        extra_keys = sorted(locale_keys - source_keys)

        for key in missing_keys:
            errors.append(f"[{locale}] 缺少 key: {key}")
        for key in extra_keys:
            errors.append(f"[{locale}] 存在多余 key: {key}")

        for key in sorted(source_keys & locale_keys):
            source_value = source_catalog[key]
            target_value = catalog[key]
            validate_translation_pair(key, source_value, target_value, locale, errors)
            if isinstance(source_value, str) == isinstance(target_value, str):
                validate_locale_content(key, source_value, target_value, locale, errors)

    return errors


def discover_prompt_locales(prompts_root: Path | None = None) -> list[str]:
    resolved_prompts_root = get_prompts_root(prompts_root)
    if not resolved_prompts_root.exists():
        return []

    locale_names = [path.name for path in resolved_prompts_root.iterdir() if path.is_dir()]
    return sorted(locale_names)


def iter_prompt_files(locale_dir: Path) -> list[Path]:
    prompt_files: list[Path] = []
    for extension in PROMPT_EXTENSIONS:
        prompt_files.extend(path for path in locale_dir.rglob(f"*{extension}") if path.is_file())
    return sorted(set(prompt_files))


def validate_prompt_templates(prompts_root: Path | None = None) -> tuple[list[str], list[str]]:
    resolved_prompts_root = get_prompts_root(prompts_root)
    prompt_locales = discover_prompt_locales(resolved_prompts_root)
    known_locales = [locale for locale in discover_locales(get_locales_root()) if locale != DEFAULT_LOCALE]
    errors: list[str] = []
    warnings: list[str] = []

    if DEFAULT_LOCALE not in prompt_locales:
        errors.append(f"缺少默认 Prompt locale 目录: {DEFAULT_LOCALE}")
        return errors, warnings

    source_dir = resolved_prompts_root / DEFAULT_LOCALE
    source_files = {path.relative_to(source_dir): path for path in iter_prompt_files(source_dir)}

    for locale in known_locales:
        locale_dir = resolved_prompts_root / locale
        if not locale_dir.exists():
            warnings.append(f"[prompt:{locale}] 缺少 locale 目录，运行时将回退到 {DEFAULT_LOCALE}")
            continue

        locale_files = {path.relative_to(locale_dir): path for path in iter_prompt_files(locale_dir)}
        source_relative_paths = set(source_files.keys())
        locale_relative_paths = set(locale_files.keys())

        for relative_path in sorted(source_relative_paths - locale_relative_paths):
            warnings.append(f"[prompt:{locale}] 缺少模板: {relative_path.as_posix()}，运行时将回退到 {DEFAULT_LOCALE}")

        for relative_path in sorted(locale_relative_paths - source_relative_paths):
            warnings.append(f"[prompt:{locale}] 存在额外模板: {relative_path.as_posix()}")

        for relative_path in sorted(source_relative_paths & locale_relative_paths):
            source_text = source_files[relative_path].read_text(encoding="utf-8")
            locale_text = locale_files[relative_path].read_text(encoding="utf-8")

            source_placeholders = extract_prompt_placeholders(source_text)
            locale_placeholders = extract_prompt_placeholders(locale_text)
            if source_placeholders != locale_placeholders:
                errors.append(
                    "[prompt:{locale}] 模板 '{path}' 的占位符集合与 source 不一致："
                    "source={source_placeholders}, target={target_placeholders}".format(
                        locale=locale,
                        path=relative_path.as_posix(),
                        source_placeholders=sorted(source_placeholders),
                        target_placeholders=sorted(locale_placeholders),
                    )
                )

            if source_text == locale_text:
                warnings.append(f"[prompt:{locale}] 模板 '{relative_path.as_posix()}' 与 source 完全相同，可能尚未翻译")

    return errors, warnings


def _print_warnings(warnings: list[str]) -> None:
    if not warnings:
        return
    print(f"warnings ({len(warnings)}):")
    for warning in warnings[:10]:
        print(f"  - {warning}")
    if len(warnings) > 10:
        print(f"  - ... 另外还有 {len(warnings) - 10} 条 warning")


def main() -> int:
    errors = validate_json_locales()
    prompt_errors, prompt_warnings = validate_prompt_templates()
    errors.extend(prompt_errors)

    if errors:
        print("i18n validation failed:")
        for error in errors:
            print(f"  - {error}")
        _print_warnings(prompt_warnings)
        return 1

    print("i18n validation passed.")
    _print_warnings(prompt_warnings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
