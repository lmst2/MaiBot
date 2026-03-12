from __future__ import annotations

from pathlib import Path
from string import Formatter

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

FORMATTER = Formatter()


def extract_placeholders(template: str) -> set[str]:
    placeholders: set[str] = set()
    for _, field_name, _, _ in FORMATTER.parse(template):
        if not field_name:
            continue
        placeholders.add(field_name.split(".", maxsplit=1)[0].split("[", maxsplit=1)[0])
    return placeholders


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


def validate_locales(locales_root: Path | None = None) -> list[str]:
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
            validate_translation_pair(key, source_catalog[key], catalog[key], locale, errors)

    return errors


def main() -> int:
    errors = validate_locales()
    if errors:
        print("i18n validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("i18n validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
