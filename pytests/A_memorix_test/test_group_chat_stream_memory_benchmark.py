from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import sys
import tempfile
import types
import typing
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLUGINS_ROOT = PROJECT_ROOT / "plugins"
SDK_ROOT = PROJECT_ROOT / "packages" / "maibot-plugin-sdk"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PLUGINS_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGINS_ROOT))
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

if "maibot_sdk" not in sys.modules:
    maibot_sdk = types.ModuleType("maibot_sdk")
    maibot_sdk_types = types.ModuleType("maibot_sdk.types")

    class _FakeMaiBotPlugin:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    def _fake_tool(*decorator_args, **decorator_kwargs):
        del decorator_args, decorator_kwargs

        def wrapper(func):
            return func

        return wrapper

    class _FakeToolParameterInfo:
        def __init__(self, name: str, param_type: str, description: str, required: bool) -> None:
            self.name = name
            self.param_type = param_type
            self.description = description
            self.required = required

    class _FakeToolParamType:
        STRING = "string"
        INTEGER = "integer"
        FLOAT = "float"
        BOOLEAN = "boolean"

    maibot_sdk.MaiBotPlugin = _FakeMaiBotPlugin
    maibot_sdk.Tool = _fake_tool
    maibot_sdk_types.ToolParameterInfo = _FakeToolParameterInfo
    maibot_sdk_types.ToolParamType = _FakeToolParamType
    sys.modules["maibot_sdk"] = maibot_sdk
    sys.modules["maibot_sdk.types"] = maibot_sdk_types

try:
    import aiohttp  # type: ignore
except Exception:
    aiohttp = types.ModuleType("aiohttp")

    class _FakeAioHttpClientError(Exception):
        pass

    aiohttp.ClientError = _FakeAioHttpClientError
    sys.modules["aiohttp"] = aiohttp

try:
    import openai  # type: ignore
except Exception:
    openai = types.ModuleType("openai")

    class _FakeOpenAIConnectionError(Exception):
        pass

    class _FakeOpenAITimeoutError(Exception):
        pass

    openai.APIConnectionError = _FakeOpenAIConnectionError
    openai.APITimeoutError = _FakeOpenAITimeoutError
    sys.modules["openai"] = openai

if "eval_type_backport" not in sys.modules:
    eval_type_backport_module = types.ModuleType("eval_type_backport")

    def _rewrite_union(expr: str) -> str:
        clean = str(expr or "").strip()
        if "|" not in clean:
            return clean
        parts = [part.strip() for part in clean.split("|")]
        if len(parts) == 2 and "None" in parts:
            target = parts[0] if parts[1] == "None" else parts[1]
            return f"typing.Optional[{target}]"
        return f"typing.Union[{', '.join(parts)}]"

    def _eval_type_backport(value, globalns=None, localns=None, try_default=False):
        del try_default
        expr = getattr(value, "__forward_arg__", value)
        if not isinstance(expr, str):
            expr = str(expr)
        gns = dict(globalns or {})
        lns = dict(localns or {})
        builtin_aliases = {
            "dict": typing.Dict,
            "list": typing.List,
            "set": typing.Set,
            "tuple": typing.Tuple,
        }
        for namespace in (gns, lns):
            namespace.setdefault("typing", typing)
            namespace.setdefault("Any", typing.Any)
            namespace.setdefault("Optional", typing.Optional)
            namespace.setdefault("Union", typing.Union)
            namespace.setdefault("Literal", getattr(typing, "Literal", None))
            namespace.update({key: value for key, value in builtin_aliases.items() if key not in namespace})

        try:
            return eval(expr, gns, lns)
        except TypeError:
            return eval(_rewrite_union(expr), gns, lns)

    eval_type_backport_module.eval_type_backport = _eval_type_backport
    sys.modules["eval_type_backport"] = eval_type_backport_module

from A_memorix.core.runtime import sdk_memory_kernel as kernel_module
from A_memorix.core.runtime.sdk_memory_kernel import KernelSearchRequest, SDKMemoryKernel
from src.chat.brain_chat.PFC import pfc_KnowledgeFetcher as knowledge_module
from src.memory_system import chat_history_summarizer as summarizer_module
from src.memory_system.retrieval_tools.query_long_term_memory import query_long_term_memory
from src.person_info import person_info as person_info_module
from src.services import memory_service as memory_service_module
from src.services.memory_service import MemorySearchResult, memory_service


def _resolve_benchmark_paths() -> tuple[Path, Path]:
    configured = str(os.environ.get("A_MEMORIX_BENCHMARK_DATA_FILE", "") or "").strip()
    if configured:
        data_file = Path(configured).expanduser()
        if not data_file.is_absolute():
            data_file = (Path.cwd() / data_file).resolve()
    else:
        data_file = Path(__file__).parent / "data" / "benchmarks" / "group_chat_stream_memory_benchmark.json"

    report_file = Path(__file__).parent / "data" / "benchmarks" / "results" / f"{data_file.stem}_report.json"
    return data_file, report_file


DATA_FILE, REPORT_FILE = _resolve_benchmark_paths()


def _load_benchmark_fixture() -> Dict[str, Any]:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


class _PatchManager:
    def __init__(self) -> None:
        self._records: List[tuple[Any, str, Any, bool]] = []

    def setattr(self, target: Any, name: str, value: Any) -> None:
        existed = hasattr(target, name)
        original = getattr(target, name) if existed else None
        self._records.append((target, name, original, existed))
        setattr(target, name, value)

    def undo(self) -> None:
        while self._records:
            target, name, original, existed = self._records.pop()
            if existed:
                setattr(target, name, original)
            else:
                delattr(target, name)


class _FakeEmbeddingAdapter:
    def __init__(self, dimension: int = 32) -> None:
        self.dimension = dimension

    async def _detect_dimension(self) -> int:
        return self.dimension

    async def encode(self, texts, dimensions=None):
        dim = int(dimensions or self.dimension)
        if isinstance(texts, str):
            sequence = [texts]
            single = True
        else:
            sequence = list(texts)
            single = False

        rows = []
        for text in sequence:
            vec = np.zeros(dim, dtype=np.float32)
            for ch in str(text or ""):
                code = ord(ch)
                vec[code % dim] += 1.0
                vec[(code * 7) % dim] += 0.5
            if not vec.any():
                vec[0] = 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            rows.append(vec)
        payload = np.vstack(rows)
        return payload[0] if single else payload


class _KnownPerson:
    def __init__(self, person_id: str, registry: Dict[str, str], reverse_registry: Dict[str, str]) -> None:
        self.person_id = person_id
        self.is_known = person_id in reverse_registry
        self.person_name = reverse_registry.get(person_id, "")
        self._registry = registry


class _KernelBackedRuntimeManager:
    def __init__(self, kernel: SDKMemoryKernel) -> None:
        self.kernel = kernel

    async def invoke(self, component_name: str, args: Dict[str, Any] | None, *, timeout_ms: int = 30000):
        del timeout_ms
        payload = args or {}
        if component_name == "search_memory":
            return await self.kernel.search_memory(
                KernelSearchRequest(
                    query=str(payload.get("query", "") or ""),
                    limit=int(payload.get("limit", 5) or 5),
                    mode=str(payload.get("mode", "hybrid") or "hybrid"),
                    chat_id=str(payload.get("chat_id", "") or ""),
                    person_id=str(payload.get("person_id", "") or ""),
                    time_start=payload.get("time_start"),
                    time_end=payload.get("time_end"),
                    respect_filter=bool(payload.get("respect_filter", True)),
                    user_id=str(payload.get("user_id", "") or ""),
                    group_id=str(payload.get("group_id", "") or ""),
                )
            )

        handler = getattr(self.kernel, component_name)
        result = handler(**payload)
        return await result if inspect.isawaitable(result) else result


async def _wait_for_import_task(task_id: str, *, max_rounds: int = 200, sleep_seconds: float = 0.05) -> Dict[str, Any]:
    for _ in range(max_rounds):
        detail = await memory_service.import_admin(action="get", task_id=task_id, include_chunks=True)
        task = detail.get("task") or {}
        status = str(task.get("status", "") or "")
        if status in {"completed", "completed_with_errors", "failed", "cancelled"}:
            return detail
        await asyncio.sleep(max(0.01, float(sleep_seconds)))
    raise AssertionError(f"导入任务在等待窗口内未结束: {task_id}")


def _join_hit_content(search_result: MemorySearchResult) -> str:
    return "\n".join(hit.content for hit in search_result.hits)


def _keyword_hits(text: str, keywords: List[str]) -> int:
    haystack = str(text or "")
    return sum(1 for keyword in keywords if keyword in haystack)


def _keyword_recall(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    return _keyword_hits(text, keywords) / float(len(keywords))


def _hit_blob(hit) -> str:
    meta = hit.metadata if isinstance(hit.metadata, dict) else {}
    return "\n".join(
        [
            str(hit.content or ""),
            str(hit.title or ""),
            str(hit.source or ""),
            json.dumps(meta, ensure_ascii=False),
        ]
    )


def _first_relevant_rank(search_result: MemorySearchResult, keywords: List[str], minimum_keyword_hits: int) -> int:
    for index, hit in enumerate(search_result.hits[:5], start=1):
        if _keyword_hits(_hit_blob(hit), keywords) >= max(1, int(minimum_keyword_hits or len(keywords))):
            return index
    return 0


def _episode_blob_from_items(items: List[Dict[str, Any]]) -> str:
    return "\n".join(
        (
            f"{item.get('title', '')}\n"
            f"{item.get('summary', '')}\n"
            f"{json.dumps(item.get('keywords', []), ensure_ascii=False)}\n"
            f"{json.dumps(item.get('participants', []), ensure_ascii=False)}"
        )
        for item in items
    )


def _episode_blob_from_hits(search_result: MemorySearchResult) -> str:
    chunks = []
    for hit in search_result.hits:
        meta = hit.metadata if isinstance(hit.metadata, dict) else {}
        chunks.append(
            "\n".join(
                [
                    str(hit.title or ""),
                    str(hit.content or ""),
                    json.dumps(meta.get("keywords", []) or [], ensure_ascii=False),
                    json.dumps(meta.get("participants", []) or [], ensure_ascii=False),
                ]
            )
        )
    return "\n".join(chunks)


async def _evaluate_episode_generation(*, session_id: str, episode_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    episode_source = f"chat_summary:{session_id}"
    payload = await memory_service.episode_admin(action="query", source=episode_source, limit=20)
    items = payload.get("items") or []
    blob = _episode_blob_from_items(items)
    reports: List[Dict[str, Any]] = []
    success_rate = 0.0
    keyword_recall = 0.0

    for case in episode_cases:
        recall = _keyword_recall(blob, case["expected_keywords"])
        success = bool(items) and recall >= float(case.get("minimum_keyword_recall", 1.0))
        success_rate += 1.0 if success else 0.0
        keyword_recall += recall
        reports.append(
            {
                "query": case["query"],
                "success": success,
                "keyword_recall": recall,
                "episode_count": len(items),
                "top_episode": items[0] if items else None,
            }
        )

    total = max(1, len(episode_cases))
    return {
        "success_rate": round(success_rate / total, 4),
        "keyword_recall": round(keyword_recall / total, 4),
        "episode_count": len(items),
        "reports": reports,
    }


async def _evaluate_episode_admin_query(*, session_id: str, episode_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    reports: List[Dict[str, Any]] = []
    success_rate = 0.0
    keyword_recall = 0.0
    episode_source = f"chat_summary:{session_id}"

    for case in episode_cases:
        payload = await memory_service.episode_admin(
            action="query",
            source=episode_source,
            query=case["query"],
            limit=5,
        )
        items = payload.get("items") or []
        blob = "\n".join(
            f"{item.get('title', '')}\n{item.get('summary', '')}\n{json.dumps(item.get('keywords', []), ensure_ascii=False)}"
            for item in items
        )
        recall = _keyword_recall(blob, case["expected_keywords"])
        success = bool(items) and recall >= float(case.get("minimum_keyword_recall", 1.0))
        success_rate += 1.0 if success else 0.0
        keyword_recall += recall
        reports.append(
            {
                "query": case["query"],
                "success": success,
                "keyword_recall": recall,
                "episode_count": len(items),
                "top_episode": items[0] if items else None,
            }
        )

    total = max(1, len(episode_cases))
    return {
        "success_rate": round(success_rate / total, 4),
        "keyword_recall": round(keyword_recall / total, 4),
        "reports": reports,
    }


async def _evaluate_episode_search_mode(*, session_id: str, episode_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    reports: List[Dict[str, Any]] = []
    success_rate = 0.0
    keyword_recall = 0.0

    for case in episode_cases:
        result = await memory_service.search(
            case["query"],
            mode="episode",
            chat_id=session_id,
            respect_filter=False,
            limit=5,
        )
        blob = _episode_blob_from_hits(result)
        recall = _keyword_recall(blob, case["expected_keywords"])
        success = bool(result.hits) and recall >= float(case.get("minimum_keyword_recall", 1.0))
        success_rate += 1.0 if success else 0.0
        keyword_recall += recall
        reports.append(
            {
                "query": case["query"],
                "success": success,
                "keyword_recall": recall,
                "episode_count": len(result.hits),
                "top_episode": result.hits[0].to_dict() if result.hits else None,
            }
        )

    total = max(1, len(episode_cases))
    return {
        "success_rate": round(success_rate / total, 4),
        "keyword_recall": round(keyword_recall / total, 4),
        "reports": reports,
    }


async def _evaluate_time_cases(*, session_id: str, time_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    reports: List[Dict[str, Any]] = []
    success_rate = 0.0
    keyword_recall = 0.0

    for case in time_cases:
        result = await memory_service.search(
            case["query"],
            mode="time",
            chat_id=session_id,
            time_start=case["time_expression"],
            time_end=case["time_expression"],
            respect_filter=False,
            limit=5,
        )
        blob = "\n".join(_hit_blob(hit) for hit in result.hits)
        recall = _keyword_recall(blob, case["expected_keywords"])
        success = bool(result.hits) and recall >= 0.67
        success_rate += 1.0 if success else 0.0
        keyword_recall += recall
        reports.append(
            {
                "query": case["query"],
                "time_expression": case["time_expression"],
                "success": success,
                "keyword_recall": recall,
                "hit_count": len(result.hits),
                "top_hit": result.hits[0].to_dict() if result.hits else None,
            }
        )

    total = max(1, len(time_cases))
    return {
        "success_rate": round(success_rate / total, 4),
        "keyword_recall": round(keyword_recall / total, 4),
        "reports": reports,
    }


async def _evaluate_tool_modes(*, session_id: str, dataset: Dict[str, Any]) -> Dict[str, Any]:
    search_case = dataset["search_cases"][0]
    time_case = dataset["time_cases"][0]
    episode_case = dataset["episode_cases"][0]
    aggregate_case = dataset["knowledge_fetcher_cases"][0]
    tool_cases = [
        {
            "name": "search",
            "kwargs": {
                "query": search_case["query"],
                "mode": "search",
                "chat_id": session_id,
                "limit": 5,
            },
            "expected_keywords": search_case["expected_keywords"],
            "minimum_keyword_recall": 0.67,
        },
        {
            "name": "time",
            "kwargs": {
                "query": time_case["query"],
                "mode": "time",
                "chat_id": session_id,
                "limit": 5,
                "time_expression": time_case["time_expression"],
            },
            "expected_keywords": time_case["expected_keywords"],
            "minimum_keyword_recall": 0.67,
        },
        {
            "name": "episode",
            "kwargs": {
                "query": episode_case["query"],
                "mode": "episode",
                "chat_id": session_id,
                "limit": 5,
            },
            "expected_keywords": episode_case["expected_keywords"],
            "minimum_keyword_recall": 0.67,
        },
        {
            "name": "aggregate",
            "kwargs": {
                "query": aggregate_case["query"],
                "mode": "aggregate",
                "chat_id": session_id,
                "limit": 5,
            },
            "expected_keywords": aggregate_case["expected_keywords"],
            "minimum_keyword_recall": 0.67,
        },
    ]
    reports: List[Dict[str, Any]] = []
    success_rate = 0.0
    keyword_recall = 0.0

    for case in tool_cases:
        text = await query_long_term_memory(**case["kwargs"])
        recall = _keyword_recall(text, case["expected_keywords"])
        success = (
            "失败" not in text
            and "无法解析" not in text
            and "未找到" not in text
            and recall >= float(case["minimum_keyword_recall"])
        )
        success_rate += 1.0 if success else 0.0
        keyword_recall += recall
        reports.append(
            {
                "name": case["name"],
                "success": success,
                "keyword_recall": recall,
                "preview": text[:320],
            }
        )

    total = max(1, len(tool_cases))
    return {
        "success_rate": round(success_rate / total, 4),
        "keyword_recall": round(keyword_recall / total, 4),
        "reports": reports,
    }


def _parse_fixture_message(
    *,
    line: str,
    session_id: str,
    platform: str,
    group_id: str,
    group_name: str,
    speaker_to_user_id: Dict[str, str],
    seq: int,
):
    match = re.match(r"^\[(?P<ts>[^\]]+)\]\s*(?P<speaker>[^：]+)：(?P<text>.*)$", str(line or "").strip())
    if not match:
        raise ValueError(f"无法解析 fixture 消息: {line}")

    dt = datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M")
    speaker = match.group("speaker").strip()
    text = match.group("text").strip()
    user_id = speaker_to_user_id.setdefault(speaker, f"user-{len(speaker_to_user_id) + 1}")

    return SimpleNamespace(
        message_id=f"{session_id}-{seq}",
        timestamp=dt,
        platform=platform,
        session_id=session_id,
        reply_to=None,
        processed_plain_text=text,
        display_message=f"{speaker}：{text}",
        message_info=SimpleNamespace(
            user_info=SimpleNamespace(
                user_id=user_id,
                user_nickname=speaker,
                user_cardname=speaker,
            ),
            group_info=SimpleNamespace(group_id=group_id, group_name=group_name),
            additional_config={},
        ),
    )


async def _evaluate_runtime_trigger_flow(*, dataset: Dict[str, Any], session: Any) -> Dict[str, Any]:
    streams = dataset["runtime_trigger_streams"]
    reports: List[Dict[str, Any]] = []
    positive_successes = 0.0
    negative_successes = 0.0

    speaker_to_user_id = {"Mai": "bot-mai"}
    for payload in dataset["person_writebacks"]:
        speaker_to_user_id.setdefault(payload["person_name"], payload["person_id"])

    original_get_messages = summarizer_module.message_api.get_messages_by_time_in_chat
    original_build_readable = summarizer_module.message_api.build_readable_messages
    original_is_bot_self = summarizer_module.is_bot_self
    original_person = summarizer_module.Person
    original_analyze = summarizer_module.ChatHistorySummarizer._analyze_topics_with_llm

    try:
        summarizer_module.message_api.build_readable_messages = (
            lambda messages, **kwargs: "\n".join(getattr(msg, "display_message", getattr(msg, "processed_plain_text", "")) for msg in messages)
        )
        summarizer_module.is_bot_self = lambda platform, user_id: str(user_id or "") == "bot-mai"
        summarizer_module.Person = lambda platform="", user_id="", person_id="": SimpleNamespace(
            person_name=next(
                (
                    name
                    for name, mapped in speaker_to_user_id.items()
                    if mapped == (str(person_id or "").strip() or str(user_id or "").strip())
                ),
                "",
            )
        )

        async def fake_analyze(self, numbered_lines, existing_topics):
            del existing_topics
            topic = str(getattr(self, "_fixture_topic", "") or "群聊话题")
            return True, {topic: list(range(1, len(numbered_lines) + 1))}

        summarizer_module.ChatHistorySummarizer._analyze_topics_with_llm = fake_analyze

        for index, stream in enumerate(streams, start=1):
            messages = [
                _parse_fixture_message(
                    line=line,
                    session_id=session.session_id,
                    platform=session.platform,
                    group_id=session.group_id,
                    group_name=dataset["session"]["display_name"],
                    speaker_to_user_id=speaker_to_user_id,
                    seq=1000 * index + seq,
                )
                for seq, line in enumerate(stream["messages"], start=1)
            ]

            def fake_get_messages_by_time_in_chat(
                *,
                chat_id: str,
                start_time: float,
                end_time: float,
                limit: int = 0,
                limit_mode: str = "latest",
                filter_mai: bool = False,
                filter_command: bool = False,
            ):
                del limit, limit_mode, filter_mai, filter_command
                if chat_id != session.session_id:
                    return []
                return [
                    msg
                    for msg in messages
                    if start_time <= msg.timestamp.timestamp() <= end_time
                ]

            summarizer_module.message_api.get_messages_by_time_in_chat = fake_get_messages_by_time_in_chat

            summarizer = summarizer_module.ChatHistorySummarizer(session.session_id)
            summarizer._fixture_topic = stream["topic"]
            summarizer._persist_topic_cache = lambda: None
            summarizer.last_check_time = float(stream["start_time"]) - 60.0
            summarizer.last_topic_check_time = float(stream["end_time"]) - float(stream["elapsed_since_last_check_hours"]) * 3600.0

            await summarizer.process(current_time=float(stream["end_time"]))

            topic_cache_keys = list(summarizer.topic_cache.keys())
            expected_topic_present = stream["topic"] in summarizer.topic_cache
            topic_cache_updated = bool(topic_cache_keys)
            batch_cleared = summarizer.current_batch is None

            if stream["bot_participated"]:
                success = expected_topic_present and topic_cache_updated and batch_cleared
                positive_successes += 1.0 if success else 0.0
            else:
                success = (not topic_cache_updated) and batch_cleared
                negative_successes += 1.0 if success else 0.0

            reports.append(
                {
                    "stream_id": stream["stream_id"],
                    "bot_participated": stream["bot_participated"],
                    "success": success,
                    "topic_cache_keys": topic_cache_keys,
                    "current_batch_cleared": batch_cleared,
                    "expected_check_outcome": stream["expected_check_outcome"],
                }
            )
    finally:
        summarizer_module.message_api.get_messages_by_time_in_chat = original_get_messages
        summarizer_module.message_api.build_readable_messages = original_build_readable
        summarizer_module.is_bot_self = original_is_bot_self
        summarizer_module.Person = original_person
        summarizer_module.ChatHistorySummarizer._analyze_topics_with_llm = original_analyze

    positive_total = max(1, sum(1 for item in streams if item["bot_participated"]))
    negative_total = max(1, sum(1 for item in streams if not item["bot_participated"]))
    return {
        "positive_trigger_rate": round(positive_successes / positive_total, 4),
        "negative_discard_rate": round(negative_successes / negative_total, 4),
        "reports": reports,
    }


def _average(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(sum(float(v) for v in values) / float(len(values)), 4)


def _target_score(actual: float, target: float) -> float:
    clean_target = float(target or 0.0)
    if clean_target <= 0:
        return 1.0
    return min(float(actual) / clean_target, 1.0)


def _build_final_score(*, metrics: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
    episode_summary = metrics["episode_summary_after_rebuild"]
    category_scores = {
        "search": _average(
            [
                _target_score(metrics["search"]["accuracy_at_1"], targets["search"]["accuracy_at_1"]),
                _target_score(metrics["search"]["recall_at_5"], targets["search"]["recall_at_5"]),
                _target_score(metrics["search"]["keyword_recall_at_5"], targets["search"]["keyword_recall_at_5"]),
            ]
        ),
        "writeback": _average(
            [
                _target_score(metrics["writeback"]["success_rate"], targets["writeback"]["success_rate"]),
                _target_score(metrics["writeback"]["keyword_recall"], targets["writeback"]["keyword_recall"]),
            ]
        ),
        "knowledge_fetcher": _average(
            [
                _target_score(metrics["knowledge_fetcher"]["success_rate"], targets["knowledge_fetcher"]["success_rate"]),
                _target_score(metrics["knowledge_fetcher"]["keyword_recall"], targets["knowledge_fetcher"]["keyword_recall"]),
            ]
        ),
        "profile": _average(
            [
                _target_score(metrics["profile"]["success_rate"], targets["profile"]["success_rate"]),
                _target_score(metrics["profile"]["evidence_rate"], targets["profile"]["evidence_rate"]),
            ]
        ),
        "episode": _average(
            [
                _target_score(episode_summary["success_rate"], targets["episode"]["success_rate"]),
                _target_score(episode_summary["keyword_recall"], targets["episode"]["keyword_recall"]),
            ]
        ),
        "negative_control": _target_score(
            metrics["negative_control"]["zero_hit_rate"], targets["negative_control"]["zero_hit_rate"]
        ),
        "runtime_trigger": _average(
            [
                _target_score(
                    metrics["runtime_trigger"]["positive_trigger_rate"],
                    targets["runtime_trigger"]["positive_trigger_rate"],
                ),
                _target_score(
                    metrics["runtime_trigger"]["negative_discard_rate"],
                    targets["runtime_trigger"]["negative_discard_rate"],
                ),
            ]
        ),
    }
    overall_ratio = _average(list(category_scores.values()))
    return {
        "overall_ratio": overall_ratio,
        "overall_score": round(overall_ratio * 100.0, 2),
        "category_scores": {key: round(value * 100.0, 2) for key, value in category_scores.items()},
    }


async def _build_group_chat_benchmark_env(patch_manager: _PatchManager, tmp_path: Path):
    dataset = _load_benchmark_fixture()
    session_cfg = dataset["session"]
    session = SimpleNamespace(
        session_id=session_cfg["session_id"],
        platform=session_cfg["platform"],
        user_id=session_cfg["user_id"],
        group_id=session_cfg["group_id"],
    )
    fake_chat_manager = SimpleNamespace(
        get_session_by_session_id=lambda session_id: session if session_id == session.session_id else None,
        get_session_name=lambda session_id: session_cfg["display_name"] if session_id == session.session_id else session_id,
    )

    registry = {item["person_name"]: item["person_id"] for item in dataset["person_writebacks"]}
    reverse_registry = {value: key for key, value in registry.items()}

    patch_manager.setattr(kernel_module, "create_embedding_api_adapter", lambda **kwargs: _FakeEmbeddingAdapter())

    async def fake_self_check(**kwargs):
        return {"ok": True, "message": "ok", "encoded_dimension": 32}

    patch_manager.setattr(kernel_module, "run_embedding_runtime_self_check", fake_self_check)
    patch_manager.setattr(summarizer_module, "_chat_manager", fake_chat_manager)
    patch_manager.setattr(knowledge_module, "_chat_manager", fake_chat_manager)
    patch_manager.setattr(person_info_module, "_chat_manager", fake_chat_manager)
    patch_manager.setattr(
        person_info_module,
        "get_person_id_by_person_name",
        lambda person_name: registry.get(str(person_name or "").strip(), ""),
    )
    patch_manager.setattr(
        person_info_module,
        "Person",
        lambda person_id: _KnownPerson(person_id=str(person_id or ""), registry=registry, reverse_registry=reverse_registry),
    )

    data_dir = (tmp_path / "a_memorix_group_chat_benchmark_data").resolve()
    kernel = SDKMemoryKernel(
        plugin_root=tmp_path / "plugin_root",
        config={
            "storage": {"data_dir": str(data_dir)},
            "advanced": {"enable_auto_save": False},
            "memory": {"base_decay_interval_hours": 24},
            "person_profile": {"refresh_interval_minutes": 5},
        },
    )
    manager = _KernelBackedRuntimeManager(kernel)
    patch_manager.setattr(memory_service_module, "a_memorix_host_service", manager)

    await kernel.initialize()
    return {
        "dataset": dataset,
        "kernel": kernel,
        "session": session,
        "person_registry": registry,
    }


async def _run_group_chat_stream_memory_benchmark(tmp_path: Path):
    patch_manager = _PatchManager()
    group_chat_benchmark_env = await _build_group_chat_benchmark_env(patch_manager, tmp_path)
    dataset = group_chat_benchmark_env["dataset"]
    session_id = group_chat_benchmark_env["session"].session_id

    try:
        created = await memory_service.import_admin(
            action="create_paste",
            name="group_chat_stream_memory_benchmark.json",
            input_mode="json",
            llm_enabled=False,
            content=json.dumps(dataset["import_payload"], ensure_ascii=False),
        )
        assert created["success"] is True

        import_detail = await _wait_for_import_task(created["task"]["task_id"])
        assert import_detail["task"]["status"] == "completed"

        for record in dataset["chat_history_records"]:
            summarizer = summarizer_module.ChatHistorySummarizer(session_id)
            await summarizer._import_to_long_term_memory(
                record_id=record["record_id"],
                theme=record["theme"],
                summary=record["summary"],
                participants=record["participants"],
                start_time=record["start_time"],
                end_time=record["end_time"],
                original_text=record["original_text"],
            )

        for payload in dataset["person_writebacks"]:
            await person_info_module.store_person_memory_from_answer(
                payload["person_name"],
                payload["memory_content"],
                session_id,
            )

        await memory_service.episode_admin(action="process_pending", limit=100, max_retry=2)

        search_case_reports: List[Dict[str, Any]] = []
        search_accuracy_at_1 = 0.0
        search_recall_at_5 = 0.0
        search_precision_at_5 = 0.0
        search_mrr = 0.0
        search_keyword_recall = 0.0

        for case in dataset["search_cases"]:
            result = await memory_service.search(
                case["query"],
                mode="search",
                chat_id=session_id,
                respect_filter=False,
                limit=5,
            )
            joined = "\n".join(_hit_blob(hit) for hit in result.hits)
            rank = _first_relevant_rank(
                result, case["expected_keywords"], case.get("minimum_keyword_hits", len(case["expected_keywords"]))
            )
            relevant_hits = sum(
                1
                for hit in result.hits[:5]
                if _keyword_hits(_hit_blob(hit), case["expected_keywords"]) >= max(
                    1, int(case.get("minimum_keyword_hits", len(case["expected_keywords"])))
                )
            )
            keyword_recall = _keyword_recall(joined, case["expected_keywords"])
            search_accuracy_at_1 += 1.0 if rank == 1 else 0.0
            search_recall_at_5 += 1.0 if rank > 0 else 0.0
            search_precision_at_5 += relevant_hits / float(max(1, min(5, len(result.hits))))
            search_mrr += 1.0 / float(rank) if rank > 0 else 0.0
            search_keyword_recall += keyword_recall
            search_case_reports.append(
                {
                    "query": case["query"],
                    "rank_of_first_relevant": rank,
                    "relevant_hits_top5": relevant_hits,
                    "keyword_recall_top5": keyword_recall,
                    "top_hit": result.hits[0].to_dict() if result.hits else None,
                }
            )

        search_total = max(1, len(dataset["search_cases"]))

        writeback_reports: List[Dict[str, Any]] = []
        writeback_success_rate = 0.0
        writeback_keyword_recall = 0.0
        for payload in dataset["person_writebacks"]:
            query = " ".join(payload["expected_keywords"])
            result = await memory_service.search(
                query,
                mode="search",
                chat_id=session_id,
                person_id=payload["person_id"],
                respect_filter=False,
                limit=5,
            )
            joined = _join_hit_content(result)
            recall = _keyword_recall(joined, payload["expected_keywords"])
            success = bool(result.hits) and recall >= 0.67
            writeback_success_rate += 1.0 if success else 0.0
            writeback_keyword_recall += recall
            writeback_reports.append(
                {
                    "person_id": payload["person_id"],
                    "success": success,
                    "keyword_recall": recall,
                    "hit_count": len(result.hits),
                }
            )
        writeback_total = max(1, len(dataset["person_writebacks"]))

        knowledge_reports: List[Dict[str, Any]] = []
        knowledge_success_rate = 0.0
        knowledge_keyword_recall = 0.0
        fetcher = knowledge_module.KnowledgeFetcher(
            private_name=dataset["session"]["display_name"],
            stream_id=session_id,
        )
        for case in dataset["knowledge_fetcher_cases"]:
            knowledge_text, _ = await fetcher.fetch(case["query"], [])
            recall = _keyword_recall(knowledge_text, case["expected_keywords"])
            success = recall >= float(case.get("minimum_keyword_recall", 1.0))
            knowledge_success_rate += 1.0 if success else 0.0
            knowledge_keyword_recall += recall
            knowledge_reports.append(
                {
                    "query": case["query"],
                    "success": success,
                    "keyword_recall": recall,
                    "preview": knowledge_text[:300],
                }
            )
        knowledge_total = max(1, len(dataset["knowledge_fetcher_cases"]))

        profile_reports: List[Dict[str, Any]] = []
        profile_success_rate = 0.0
        profile_keyword_recall = 0.0
        profile_evidence_rate = 0.0
        for case in dataset["profile_cases"]:
            profile = await memory_service.get_person_profile(case["person_id"], chat_id=session_id)
            recall = _keyword_recall(profile.summary, case["expected_keywords"])
            has_evidence = bool(profile.evidence)
            success = recall >= float(case.get("minimum_keyword_recall", 1.0)) and has_evidence
            profile_success_rate += 1.0 if success else 0.0
            profile_keyword_recall += recall
            profile_evidence_rate += 1.0 if has_evidence else 0.0
            profile_reports.append(
                {
                    "person_id": case["person_id"],
                    "success": success,
                    "keyword_recall": recall,
                    "evidence_count": len(profile.evidence),
                    "summary_preview": profile.summary[:240],
                }
            )
        profile_total = max(1, len(dataset["profile_cases"]))

        time_mode = await _evaluate_time_cases(session_id=session_id, time_cases=dataset["time_cases"])
        episode_generation_auto = await _evaluate_episode_generation(
            session_id=session_id, episode_cases=dataset["episode_cases"]
        )
        episode_admin_query_auto = await _evaluate_episode_admin_query(
            session_id=session_id, episode_cases=dataset["episode_cases"]
        )
        episode_search_mode_auto = await _evaluate_episode_search_mode(
            session_id=session_id, episode_cases=dataset["episode_cases"]
        )
        episode_rebuild = await memory_service.episode_admin(action="rebuild", source=f"chat_summary:{session_id}")
        episode_generation_after_rebuild = await _evaluate_episode_generation(
            session_id=session_id, episode_cases=dataset["episode_cases"]
        )
        episode_admin_query_after_rebuild = await _evaluate_episode_admin_query(
            session_id=session_id, episode_cases=dataset["episode_cases"]
        )
        episode_search_mode_after_rebuild = await _evaluate_episode_search_mode(
            session_id=session_id, episode_cases=dataset["episode_cases"]
        )
        tool_modes = await _evaluate_tool_modes(session_id=session_id, dataset=dataset)

        negative_reports: List[Dict[str, Any]] = []
        negative_zero_hits = 0.0
        for case in dataset["negative_control_cases"]:
            result = await memory_service.search(
                case["query"],
                mode="search",
                chat_id=session_id,
                respect_filter=False,
                limit=5,
            )
            success = len(result.hits) == 0
            negative_zero_hits += 1.0 if success else 0.0
            negative_reports.append(
                {
                    "query": case["query"],
                    "success": success,
                    "hit_count": len(result.hits),
                    "top_hit": result.hits[0].to_dict() if result.hits else None,
                }
            )
        negative_total = max(1, len(dataset["negative_control_cases"]))

        runtime_trigger = await _evaluate_runtime_trigger_flow(
            dataset=dataset,
            session=group_chat_benchmark_env["session"],
        )

        episode_summary_after_rebuild = {
            "success_rate": _average(
                [
                    episode_generation_after_rebuild["success_rate"],
                    episode_admin_query_after_rebuild["success_rate"],
                    episode_search_mode_after_rebuild["success_rate"],
                ]
            ),
            "keyword_recall": _average(
                [
                    episode_generation_after_rebuild["keyword_recall"],
                    episode_admin_query_after_rebuild["keyword_recall"],
                    episode_search_mode_after_rebuild["keyword_recall"],
                ]
            ),
        }

        metrics = {
            "search": {
                "accuracy_at_1": round(search_accuracy_at_1 / search_total, 4),
                "recall_at_5": round(search_recall_at_5 / search_total, 4),
                "precision_at_5": round(search_precision_at_5 / search_total, 4),
                "mrr": round(search_mrr / search_total, 4),
                "keyword_recall_at_5": round(search_keyword_recall / search_total, 4),
            },
            "time_mode": {
                "success_rate": time_mode["success_rate"],
                "keyword_recall": time_mode["keyword_recall"],
            },
            "writeback": {
                "success_rate": round(writeback_success_rate / writeback_total, 4),
                "keyword_recall": round(writeback_keyword_recall / writeback_total, 4),
            },
            "knowledge_fetcher": {
                "success_rate": round(knowledge_success_rate / knowledge_total, 4),
                "keyword_recall": round(knowledge_keyword_recall / knowledge_total, 4),
            },
            "profile": {
                "success_rate": round(profile_success_rate / profile_total, 4),
                "keyword_recall": round(profile_keyword_recall / profile_total, 4),
                "evidence_rate": round(profile_evidence_rate / profile_total, 4),
            },
            "tool_modes": {
                "success_rate": tool_modes["success_rate"],
                "keyword_recall": tool_modes["keyword_recall"],
            },
            "episode_generation_auto": {
                "success_rate": episode_generation_auto["success_rate"],
                "keyword_recall": episode_generation_auto["keyword_recall"],
                "episode_count": episode_generation_auto["episode_count"],
            },
            "episode_generation_after_rebuild": {
                "success_rate": episode_generation_after_rebuild["success_rate"],
                "keyword_recall": episode_generation_after_rebuild["keyword_recall"],
                "episode_count": episode_generation_after_rebuild["episode_count"],
                "rebuild_success": bool(episode_rebuild.get("success", False)),
            },
            "episode_admin_query_auto": {
                "success_rate": episode_admin_query_auto["success_rate"],
                "keyword_recall": episode_admin_query_auto["keyword_recall"],
            },
            "episode_admin_query_after_rebuild": {
                "success_rate": episode_admin_query_after_rebuild["success_rate"],
                "keyword_recall": episode_admin_query_after_rebuild["keyword_recall"],
                "rebuild_success": bool(episode_rebuild.get("success", False)),
            },
            "episode_search_mode_auto": {
                "success_rate": episode_search_mode_auto["success_rate"],
                "keyword_recall": episode_search_mode_auto["keyword_recall"],
            },
            "episode_search_mode_after_rebuild": {
                "success_rate": episode_search_mode_after_rebuild["success_rate"],
                "keyword_recall": episode_search_mode_after_rebuild["keyword_recall"],
                "rebuild_success": bool(episode_rebuild.get("success", False)),
            },
            "episode_summary_after_rebuild": episode_summary_after_rebuild,
            "negative_control": {
                "zero_hit_rate": round(negative_zero_hits / negative_total, 4),
            },
            "runtime_trigger": {
                "positive_trigger_rate": runtime_trigger["positive_trigger_rate"],
                "negative_discard_rate": runtime_trigger["negative_discard_rate"],
            },
        }
        final_score = _build_final_score(metrics=metrics, targets=dataset["meta"]["quantitative_targets"])

        report = {
            "dataset": dataset["meta"],
            "import": {
                "task_id": created["task"]["task_id"],
                "status": import_detail["task"]["status"],
                "paragraph_count": len(dataset["import_payload"]["paragraphs"]),
                "relation_count": len(dataset["import_payload"].get("relations") or []),
            },
            "metrics": metrics,
            "score": final_score,
            "cases": {
                "search": search_case_reports,
                "time_mode": time_mode["reports"],
                "writeback": writeback_reports,
                "knowledge_fetcher": knowledge_reports,
                "profile": profile_reports,
                "tool_modes": tool_modes["reports"],
                "episode_generation_auto": episode_generation_auto["reports"],
                "episode_generation_after_rebuild": episode_generation_after_rebuild["reports"],
                "episode_admin_query_auto": episode_admin_query_auto["reports"],
                "episode_admin_query_after_rebuild": episode_admin_query_after_rebuild["reports"],
                "episode_search_mode_auto": episode_search_mode_auto["reports"],
                "episode_search_mode_after_rebuild": episode_search_mode_after_rebuild["reports"],
                "negative_control": negative_reports,
                "runtime_trigger": runtime_trigger["reports"],
            },
        }

        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"metrics": report["metrics"], "score": report["score"]}, ensure_ascii=False, indent=2))

        assert report["import"]["status"] == "completed"
        assert report["score"]["overall_score"] >= 0.0
        return report
    finally:
        await group_chat_benchmark_env["kernel"].shutdown()
        patch_manager.undo()


def run_group_chat_stream_memory_benchmark() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="a_memorix_group_chat_benchmark_") as tmp_dir:
        return asyncio.run(_run_group_chat_stream_memory_benchmark(Path(tmp_dir)))


if __name__ == "__main__":
    report = run_group_chat_stream_memory_benchmark()
    print(json.dumps({"final_score": report["score"], "metrics": report["metrics"]}, ensure_ascii=False, indent=2))
