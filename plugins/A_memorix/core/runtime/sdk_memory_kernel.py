from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.common.logger import get_logger

from ..embedding import create_embedding_api_adapter
from ..retrieval import (
    DualPathRetriever,
    DualPathRetrieverConfig,
    RetrievalResult,
    SparseBM25Config,
    SparseBM25Index,
    TemporalQueryOptions,
)
from ..storage import GraphStore, MetadataStore, QuantizationType, SparseMatrixFormat, VectorStore
from ..utils.aggregate_query_service import AggregateQueryService
from ..utils.episode_retrieval_service import EpisodeRetrievalService
from ..utils.hash import normalize_text
from ..utils.relation_write_service import RelationWriteService

logger = get_logger("A_Memorix.SDKMemoryKernel")


@dataclass
class KernelSearchRequest:
    query: str = ""
    limit: int = 5
    mode: str = "hybrid"
    chat_id: str = ""
    person_id: str = ""
    time_start: Optional[float] = None
    time_end: Optional[float] = None


class SDKMemoryKernel:
    def __init__(self, *, plugin_root: Path, config: Optional[Dict[str, Any]] = None) -> None:
        self.plugin_root = Path(plugin_root).resolve()
        self.config = config or {}
        storage_cfg = self._cfg("storage", {}) or {}
        data_dir = str(storage_cfg.get("data_dir", "./data") or "./data")
        self.data_dir = (self.plugin_root / data_dir).resolve() if data_dir.startswith(".") else Path(data_dir)
        self.embedding_dimension = max(32, int(self._cfg("embedding.dimension", 256)))
        self.relation_vectors_enabled = bool(self._cfg("retrieval.relation_vectorization.enabled", False))

        self.embedding_manager = None
        self.vector_store: Optional[VectorStore] = None
        self.graph_store: Optional[GraphStore] = None
        self.metadata_store: Optional[MetadataStore] = None
        self.relation_write_service: Optional[RelationWriteService] = None
        self.sparse_index = None
        self.retriever: Optional[DualPathRetriever] = None
        self.episode_retriever: Optional[EpisodeRetrievalService] = None
        self.aggregate_query_service: Optional[AggregateQueryService] = None
        self._initialized = False
        self._last_maintenance_at: Optional[float] = None

    def _cfg(self, key: str, default: Any = None) -> Any:
        current: Any = self.config
        if key in {"storage", "embedding", "retrieval"} and isinstance(current, dict):
            return current.get(key, default)
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    async def initialize(self) -> None:
        if self._initialized:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_manager = create_embedding_api_adapter(
            batch_size=int(self._cfg("embedding.batch_size", 32)),
            max_concurrent=int(self._cfg("embedding.max_concurrent", 5)),
            default_dimension=self.embedding_dimension,
            model_name=str(self._cfg("embedding.model_name", "hash-v1")),
            retry_config=self._cfg("embedding.retry", {}) or {},
        )
        self.embedding_dimension = int(await self.embedding_manager._detect_dimension())
        self.vector_store = VectorStore(
            dimension=self.embedding_dimension,
            quantization_type=QuantizationType.INT8,
            data_dir=self.data_dir / "vectors",
        )
        self.graph_store = GraphStore(matrix_format=SparseMatrixFormat.CSR, data_dir=self.data_dir / "graph")
        self.metadata_store = MetadataStore(data_dir=self.data_dir / "metadata")
        self.metadata_store.connect()
        if self.vector_store.has_data():
            self.vector_store.load()
            self.vector_store.warmup_index(force_train=True)
        if self.graph_store.has_data():
            self.graph_store.load()

        sparse_cfg = self._cfg("retrieval.sparse", {}) or {}
        self.sparse_index = SparseBM25Index(metadata_store=self.metadata_store, config=SparseBM25Config(**sparse_cfg))
        if getattr(self.sparse_index.config, "enabled", False):
            self.sparse_index.ensure_loaded()

        self.relation_write_service = RelationWriteService(
            metadata_store=self.metadata_store,
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
        )
        self.retriever = DualPathRetriever(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            metadata_store=self.metadata_store,
            embedding_manager=self.embedding_manager,
            sparse_index=self.sparse_index,
            config=DualPathRetrieverConfig(
                top_k_paragraphs=int(self._cfg("retrieval.top_k_paragraphs", 24)),
                top_k_relations=int(self._cfg("retrieval.top_k_relations", 12)),
                top_k_final=int(self._cfg("retrieval.top_k_final", 10)),
                alpha=float(self._cfg("retrieval.alpha", 0.5)),
                enable_ppr=bool(self._cfg("retrieval.enable_ppr", True)),
                ppr_alpha=float(self._cfg("retrieval.ppr_alpha", 0.85)),
                ppr_concurrency_limit=int(self._cfg("retrieval.ppr_concurrency_limit", 4)),
                enable_parallel=bool(self._cfg("retrieval.enable_parallel", True)),
                sparse=sparse_cfg,
                fusion=self._cfg("retrieval.fusion", {}) or {},
                graph_recall=self._cfg("retrieval.search.graph_recall", {}) or {},
                relation_intent=self._cfg("retrieval.search.relation_intent", {}) or {},
            ),
        )
        self.episode_retriever = EpisodeRetrievalService(metadata_store=self.metadata_store, retriever=self.retriever)
        self.aggregate_query_service = AggregateQueryService(plugin_config=self.config)
        self._initialized = True

    def close(self) -> None:
        if self.vector_store is not None:
            self.vector_store.save()
        if self.graph_store is not None:
            self.graph_store.save()
        if self.metadata_store is not None:
            self.metadata_store.close()
        self._initialized = False

    async def ingest_summary(
        self,
        *,
        external_id: str,
        chat_id: str,
        text: str,
        participants: Optional[Sequence[str]] = None,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        summary_meta = dict(metadata or {})
        summary_meta.setdefault("kind", "chat_summary")
        return await self.ingest_text(
            external_id=external_id,
            source_type="chat_summary",
            text=text,
            chat_id=chat_id,
            participants=participants,
            time_start=time_start,
            time_end=time_end,
            tags=tags,
            metadata=summary_meta,
        )

    async def ingest_text(
        self,
        *,
        external_id: str,
        source_type: str,
        text: str,
        chat_id: str = "",
        person_ids: Optional[Sequence[str]] = None,
        participants: Optional[Sequence[str]] = None,
        timestamp: Optional[float] = None,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entities: Optional[Sequence[str]] = None,
        relations: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        await self.initialize()
        assert self.metadata_store and self.vector_store and self.graph_store and self.embedding_manager
        assert self.relation_write_service
        content = normalize_text(text)
        if not content:
            return {"stored_ids": [], "skipped_ids": [external_id], "reason": "empty_text"}
        if ref := self.metadata_store.get_external_memory_ref(external_id):
            return {"stored_ids": [], "skipped_ids": [str(ref.get("paragraph_hash", ""))], "reason": "exists"}

        person_tokens = self._tokens(person_ids)
        participant_tokens = self._tokens(participants)
        entity_tokens = self._merge_tokens(entities, person_tokens, participant_tokens)
        source = self._build_source(source_type, chat_id, person_tokens)
        paragraph_meta = dict(metadata or {})
        paragraph_meta.update(
            {
                "external_id": external_id,
                "source_type": str(source_type or "").strip(),
                "chat_id": str(chat_id or "").strip(),
                "person_ids": person_tokens,
                "participants": participant_tokens,
                "tags": self._tokens(tags),
            }
        )
        paragraph_hash = self.metadata_store.add_paragraph(
            content=content,
            source=source,
            metadata=paragraph_meta,
            knowledge_type="factual" if source_type == "person_fact" else "narrative" if source_type == "chat_summary" else "mixed",
            time_meta=self._time_meta(timestamp, time_start, time_end),
        )
        embedding = await self.embedding_manager.encode(content)
        self.vector_store.add(vectors=embedding.reshape(1, -1), ids=[paragraph_hash])
        for name in entity_tokens:
            self.metadata_store.add_entity(name=name, source_paragraph=paragraph_hash)

        stored_relations: List[str] = []
        for row in [dict(item) for item in (relations or []) if isinstance(item, dict)]:
            s = str(row.get("subject", "") or "").strip()
            p = str(row.get("predicate", "") or "").strip()
            o = str(row.get("object", "") or "").strip()
            if not (s and p and o):
                continue
            result = await self.relation_write_service.upsert_relation_with_vector(
                subject=s,
                predicate=p,
                obj=o,
                confidence=float(row.get("confidence", 1.0) or 1.0),
                source_paragraph=paragraph_hash,
                metadata={"external_id": external_id, "source_type": source_type},
                write_vector=self.relation_vectors_enabled,
            )
            self.metadata_store.link_paragraph_relation(paragraph_hash, result.hash_value)
            stored_relations.append(result.hash_value)

        self.metadata_store.upsert_external_memory_ref(
            external_id=external_id,
            paragraph_hash=paragraph_hash,
            source_type=source_type,
            metadata={"chat_id": chat_id, "person_ids": person_tokens},
        )
        self._persist()
        self.rebuild_episodes_for_sources([source])
        for person_id in person_tokens:
            await self.refresh_person_profile(person_id)
        return {"stored_ids": [paragraph_hash, *stored_relations], "skipped_ids": []}

    async def search_memory(self, request: KernelSearchRequest) -> Dict[str, Any]:
        await self.initialize()
        assert self.retriever and self.episode_retriever and self.aggregate_query_service
        mode = str(request.mode or "hybrid").strip().lower() or "hybrid"
        clean_query = str(request.query or "").strip()
        limit = max(1, int(request.limit or 5))
        temporal = self._temporal(request)
        if mode == "episode":
            rows = await self.episode_retriever.query(
                query=clean_query,
                top_k=limit,
                time_from=request.time_start,
                time_to=request.time_end,
                source=self._chat_source(request.chat_id),
            )
            hits = [self._episode_hit(row) for row in rows]
            return {"summary": self._summary(hits), "hits": hits}
        if mode == "aggregate":
            payload = await self.aggregate_query_service.execute(
                query=clean_query,
                top_k=limit,
                mix=True,
                mix_top_k=limit,
                time_from=str(request.time_start) if request.time_start is not None else None,
                time_to=str(request.time_end) if request.time_end is not None else None,
                search_runner=lambda: self._aggregate_search(clean_query, limit, temporal),
                time_runner=lambda: self._aggregate_time(clean_query, limit, temporal),
                episode_runner=lambda: self._aggregate_episode(clean_query, limit, request),
            )
            hits = [dict(item) for item in payload.get("mixed_results", []) if isinstance(item, dict)]
            for item in hits:
                item.setdefault("metadata", {})
            return {"summary": self._summary(hits), "hits": hits}
        results = await self.retriever.retrieve(query=clean_query, top_k=limit, temporal=temporal)
        hits = [self._retrieval_hit(item) for item in results]
        return {"summary": self._summary(self._filter_hits(hits, request.person_id)), "hits": self._filter_hits(hits, request.person_id)}

    async def get_person_profile(self, *, person_id: str, chat_id: str = "", limit: int = 10) -> Dict[str, Any]:
        _ = chat_id
        await self.initialize()
        assert self.metadata_store
        snapshot = self.metadata_store.get_latest_person_profile_snapshot(person_id) or await self.refresh_person_profile(person_id, limit=limit)
        evidence = []
        for hash_value in snapshot.get("evidence_ids", [])[: max(1, int(limit))]:
            paragraph = self.metadata_store.get_paragraph(hash_value)
            if paragraph is not None:
                evidence.append({"hash": hash_value, "content": str(paragraph.get("content", "") or "")[:220], "metadata": paragraph.get("metadata", {}) or {}})
        text = str(snapshot.get("profile_text", "") or "").strip()
        traits = [line.strip("- ").strip() for line in text.splitlines() if line.strip()][:8]
        return {"summary": text, "traits": traits, "evidence": evidence}

    async def refresh_person_profile(self, person_id: str, limit: int = 10) -> Dict[str, Any]:
        await self.initialize()
        assert self.metadata_store
        rows = self.metadata_store.query(
            """
            SELECT DISTINCT p.*
            FROM paragraphs p
            JOIN paragraph_entities pe ON pe.paragraph_hash = p.hash
            JOIN entities e ON e.hash = pe.entity_hash
            WHERE e.name = ?
              AND (p.is_deleted IS NULL OR p.is_deleted = 0)
            ORDER BY COALESCE(p.event_time_end, p.event_time_start, p.event_time, p.updated_at, p.created_at) DESC
            LIMIT ?
            """,
            (person_id, max(1, int(limit)) * 3),
        )
        evidence_ids = [str(row.get("hash", "") or "") for row in rows if str(row.get("hash", "")).strip()]
        vector_evidence = [{"hash": str(row.get("hash", "") or ""), "type": "paragraph", "score": 0.0, "content": str(row.get("content", "") or "")[:220], "metadata": row.get("metadata", {}) or {}} for row in rows[: max(1, int(limit))]]
        relation_edges = [{"hash": str(row.get("hash", "") or ""), "subject": str(row.get("subject", "") or ""), "predicate": str(row.get("predicate", "") or ""), "object": str(row.get("object", "") or ""), "confidence": float(row.get("confidence", 1.0) or 1.0)} for row in self.metadata_store.get_relations(subject=person_id)[:limit]]
        if relation_edges:
            profile_text = "\n".join(f"{item['subject']} {item['predicate']} {item['object']}" for item in relation_edges[:6])
        elif vector_evidence:
            profile_text = "\n".join(f"- {item['content']}" for item in vector_evidence[:6])
        else:
            profile_text = "暂无稳定画像证据。"
        return self.metadata_store.upsert_person_profile_snapshot(
            person_id=person_id,
            profile_text=profile_text,
            aliases=[person_id],
            relation_edges=relation_edges,
            vector_evidence=vector_evidence,
            evidence_ids=evidence_ids[: max(1, int(limit))],
            expires_at=time.time() + 6 * 3600,
            source_note="sdk_memory_kernel",
        )

    async def maintain_memory(self, *, action: str, target: str, hours: Optional[float] = None, reason: str = "") -> Dict[str, Any]:
        _ = reason
        await self.initialize()
        assert self.metadata_store
        hashes = self._resolve_relation_hashes(target)
        if not hashes:
            return {"success": False, "detail": "未命中可维护关系"}
        act = str(action or "").strip().lower()
        if act == "reinforce":
            self.metadata_store.reinforce_relations(hashes)
        elif act == "protect":
            ttl_seconds = max(0.0, float(hours or 0.0)) * 3600.0
            self.metadata_store.protect_relations(hashes, ttl_seconds=ttl_seconds, is_pinned=ttl_seconds <= 0)
        elif act == "restore":
            restored = sum(1 for hash_value in hashes if self.metadata_store.restore_relation(hash_value))
            if restored <= 0:
                return {"success": False, "detail": "未恢复任何关系"}
        else:
            return {"success": False, "detail": f"不支持的维护动作: {act}"}
        self._last_maintenance_at = time.time()
        self._persist()
        return {"success": True, "detail": f"{act} {len(hashes)} 条关系"}

    def rebuild_episodes_for_sources(self, sources: Iterable[str]) -> int:
        assert self.metadata_store
        rebuilt = 0
        for source in self._tokens(sources):
            rows = self.metadata_store.query(
                """
                SELECT * FROM paragraphs
                WHERE source = ?
                  AND (is_deleted IS NULL OR is_deleted = 0)
                ORDER BY COALESCE(event_time_start, event_time, created_at) ASC, hash ASC
                """,
                (source,),
            )
            if not rows:
                continue
            paragraph_hashes = [str(row.get("hash", "") or "") for row in rows if str(row.get("hash", "")).strip()]
            payload = self.metadata_store.upsert_episode(
                {
                    "source": source,
                    "title": str((rows[0].get("metadata", {}) or {}).get("theme", "") or f"{source} 情景记忆")[:80],
                    "summary": "；".join(str(row.get("content", "") or "").strip().replace("\n", " ")[:120] for row in rows[:3] if str(row.get("content", "") or "").strip())[:500] or "自动构建的情景记忆。",
                    "participants": self._episode_participants(rows),
                    "keywords": self._episode_keywords(rows),
                    "evidence_ids": paragraph_hashes,
                    "paragraph_count": len(paragraph_hashes),
                    "event_time_start": self._time_bound(rows, "event_time_start", "event_time", reverse=False),
                    "event_time_end": self._time_bound(rows, "event_time_end", "event_time", reverse=True),
                    "time_granularity": "day",
                    "time_confidence": 0.7,
                    "llm_confidence": 0.0,
                    "segmentation_model": "rule_based_sdk",
                    "segmentation_version": "1",
                }
            )
            self.metadata_store.bind_episode_paragraphs(payload["episode_id"], paragraph_hashes)
            rebuilt += 1
        return rebuilt

    def memory_stats(self) -> Dict[str, Any]:
        assert self.metadata_store
        stats = self.metadata_store.get_statistics()
        episodes = self.metadata_store.query("SELECT COUNT(*) AS c FROM episodes")[0]["c"]
        profiles = self.metadata_store.query("SELECT COUNT(*) AS c FROM person_profile_snapshots")[0]["c"]
        return {"paragraphs": int(stats.get("paragraph_count", 0) or 0), "relations": int(stats.get("relation_count", 0) or 0), "episodes": int(episodes or 0), "profiles": int(profiles or 0), "last_maintenance_at": self._last_maintenance_at}

    async def _aggregate_search(self, query: str, limit: int, temporal: Optional[TemporalQueryOptions]) -> Dict[str, Any]:
        assert self.retriever
        hits = [self._retrieval_hit(item) for item in await self.retriever.retrieve(query=query, top_k=limit, temporal=temporal)]
        return {"success": True, "results": hits, "count": len(hits), "query_type": "search"}

    async def _aggregate_time(self, query: str, limit: int, temporal: Optional[TemporalQueryOptions]) -> Dict[str, Any]:
        if temporal is None:
            return {"success": False, "error": "missing temporal window", "results": []}
        assert self.retriever
        hits = [self._retrieval_hit(item) for item in await self.retriever.retrieve(query=query, top_k=limit, temporal=temporal)]
        return {"success": True, "results": hits, "count": len(hits), "query_type": "time"}

    async def _aggregate_episode(self, query: str, limit: int, request: KernelSearchRequest) -> Dict[str, Any]:
        assert self.episode_retriever
        rows = await self.episode_retriever.query(query=query, top_k=limit, time_from=request.time_start, time_to=request.time_end, source=self._chat_source(request.chat_id))
        hits = [self._episode_hit(row) for row in rows]
        return {"success": True, "results": hits, "count": len(hits), "query_type": "episode"}

    def _persist(self) -> None:
        if self.vector_store is not None:
            self.vector_store.save()
        if self.graph_store is not None:
            self.graph_store.save()
        if self.sparse_index is not None and getattr(self.sparse_index.config, "enabled", False):
            self.sparse_index.ensure_loaded()

    @staticmethod
    def _tokens(values: Optional[Iterable[Any]]) -> List[str]:
        result: List[str] = []
        seen = set()
        for item in values or []:
            token = str(item or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            result.append(token)
        return result

    @classmethod
    def _merge_tokens(cls, *groups: Optional[Iterable[Any]]) -> List[str]:
        merged: List[str] = []
        seen = set()
        for group in groups:
            for item in cls._tokens(group):
                if item in seen:
                    continue
                seen.add(item)
                merged.append(item)
        return merged

    @staticmethod
    def _build_source(source_type: str, chat_id: str, person_ids: Sequence[str]) -> str:
        clean_type = str(source_type or "").strip() or "memory"
        if clean_type == "chat_summary" and chat_id:
            return f"chat_summary:{chat_id}"
        if clean_type == "person_fact" and person_ids:
            return f"person_fact:{person_ids[0]}"
        return f"{clean_type}:{chat_id}" if chat_id else clean_type

    @staticmethod
    def _chat_source(chat_id: str) -> Optional[str]:
        clean = str(chat_id or "").strip()
        return f"chat_summary:{clean}" if clean else None

    @staticmethod
    def _time_meta(timestamp: Optional[float], time_start: Optional[float], time_end: Optional[float]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if timestamp is not None:
            payload["event_time"] = float(timestamp)
        if time_start is not None:
            payload["event_time_start"] = float(time_start)
        if time_end is not None:
            payload["event_time_end"] = float(time_end)
        if payload:
            payload["time_granularity"] = "minute"
            payload["time_confidence"] = 0.95
        return payload

    def _temporal(self, request: KernelSearchRequest) -> Optional[TemporalQueryOptions]:
        if request.time_start is None and request.time_end is None and not request.chat_id:
            return None
        return TemporalQueryOptions(time_from=request.time_start, time_to=request.time_end, source=self._chat_source(request.chat_id))

    @staticmethod
    def _retrieval_hit(item: RetrievalResult) -> Dict[str, Any]:
        payload = item.to_dict()
        return {"hash": payload.get("hash", ""), "content": payload.get("content", ""), "score": payload.get("score", 0.0), "type": payload.get("type", ""), "source": payload.get("source", ""), "metadata": payload.get("metadata", {}) or {}}

    @staticmethod
    def _episode_hit(row: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": "episode", "episode_id": str(row.get("episode_id", "") or ""), "title": str(row.get("title", "") or ""), "content": str(row.get("summary", "") or ""), "score": float(row.get("lexical_score", 0.0) or 0.0), "source": "episode", "metadata": {"participants": row.get("participants", []) or [], "keywords": row.get("keywords", []) or [], "source": row.get("source"), "event_time_start": row.get("event_time_start"), "event_time_end": row.get("event_time_end")}}

    @staticmethod
    def _summary(hits: Sequence[Dict[str, Any]]) -> str:
        if not hits:
            return ""
        lines = []
        for index, item in enumerate(hits[:5], start=1):
            content = str(item.get("content", "") or "").strip().replace("\n", " ")
            lines.append(f"{index}. {(content[:120] + '...') if len(content) > 120 else content}")
        return "\n".join(lines)

    @staticmethod
    def _filter_hits(hits: List[Dict[str, Any]], person_id: str) -> List[Dict[str, Any]]:
        if not person_id:
            return hits
        filtered = []
        for item in hits:
            metadata = item.get("metadata", {}) or {}
            if person_id in (metadata.get("person_ids", []) or []):
                filtered.append(item)
                continue
            if person_id and person_id in str(item.get("content", "") or ""):
                filtered.append(item)
        return filtered or hits

    @staticmethod
    def _episode_participants(rows: Sequence[Dict[str, Any]]) -> List[str]:
        seen = set()
        result: List[str] = []
        for row in rows:
            meta = row.get("metadata", {}) or {}
            for key in ("participants", "person_ids"):
                for item in meta.get(key, []) or []:
                    token = str(item or "").strip()
                    if not token or token in seen:
                        continue
                    seen.add(token)
                    result.append(token)
        return result[:16]

    @staticmethod
    def _episode_keywords(rows: Sequence[Dict[str, Any]]) -> List[str]:
        seen = set()
        result: List[str] = []
        for row in rows:
            meta = row.get("metadata", {}) or {}
            for item in meta.get("tags", []) or []:
                token = str(item or "").strip()
                if not token or token in seen:
                    continue
                seen.add(token)
                result.append(token)
        return result[:12]

    @staticmethod
    def _time_bound(rows: Sequence[Dict[str, Any]], primary: str, fallback: str, reverse: bool) -> Optional[float]:
        values: List[float] = []
        for row in rows:
            for key in (primary, fallback):
                value = row.get(key)
                try:
                    if value is not None:
                        values.append(float(value))
                        break
                except Exception:
                    continue
        if not values:
            return None
        return max(values) if reverse else min(values)

    def _resolve_relation_hashes(self, target: str) -> List[str]:
        assert self.metadata_store
        token = str(target or "").strip()
        if not token:
            return []
        if len(token) == 64 and all(ch in "0123456789abcdef" for ch in token.lower()):
            return [token]
        hashes = self.metadata_store.search_relation_hashes_by_text(token, limit=10)
        if hashes:
            return hashes
        return [str(row.get("hash", "") or "") for row in self.metadata_store.get_relations(subject=token)[:10] if str(row.get("hash", "")).strip()]
