"""Request-scoped collection isolation for the HTTP API."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import fields, replace
from typing import Any

from fastapi import HTTPException, Request, status

from lightrag.utils import validate_workspace


COLLECTION_NAME_MAX_LENGTH = 256


def normalize_collection_name(value: str) -> str:
    """Validate a client collection name without rewriting it."""
    if not isinstance(value, str):
        raise ValueError("collection_name must be a string")
    value = value.strip()
    if not value:
        raise ValueError("collection_name must not be empty")
    if len(value) > COLLECTION_NAME_MAX_LENGTH:
        raise ValueError(
            f"collection_name must be at most {COLLECTION_NAME_MAX_LENGTH} characters"
        )
    return value


def collection_workspace(user_id: str, collection_name: str) -> str:
    """Return a filesystem-safe, collision-resistant tenant workspace.

    The original values are deliberately not sanitized into a workspace name:
    sanitization can make distinct user/collection pairs collide. A digest of
    the length-delimited composite key preserves isolation while satisfying the
    storage backends' single-path-component requirement.
    """
    user_id = str(user_id or "").strip()
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user context is required",
        )
    collection_name = normalize_collection_name(collection_name)
    raw_key = f"{len(user_id)}:{user_id}{len(collection_name)}:{collection_name}"
    workspace = "api_" + hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    return validate_workspace(workspace)


def _unwrap_priority_wrapper(value: Any) -> Any:
    """Recover the provider callable before LightRAG's queue wrapper."""
    while callable(value) and hasattr(value, "__wrapped__"):
        value = value.__wrapped__
    return value


def clone_rag_for_workspace(base_rag: Any, workspace: str) -> Any:
    """Construct a LightRAG instance with the base instance's configuration."""
    kwargs: dict[str, Any] = {}
    for item in fields(base_rag):
        if not item.init or item.name.startswith("_") or item.name == "workspace":
            continue
        if not hasattr(base_rag, item.name):
            continue
        value = getattr(base_rag, item.name)
        if item.name == "embedding_func" and value is not None:
            value = replace(value, func=_unwrap_priority_wrapper(value.func))
        elif item.name == "rerank_model_func" and value is not None:
            value = _unwrap_priority_wrapper(value)
        elif item.name in {"role_llm_configs", "vector_db_storage_cls_kwargs"}:
            value = deepcopy(value)
        kwargs[item.name] = value

    kwargs["workspace"] = workspace
    kwargs["addon_params"] = deepcopy(dict(base_rag.addon_params))
    return type(base_rag)(**kwargs)


class CollectionRAGManager:
    """Lazily initialize one LightRAG instance per user/collection pair."""

    def __init__(
        self,
        rag_factory: Callable[[str], Any],
        configure_rag: Callable[[Any], None] | None = None,
    ) -> None:
        self._rag_factory = rag_factory
        self._configure_rag = configure_rag
        self._rags: dict[tuple[str, str], Any] = {}
        self._lock = asyncio.Lock()

    async def get(self, user_id: str, collection_name: str) -> Any:
        collection_name = normalize_collection_name(collection_name)
        user_id = str(user_id or "").strip()
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authenticated user context is required",
            )
        key = (user_id, collection_name)
        rag = self._rags.get(key)
        if rag is not None:
            return rag

        async with self._lock:
            rag = self._rags.get(key)
            if rag is None:
                rag = self._rag_factory(collection_workspace(*key))
                if self._configure_rag is not None:
                    self._configure_rag(rag)
                try:
                    await rag.initialize_storages()
                except Exception:
                    # Do not retain a partially initialized instance. The
                    # caller can retry after the underlying storage recovers.
                    try:
                        await rag.finalize_storages()
                    except Exception:
                        pass
                    raise
                self._rags[key] = rag
        return rag

    async def close(self) -> None:
        """Finalize all lazily created instances during application shutdown."""
        async with self._lock:
            rags = list(self._rags.values())
            self._rags.clear()
        results = await asyncio.gather(
            *(rag.finalize_storages() for rag in rags), return_exceptions=True
        )
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            raise errors[0]


async def resolve_collection_rag(
    request: Request,
    collection_name: str | None,
    resolver: Callable[[str, str], Awaitable[Any]] | None,
    fallback_rag: Any,
) -> Any:
    """Resolve the request's isolated RAG, retaining direct-test compatibility."""
    if resolver is None:
        return fallback_rag
    try:
        collection_name = normalize_collection_name(collection_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    user_id = getattr(request.state, "user_id", None)
    return await resolver(user_id, collection_name)
