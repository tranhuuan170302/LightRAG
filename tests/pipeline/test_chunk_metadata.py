"""Custom chunk metadata survives ingestion and never replaces core fields."""

import asyncio

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.utils import EmbeddingFunc, Tokenizer
from lightrag.utils_pipeline import (
    doc_status_reset_metadata,
    doc_status_transition_metadata,
    normalize_chunk_metadata_batch,
)

pytestmark = pytest.mark.offline


class _Tokenizer:
    def encode(self, text):
        return list(text.encode())

    def decode(self, tokens):
        return bytes(tokens).decode()


async def _embed(texts, **kwargs):
    return np.ones((len(texts), 8))


async def _llm(*args, **kwargs):
    raise AssertionError("Chunk-only ingestion must not call the LLM")


def _rag(tmp_path):
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"metadata-{tmp_path.name}",
        llm_model_func=_llm,
        tokenizer=Tokenizer("test", _Tokenizer()),
        embedding_func=EmbeddingFunc(embedding_dim=8, func=_embed),
        chunk_token_size=20,
        chunk_overlap_token_size=0,
    )


def test_metadata_survives_restart_and_chunk_vector_round_trip(tmp_path):
    async def run():
        rag = _rag(tmp_path)
        await rag.initialize_storages()
        supplied = {
            "department": "engineering",
            "tags": ["internal"],
            "content": "custom",
        }
        expected = {**supplied, "tags": ["internal"]}
        try:
            await rag.apipeline_enqueue_documents(
                [
                    "First document has several chunks of content.",
                    "Second document content.",
                ],
                ids=["doc-first", "doc-second"],
                file_paths=["first.txt", "second.txt"],
                metadata=[supplied, {"department": "sales"}],
                process_options="F!",
            )
            supplied["tags"].append("changed-after-enqueue")
        finally:
            await rag.finalize_storages()

        rag = _rag(tmp_path)
        await rag.initialize_storages()
        try:
            await rag.apipeline_process_enqueue_documents()
            for doc_id, metadata, filename in (
                ("doc-first", expected, "first.txt"),
                ("doc-second", {"department": "sales"}, "second.txt"),
            ):
                status = await rag.doc_status.get_by_id(doc_id)
                assert DocStatus(status["status"]) is DocStatus.PROCESSED
                assert status["metadata"]["chunk_metadata"] == metadata
                assert status["chunks_list"]
                for chunk_id in status["chunks_list"]:
                    for store in (rag.text_chunks, rag.chunks_vdb):
                        row = await store.get_by_id(chunk_id)
                        assert row["metadata"] == metadata
                        assert row["full_doc_id"] == doc_id
                        assert row["file_path"] == filename
                        assert row["content"] != "custom"
            assert "metadata" not in rag.entities_vdb.meta_fields
            assert "metadata" not in rag.relationships_vdb.meta_fields
        finally:
            await rag.finalize_storages()

    asyncio.run(run())


def test_metadata_survives_failed_retry_reset_and_status_transitions():
    row = {"metadata": {"chunk_metadata": {"tags": ["a"]}, "process_start_time": 42}}
    reset = doc_status_reset_metadata(row)
    assert reset == {"chunk_metadata": {"tags": ["a"]}}
    assert doc_status_transition_metadata({"metadata": reset}) == reset


def test_broadcast_metadata_is_detached_per_document():
    source = {"tags": ["a"], "nested": {"active": True, "value": None}}
    result = normalize_chunk_metadata_batch(source, 2)
    result[0]["tags"].append("b")
    assert result[1] == source


@pytest.mark.parametrize(
    "metadata", [[], [{}, {}], "bad", {"x": float("nan")}, {"x": object()}, {1: "bad"}]
)
def test_invalid_metadata_rejected_before_storage_writes(tmp_path, metadata):
    async def run():
        rag = _rag(tmp_path)
        await rag.initialize_storages()
        try:
            with pytest.raises(ValueError):
                await rag.apipeline_enqueue_documents(
                    "body", ids=["doc-bad"], metadata=metadata
                )
            assert await rag.full_docs.get_by_id("doc-bad") is None
            assert await rag.doc_status.get_by_id("doc-bad") is None
        finally:
            await rag.finalize_storages()

    asyncio.run(run())
