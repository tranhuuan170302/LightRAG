"""Request-scoped ontology is carried without becoming persisted document data."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline


class _TokenizerImpl:
    def encode(self, content: str):
        return [ord(char) for char in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


async def _embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=np.float32)


async def _llm(prompt, **kwargs):
    return ""


def test_request_ontology_is_transient_and_available_to_extraction(tmp_path):
    async def _run():
        rag = LightRAG(
            working_dir=str(tmp_path),
            workspace="request-ontology",
            llm_model_func=_llm,
            entity_extraction_use_json=False,
            embedding_func=EmbeddingFunc(
                embedding_dim=8, max_token_size=4096, func=_embedding
            ),
            tokenizer=Tokenizer("request-ontology", _TokenizerImpl()),
        )
        await rag.initialize_storages()
        try:
            ontology = {
                "entity_types_guidance": "Use domain-specific types.",
                "entity_extraction_examples": ["custom text example"],
            }
            await rag.apipeline_enqueue_documents(
                "Alice works at Acme.",
                ids=["ontology-doc"],
                file_paths="ontology.txt",
                ontology=ontology,
            )

            full_doc = await rag.full_docs.get_by_id("ontology-doc")
            assert "ontology" not in full_doc
            assert "_request_entity_extraction_profiles" not in rag._build_global_config()

            profile = rag._pop_request_entity_extraction_profile("ontology-doc")
            assert profile["entity_types_guidance"] == ontology["entity_types_guidance"]
            assert profile["entity_extraction_examples"] == ontology[
                "entity_extraction_examples"
            ]
            assert rag._pop_request_entity_extraction_profile("ontology-doc") is None
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
