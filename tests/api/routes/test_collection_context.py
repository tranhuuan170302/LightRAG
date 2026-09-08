import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from lightrag.api.collection_context import (
    CollectionRAGManager,
    collection_workspace,
    resolve_collection_rag,
)
from lightrag.api.routers.document_routes import (
    CollectionInsertTextsRequest,
    InsertTextsRequest,
    OntologyRequest,
    _normalize_ontology_request,
    _parse_upload_ontology,
)
from lightrag.api.routers.query_routes import CollectionQueryRequest, QueryRequest
from pydantic import ValidationError


def test_collection_name_can_be_omitted_only_by_direct_sdk_compatibility_models():
    assert InsertTextsRequest(texts=["text"]).collection_name is None
    assert QueryRequest(query="question").collection_name is None

    with pytest.raises(ValidationError):
        CollectionInsertTextsRequest(texts=["text"])
    with pytest.raises(ValidationError):
        CollectionQueryRequest(query="question")


def test_query_http_schema_requires_collection_name():
    from lightrag.api.routers.query_routes import create_query_routes

    app = FastAPI()
    app.include_router(create_query_routes(object()))
    client = TestClient(app)

    response = client.post("/query", json={"query": "question"})
    assert response.status_code == 422
    assert "collection_name" in response.text
    assert "collection_name" in app.openapi()["components"]["schemas"][
        "CollectionQueryRequest"
    ]["required"]


def test_request_ontology_requires_examples_for_active_mode():
    ontology = OntologyRequest(
        entity_types_guidance="Use domain types.",
        entity_extraction_examples=["text example"],
    )
    rag = type("RAG", (), {"entity_extraction_use_json": False})()
    normalized = _normalize_ontology_request(ontology, rag)
    assert normalized["entity_types_guidance"] == "Use domain types."
    assert normalized["entity_extraction_examples"] == ["text example"]

    json_rag = type("RAG", (), {"entity_extraction_use_json": True})()
    with pytest.raises(HTTPException) as error:
        _normalize_ontology_request(ontology, json_rag)
    assert error.value.status_code == 422
    assert "entity_extraction_json_examples" in str(error.value.detail)

    json_ontology = OntologyRequest(
        entity_types_guidance="Use JSON domain types.",
        entity_extraction_json_examples=['{"entities": [], "relationships": []}'],
    )
    normalized_json = _normalize_ontology_request(json_ontology, json_rag)
    assert normalized_json["entity_extraction_json_examples"] == [
        '{"entities": [], "relationships": []}'
    ]


def test_upload_ontology_is_json_string_and_is_not_required():
    rag = type("RAG", (), {"entity_extraction_use_json": False})()
    assert _parse_upload_ontology(None, rag) is None

    parsed = _parse_upload_ontology(
        '{"entity_extraction_examples": ["custom example"]}', rag
    )
    assert parsed["entity_extraction_examples"] == ["custom example"]

    with pytest.raises(HTTPException) as error:
        _parse_upload_ontology("not-json", rag)
    assert error.value.status_code == 422


def test_collection_workspace_is_isolated_by_user_and_collection():
    assert collection_workspace("alice", "research") != collection_workspace(
        "bob", "research"
    )
    assert collection_workspace("alice", "research") != collection_workspace(
        "alice", "support"
    )
    assert collection_workspace("alice", "a/b") != collection_workspace(
        "alice", "a_b"
    )


@pytest.mark.asyncio
async def test_collection_rag_manager_reuses_only_exact_user_collection_pair():
    created = []

    class FakeRAG:
        def __init__(self, workspace):
            self.workspace = workspace
            self.initialized = 0
            self.finalized = 0
            created.append(self)

        async def initialize_storages(self):
            self.initialized += 1

        async def finalize_storages(self):
            self.finalized += 1

    manager = CollectionRAGManager(FakeRAG)
    alice = await manager.get("alice", "research")
    assert await manager.get("alice", "research") is alice
    assert await manager.get("bob", "research") is not alice
    assert len(created) == 2
    assert all(item.initialized == 1 for item in created)

    await manager.close()
    assert all(item.finalized == 1 for item in created)


@pytest.mark.asyncio
async def test_resolver_uses_authenticated_user_context():
    request = Request({"type": "http", "state": {}})
    request.state.user_id = "alice"
    calls = []

    async def resolver(user_id, collection_name):
        calls.append((user_id, collection_name))
        return "rag"

    assert await resolve_collection_rag(request, "research", resolver, "fallback") == "rag"
    assert calls == [("alice", "research")]

    with pytest.raises(HTTPException) as error:
        await resolve_collection_rag(request, None, resolver, "fallback")
    assert error.value.status_code == 422
