# Custom chunk metadata

Supply an optional `metadata` JSON object when inserting documents. Each chunk
inherits that object under its own `metadata` field, alongside LightRAG's
default `full_doc_id`, `content`, and `file_path` fields. Custom fields do not
replace those defaults and are not added to the text sent for embedding.
Entity and relationship vector payloads and Neo4j properties are unaffected.

## Text API

`POST /documents/text`:

```json
{
  "text": "The maintenance procedure starts with an inspection.",
  "file_source": "maintenance.txt",
  "metadata": {
    "department": "engineering",
    "category": "manual",
    "tags": ["internal", "maintenance"],
    "revision": 3
  }
}
```

`POST /documents/texts` accepts either one shared metadata object or an array
of objects aligned with `texts`. An array must contain exactly one object per
text; use `{}` for a document without custom fields.

```json
{
  "collection_name": "manuals",
  "texts": ["Maintenance procedure.", "Equipment specifications."],
  "file_sources": ["maintenance.txt", "equipment.txt"],
  "metadata": [{"category": "procedure"}, {"category": "specification"}]
}
```

## File upload

`POST /documents/upload` accepts a `metadata` multipart form field containing
the serialized JSON object, alongside `file` and `collection_name`:

```bash
curl -X POST http://localhost:9621/documents/upload \
  -H 'X-API-Key: YOUR_API_KEY' \
  -F 'collection_name=manuals' \
  -F 'file=@maintenance.pdf' \
  -F 'metadata={"department":"engineering","category":"manual"}'
```

Malformed metadata is rejected with HTTP 422 before background indexing.
Values may be JSON strings, numbers, booleans, null, arrays, or nested objects;
non-finite numbers such as NaN are rejected.

## Python API

After `await rag.initialize_storages()`:

```python
await rag.ainsert(
    "Maintenance procedure.",
    file_paths="maintenance.txt",
    metadata={"category": "manual", "revision": 3},
)
```

The synchronous `rag.insert(...)` and `rag.apipeline_enqueue_documents(...)`
accept the same `metadata` argument, including a per-document list for batches.
Custom metadata is persisted in `doc_status.metadata.chunk_metadata` so parser
stages, service restarts, and manual retries preserve it. At chunk creation it
is copied into both the text-chunk KV payload and the chunk-vector payload.
If a custom chunking callback already supplies a chunk's `metadata` object,
its more specific keys override shared document keys.

## Storage and existing documents

Milvus stores the object as the dynamic `metadata` field on chunk rows. The
NanoVectorDB adapter also preserves it. Other vector adapters must support
JSON metadata and honor the configured `meta_fields`; adapters with fixed
schemas may require additional work. KV adapters with fixed chunk schemas may
similarly omit custom fields from their text-chunk rows.

This applies to newly ingested documents. Existing documents are not updated
automatically, and duplicate insertion does not update their metadata. The
existing delete-and-reinsert workflow can replace a document with new metadata.
Storage support does not add metadata filters to the LightRAG query API.
