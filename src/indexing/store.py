import logging
import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from ..config import load_indexing_config

load_dotenv()
logger = logging.getLogger(__name__)

_PAYLOAD_KEYS = (
    "chunk_id",
    "source",
    "file_type",
    "page_number",
    "page_start",
    "page_end",
    "chunk_index",
    "char_count",
    "embedding_model",
    "embedding_dim",
)

def _qdrant_client(cfg: dict | None = None) -> QdrantClient:
    c = {**load_indexing_config(), **(cfg or {})}
    url = os.getenv("QDRANT_URL") or c["qdrant_url"]
    api_key = os.getenv("QDRANT_API_KEY") or c.get("qdrant_api_key") or None
    if api_key:
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(url=url)

def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

def _build_payload(doc: Document) -> dict:
    meta = doc.metadata
    payload = {"text": doc.page_content}
    for key in _PAYLOAD_KEYS:
        if key in meta and meta[key] is not None:
            payload[key] = meta[key]
    if "chunk_id" not in payload:
        raise ValueError("Each chunk must have metadata['chunk_id'] before upsert")
    return payload

def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    *,
    recreate: bool = False,
) -> None:
    exists = client.collection_exists(collection_name)
    if exists and not recreate:
        info = client.get_collection(collection_name)
        size = info.config.params.vectors.size
        if size != vector_size:
            raise ValueError(
                f"Collection {collection_name} has vector size {size}, "
                f"expected {vector_size}. Use recreate=True or a new collection name."
            )
        logger.info("Collection %s already exists", collection_name)
        return
    
    if exists and recreate:
        client.delete_collection(collection_name)
        logger.warning("Deleted collection %s (recreate=True)", collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size,distance=Distance.COSINE)
    )

    for field in ("source", "chunk_id"):
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )
    logger.info("Created collection %s (dim=%d, cosine)", collection_name, vector_size)

def upsert_chunks(
    embedded: list[Document],
    *,
    cfg: dict | None = None,
    recreate: bool = False,
) -> int:
    c = {**load_indexing_config(), **(cfg or {})}
    collection = c["qdrant_collection"]
    vector_size = c["qdrant_vector_size"]
    batch_size = c["qdrant_batch_size"]

    if not embedded:
        logger.warning("No embedded chunks to upsert")
        return 0
    
    client = _qdrant_client(c)
    ensure_collection(client,collection,vector_size,recreate=recreate)

    points: list[PointStruct] = []
    for doc in embedded:
        vector = doc.metadata.get("embedding")
        if not vector:
            raise ValueError(f"Missing embedding for chunk {doc.metadata.get('chunk_id')}")
        if len(vector) != vector_size:
            raise ValueError(
                f"Vector dim {len(vector)} != {vector_size} "
                f"for chunk {doc.metadata.get('chunk_id')}"
            )

        chunk_id = doc.metadata["chunk_id"]
        points.append(
            PointStruct(
                id = _point_id(chunk_id),
                vector=vector,
                payload=_build_payload(doc),
            )
        )

    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=collection, points=batch)
        logger.info(
            "Upserted %d / %d points",
            min(start + batch_size, len(points)),
            len(points),
        )
    logger.info("Stored %d points in collection %s", len(points), collection)
    return len(points)

def search(
    query_vector: list[float],
    *,
    limit: int = 5,
    source_filter: str | None = None,
    cfg: dict | None = None,
):
    c = {**load_indexing_config(), **(cfg or {})}
    client = _qdrant_client(c)
    collection = c["qdrant_collection"]

    query_filter = None
    if source_filter:
        query_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
        )

    return client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    )