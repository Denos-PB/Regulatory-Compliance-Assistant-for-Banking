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
    Fusion,
    FusionQuery,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    SparseVectorParams,
    VectorParams,
)
from ..config import load_indexing_config, load_rag_config
from .sparse import embed_sparse_documents
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

def _hybrid_enabled() -> bool:
    return bool(load_rag_config().get("hybrid_enabled"))


def _build_filter(source_filter: str | None) -> Filter | None:
    if not source_filter:
        return None
    return Filter(
        must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
    )

def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    *,
    recreate: bool = False,
    hybrid: bool = False,
    dense_name: str = "dense",
    sparse_name: str = "sparse",
) -> None:
    exists = client.collection_exists(collection_name)

    if exists and not recreate:
        info = client.get_collection(collection_name)
        vectors = info.config.params.vectors

        if hybrid:
            if not isinstance(vectors, dict) or dense_name not in vectors:
                raise ValueError(
                    f"Collection {collection_name} is not hybrid "
                    f"(missing vector '{dense_name}'). Run: cli run --recreate"
                )
            size = vectors[dense_name].size
            sparse_cfg = info.config.params.sparse_vectors
            if not sparse_cfg or sparse_name not in sparse_cfg:
                raise ValueError(
                    f"Collection {collection_name} has no sparse vector '{sparse_name}'. "
                    "Run: cli run --recreate"
                )
        else:
            if isinstance(vectors, dict):
                if dense_name in vectors:
                    size = vectors[dense_name].size
                else:
                    raise ValueError("Hybrid collection exists but hybrid=False in config.")
            else:
                size = vectors.size

        if size != vector_size:
            raise ValueError(
                f"Collection {collection_name} has vector size {size}, "
                f"expected {vector_size}. Use recreate=True."
            )
        logger.info("Collection %s already exists", collection_name)
        return

    if exists and recreate:
        client.delete_collection(collection_name)
        logger.warning("Deleted collection %s (recreate=True)", collection_name)

    if hybrid:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                dense_name: VectorParams(size=vector_size, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                sparse_name: SparseVectorParams(),
            },
        )
        logger.info(
            "Created hybrid collection %s (dense=%s, sparse=%s, dim=%d)",
            collection_name, dense_name, sparse_name, vector_size,
        )
    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Created collection %s (dim=%d, cosine)", collection_name, vector_size)

    for field in ("source", "chunk_id"):
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )

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
    dense_name = c["dense_vector_name"]
    sparse_name = c["sparse_vector_name"]
    hybrid = _hybrid_enabled()

    if not embedded:
        logger.warning("No embedded chunks to upsert")
        return 0

    client = _qdrant_client(c)
    ensure_collection(
        client, collection, vector_size,
        recreate=recreate, hybrid=hybrid,
        dense_name=dense_name, sparse_name=sparse_name,
    )

    sparse_vectors = []
    if hybrid:
        texts = [doc.page_content for doc in embedded]
        sparse_vectors = embed_sparse_documents(texts, cfg=c)
        if len(sparse_vectors) != len(embedded):
            raise RuntimeError("Sparse embedding count mismatch")

    points: list[PointStruct] = []
    for i, doc in enumerate(embedded):
        dense = doc.metadata.get("embedding")
        if not dense:
            raise ValueError(f"Missing embedding for {doc.metadata.get('chunk_id')}")
        if len(dense) != vector_size:
            raise ValueError(f"Vector dim mismatch for {doc.metadata.get('chunk_id')}")

        if hybrid:
            vector = {
                dense_name: dense,
                sparse_name: sparse_vectors[i],
            }
        else:
            vector = dense

        points.append(
            PointStruct(
                id=_point_id(doc.metadata["chunk_id"]),
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
    dense_name = c["dense_vector_name"]
    hybrid = _hybrid_enabled()

    kwargs = {
        "collection_name": collection,
        "query": query_vector,
        "limit": limit,
        "query_filter": _build_filter(source_filter),
        "with_payload": True,
    }
    if hybrid:
        kwargs["using"] = dense_name

    return client.query_points(**kwargs)

def search_hybrid(
    dense_vector: list[float],
    sparse_vector,
    *,
    limit: int = 5,
    prefetch_limit: int = 20,
    source_filter: str | None = None,
    cfg: dict | None = None,
):
    c = {**load_indexing_config(), **(cfg or {})}
    rag = load_rag_config()
    client = _qdrant_client(c)
    collection = c["qdrant_collection"]
    dense_name = c["dense_vector_name"]
    sparse_name = c["sparse_vector_name"]
    prefetch_limit = prefetch_limit or rag.get("hybrid_prefetch_limit", 20)

    return client.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(
                query=dense_vector,
                using=dense_name,
                limit=prefetch_limit,
                filter=_build_filter(source_filter),
            ),
            Prefetch(
                query=sparse_vector,
                using=sparse_name,
                limit=prefetch_limit,
                filter=_build_filter(source_filter),
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        query_filter=_build_filter(source_filter),
        with_payload=True,
    )

def list_sources(*, cfg: dict | None = None, page_size: int = 256) -> list[str]:
    c = {**load_indexing_config(), **(cfg or {})}
    client = _qdrant_client(c)
    collection = c["qdrant_collection"]
    seen: set[str] = set()
    sources: list[str] = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=page_size,
            offset=offset,
            with_payload=["source"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            src = payload.get("source")
            if src and src not in seen:
                seen.add(src)
                sources.append(src)
        if offset is None:
            break

    return sorted(sources)