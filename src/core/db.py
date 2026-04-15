import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from google import genai
from google.genai.types import EmbedContentConfig
import base64
import hashlib
import json
import os
import pathlib
from langchain_community.utilities import SQLDatabase
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
_EMBED_BATCH_SIZE = 50


load_dotenv(override=True)

PG_CONNECTION = os.getenv("PG_CONNECTION_STRING")
if not PG_CONNECTION:
    raise ValueError("PG_CONNECTION_STRING is not set")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set")

_PG_CONNECTION = os.getenv("PG_CONNECTION_STRING", "")
_PG_DSN = _PG_CONNECTION.replace("postgresql+psycopg://", "postgresql://")


class Gemini1536Embeddings:
    """Forces exactly 1536 dimensions using official Google GenAI SDK"""
    
    def __init__(self):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model = "gemini-embedding-2-preview"
        self.dim = 1536

    def embed_query(self, text: str) -> list[float]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=EmbedContentConfig(
                output_dimensionality=self.dim,
                task_type="RETRIEVAL_QUERY"
            )
        )
        return result.embeddings[0].values

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=texts,   # batch support
            config=EmbedContentConfig(
                output_dimensionality=self.dim,
                task_type="RETRIEVAL_DOCUMENT"
            )
        )
        return [embedding.values for embedding in result.embeddings]


def get_embedding_model():
    return Gemini1536Embeddings()

embeddings_model=get_embedding_model()


def get_vector_store(collection_name: str = "policy_docs"):
    return PGVector(
        collection_name=collection_name,
        connection=PG_CONNECTION,
        embeddings=get_embedding_model(),
        use_jsonb=True,
    )

_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    """Return the module-level connection pool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            _PG_DSN,
            min_size=2,
            max_size=10,
            kwargs={"row_factory": dict_row},
        )
    return _pool


def get_db_conn():
    """Return a pooled connection context manager.

    Usage:
        with get_db_conn() as conn:
            with conn.cursor() as cur: ...
    """
    return _get_pool().connection()


# ---------------------------------------------------------------------------
# Document registry
# ---------------------------------------------------------------------------

def upsert_document(filename: str, source_path: str) -> str:
    """Insert a document record and return its UUID.

    Uses ON CONFLICT so re-ingesting the same filename updates the path
    and returns the *existing* doc_id rather than creating a duplicate.
    This makes ingestion idempotent at the document level.
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (filename, source_path)
                VALUES (%s, %s)
                ON CONFLICT (filename) DO UPDATE
                    SET source_path = EXCLUDED.source_path,
                        ingested_at  = now()
                RETURNING id
                """,
                (filename, source_path),
            )
            row = cur.fetchone()
        conn.commit()
    return str(row["id"])


# ---------------------------------------------------------------------------
# Chunk storage
# ---------------------------------------------------------------------------

def store_chunks(chunks: list[dict], doc_id: str) -> int:
    """Embed each chunk and insert it into the multimodal_chunks table.

    Args:
        chunks:  List of dicts produced by parse_document() / ingestion.py.
                 Each dict must have: content (str), content_type (str),
                 metadata (dict with page_number, section, source_file,
                 element_type, position, image_base64).
        doc_id:  UUID string of the parent document (from upsert_document).

    Returns:
        Number of rows inserted.

    Embedding strategy:
        Texts are embedded in batches of _EMBED_BATCH_SIZE to minimise
        API round-trips. embed_documents() takes a list and returns a
        list of 768-dimensional float vectors in the same order.

    Vector storage:
        pgvector accepts the '[f1,f2,…]' string literal when cast with
        ::vector. We build that string directly to avoid needing the
        separate pgvector Python package.

    Image storage:
        image_base64 from metadata is decoded to raw bytes and stored in
        the BYTEA column. The JSONB metadata column does NOT duplicate it,
        keeping metadata lean.
    """
    if not chunks:
        return 0

    contents = [c["content"] for c in chunks]

    # ── Batch embed all chunks ────────────────────────────────────────────────
    all_embeddings: list[list[float]] = []
    for i in range(0, len(contents), _EMBED_BATCH_SIZE):
        batch = contents[i : i + _EMBED_BATCH_SIZE]
        all_embeddings.extend(embeddings_model.embed_documents(batch))  # Issue 8

    # ── Insert rows ───────────────────────────────────────────────────────────
    # Issue 10 fix: Only store fields in JSONB that don't already have a
    # dedicated column — the rest are redundant and waste storage.
    _DEDICATED_COLUMNS = {
        "content_type", "element_type", "section",
        "page_number", "source_file", "position", "image_base64",
    }

    rows_inserted = 0
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            # Issue 4 fix: Delete stale chunks before re-inserting so that
            # re-ingesting the same document does not create duplicates.
            cur.execute(
                "DELETE FROM multimodal_chunks WHERE doc_id = %s::uuid",
                (doc_id,),
            )

            for chunk, embedding in zip(chunks, all_embeddings):
                meta = chunk["metadata"]

                # Issue 18 fix: Save image bytes to the filesystem and store
                # only the file path in the DB. This avoids bloating PostgreSQL
                # with large BYTEA columns that slow down vacuuming and queries.
                img_b64 = meta.get("image_base64")
                image_path: str | None = None
                mime_type = "image/png" if img_b64 else None
                if img_b64:
                    image_bytes = base64.b64decode(img_b64)
                    img_dir = pathlib.Path("data/images")
                    img_dir.mkdir(parents=True, exist_ok=True)
                    img_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
                    img_file = img_dir / f"{doc_id}_{img_hash}.png"
                    img_file.write_bytes(image_bytes)
                    image_path = str(img_file)

                # pgvector vector literal: '[0.1, 0.2, …]'
                embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

                # Exclude fields that already have dedicated columns from JSONB.
                clean_meta = {k: v for k, v in meta.items() if k not in _DEDICATED_COLUMNS}

                cur.execute(
                    """
                    INSERT INTO multimodal_chunks (
                        doc_id, chunk_type, element_type, content,
                        image_path, mime_type,
                        page_number, section, source_file,
                        position, embedding, metadata
                    ) VALUES (
                        %s::uuid, %s, %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s::jsonb, %s::vector, %s::jsonb
                    )
                    """,
                    (
                        doc_id,
                        chunk["content_type"],       # chunk_type column
                        meta.get("element_type"),    # raw Docling label
                        chunk["content"],            # text / markdown / caption
                        image_path,                  # filesystem path (None for text/table)
                        mime_type,
                        meta.get("page_number"),
                        meta.get("section"),
                        meta.get("source_file"),
                        json.dumps(meta.get("position")) if meta.get("position") else None,
                        embedding_str,               # ::vector cast
                        json.dumps(clean_meta),      # JSONB catch-all
                    ),
                )
                rows_inserted += 1
        conn.commit()

    return rows_inserted


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------

def similarity_search(
    query: str,
    k: int = 5,
    chunk_type: str | None = None,
) -> list[dict]:
    """Find the k most similar chunks to a natural-language query.

    Args:
        query:      Natural-language question or search string.
        k:          Number of results to return.
        chunk_type: Optional filter — 'text', 'table', or 'image'.

    Returns:
        List of dicts with keys: content, chunk_type, page_number, section,
        source_file, element_type, image_base64, mime_type, position,
        metadata, similarity (0–1 cosine similarity score).

    The <=> operator is pgvector's cosine distance operator.
    Similarity = 1 − cosine_distance, so 1.0 = identical, 0.0 = orthogonal.
    """
    query_vec = embeddings_model.embed_query(query)  # Issue 8: use singleton
    embedding_str = "[" + ",".join(str(v) for v in query_vec) + "]"

    # Conditionally add a chunk_type filter without SQL injection risk
    # (chunk_type is always passed as a parameterised value, never interpolated)
    type_clause = "AND chunk_type = %(chunk_type)s" if chunk_type else ""

    sql = f"""
        SELECT
            content, chunk_type, page_number, section,
            source_file, element_type, image_path, mime_type,
            position, metadata,
            1 - (embedding <=> %(vec)s::vector) AS similarity
        FROM multimodal_chunks
        WHERE 1=1 {type_clause}
        ORDER BY embedding <=> %(vec)s::vector
        LIMIT %(k)s
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"vec": embedding_str, "chunk_type": chunk_type, "k": k})
            rows = cur.fetchall()

    # Return image_path as-is; callers open the file from disk if needed.
    results = []
    for row in rows:
        row = dict(row)
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Chunk listing (for preview / debugging)
# ---------------------------------------------------------------------------

def get_all_chunks(chunk_type: str | None = None, limit: int = 200) -> list[dict]:
    """Return all stored chunks, optionally filtered by type.

    Args:
        chunk_type: Optional filter — 'text', 'table', or 'image'.
        limit:      Max rows to return (default 200, safety cap).

    Returns:
        List of dicts with keys: id, content, chunk_type, page_number,
        section, source_file, element_type, image_base64, mime_type,
        position, metadata.
    """
    type_clause = "WHERE chunk_type = %(chunk_type)s" if chunk_type else ""

    sql = f"""
        SELECT
            id, content, chunk_type, page_number, section,
            source_file, element_type, image_path, mime_type,
            position, metadata
        FROM multimodal_chunks
        {type_clause}
        ORDER BY page_number ASC NULLS LAST, id ASC
        LIMIT %(limit)s
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"chunk_type": chunk_type, "limit": limit})
            rows = cur.fetchall()

    results = []
    for row in rows:
        row = dict(row)
        results.append(row)

    return results

def get_sql_database() -> SQLDatabase:
    """Return a LangChain SQLDatabase connected to the agentic_rag_db (read-only).

    Uses the rag_readonly role from sql/seed.sql — SELECT privileges only.
    Connection string is read from AGENTIC_RAG_DB_URL in the environment.
    """
    db_url = os.getenv("AGENTIC_RAG_DB_URL")
    if not db_url:
        raise ValueError("AGENTIC_RAG_DB_URL is not set. Check your .env file.")
    return SQLDatabase.from_uri(
        db_url,
        include_tables=["billing_statements","card_transactions","credit_cards","customers","reward_transactions"],
        sample_rows_in_table_info=2,
    )