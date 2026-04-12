1.Use the below for db creation:

-- =============================================================================
-- Multimodal RAG System — Database Schema
-- Issue 13 fix: Provides a single source of truth for the DB structure so
--               developers can set up the database without reverse-engineering
--               the application code.
--
-- Usage:
--   psql -U <user> -d <database> -f schema.sql
-- =============================================================================

-- Enable pgvector extension (must be done once per database by a superuser)
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- documents
-- Tracks every source PDF that has been ingested. The unique constraint on
-- filename makes upsert_document() idempotent: re-ingesting the same file
-- updates the path/timestamp without creating a new record.
-- =============================================================================
CREATE TABLE IF NOT EXISTS documents (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    filename    TEXT        UNIQUE NOT NULL,
    source_path TEXT        NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =============================================================================
-- multimodal_chunks
-- One row per searchable unit of content extracted from a document.
-- chunk_type is one of: 'text', 'table', 'image'.
--
-- Issue 18 fix: Images are stored on the filesystem; only the path is kept
--               here. This avoids bloating PostgreSQL with large BYTEA columns.
--
-- Issue 10 fix: Metadata fields that have dedicated columns (page_number,
--               section, source_file, element_type, position) are NOT
--               duplicated inside the JSONB `metadata` column.
-- =============================================================================
CREATE TABLE IF NOT EXISTS multimodal_chunks (
    id           BIGSERIAL    PRIMARY KEY,
    doc_id       UUID         NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_type   TEXT         NOT NULL CHECK (chunk_type IN ('text', 'table', 'image')),
    element_type TEXT,                    -- raw Docling label, e.g. 'section_header'
    content      TEXT         NOT NULL,
    image_path   TEXT,
    mime_type    TEXT,
    page_number  INT,
    section      TEXT,
    source_file  TEXT,

    position     JSONB,

    embedding    VECTOR(1536),
    metadata     JSONB
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- HNSW index for fast approximate nearest-neighbour cosine search.
-- Build after bulk-loading data for best performance.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON multimodal_chunks USING hnsw (embedding vector_cosine_ops);

-- B-tree indexes to accelerate filtered queries (chunk_type, doc_id, page).
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
    ON multimodal_chunks (doc_id);

CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type
    ON multimodal_chunks (chunk_type);

CREATE INDEX IF NOT EXISTS idx_chunks_page_number
    ON multimodal_chunks (page_number);

2. Ingest once more to save the image paths to the data folder
