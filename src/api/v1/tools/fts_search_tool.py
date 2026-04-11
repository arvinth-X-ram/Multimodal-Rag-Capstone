import os
from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row
from langchain_core.tools import tool

load_dotenv()
# Clean up the connection string for psycopg
_raw_conn = os.getenv("PG_CONNECTION_STRING", "").replace("postgresql+psycopg", "postgresql")

def fts_search(query: str, k: int = 5) -> list[dict]:
    sql = """
        SELECT
            mc.content,
            mc.chunk_type,
            mc.metadata,
            mc.page_number,
            mc.image_path,
            d.filename AS source_document,
            ts_rank(
                to_tsvector('english', mc.content),
                websearch_to_tsquery('english', %(query)s)
            ) AS fts_rank
        FROM multimodal_chunks mc
        JOIN documents d ON mc.doc_id = d.id
        WHERE to_tsvector('english', mc.content) 
              @@ websearch_to_tsquery('english', %(query)s)
        ORDER BY fts_rank DESC
        LIMIT %(k)s;
    """
    
    results = []
    try:
        with psycopg.connect(_raw_conn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"query": query, "k": k})
                rows = cur.fetchall()

                for row in rows:
                    # Combine native columns and the metadata JSONB for a clean output
                    result_metadata = row["metadata"] if row["metadata"] else {}
                    result_metadata.update({
                        "source": row["source_document"],
                        "page": row["page_number"],
                        "type": row["chunk_type"],
                        "image_path": row["image_path"]
                    })

                    results.append({
                        "content": row["content"],
                        "metadata": result_metadata,
                        "fts_rank": round(float(row["fts_rank"]), 4),
                    })
    except Exception as e:
        print(f"Database error: {e}")
        
    return results