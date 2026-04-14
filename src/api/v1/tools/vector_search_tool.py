import os
from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row
from langchain_core.tools import tool

from src.core.db import similarity_search
from psycopg import connect

from psycopg.conninfo import make_conninfo
from langchain_core.tools import tool
import urllib.parse

load_dotenv(override=True)

def vector_search(query: str, k: int = 5, chunk_type: str | None = None) -> list[dict]:
    """
    Returns a raw list of chunk dictionaries for hybrid search fusion.
    """
    # 1. Fetch raw chunks from your core DB utility
    # This assumes similarity_search returns list[dict] with 'content', 'metadata', etc.
    chunks = similarity_search(query, k=k, chunk_type=chunk_type)

    # 2. Ensure each chunk has a consistent structure for the RRF map
    formatted_chunks = []
    for chunk in chunks:
<<<<<<< HEAD
        # We preserve all original fields (content, image_base64, metadata)
        # but ensure 'content' is easily accessible for the RRF key mapping
=======
>>>>>>> raghul
        formatted_chunks.append({
            "content": chunk.get("content", ""),
            "chunk_type": chunk.get("chunk_type"),
            "metadata": {
                "source": chunk.get("source_file"),
                "page": chunk.get("page_number"),
                "section": chunk.get("section"),
                "element_type": chunk.get("element_type"),
                "similarity": round(chunk.get("similarity", 0), 4)
            },
<<<<<<< HEAD
            "image_base64": chunk.get("image_base64"), # Preserved for the final LLM formatting
=======
            "image_path": chunk.get("image_path"),   # local filesystem path (None for text/table)
>>>>>>> raghul
            "page_number": chunk.get("page_number")
        })

    print(formatted_chunks)
    
    return formatted_chunks