<<<<<<< HEAD
# from src.core.db import get_vector_store 
from src.api.v1.tools.fts_search_tool import fts_search
from src.api.v1.tools.vector_search_tool import vector_search
from langchain_core.tools import tool

def _hybrid_search(query: str, k: int = 5) -> list[dict]:
#    vector_store = get_vector_store()
   vector_docs = vector_search(query, k=k)
   fts_docs    = fts_search(query, k=k)


   rrf_scores: dict[str, float] = {}
   chunk_map:  dict[str, dict]  = {}


   for rank, doc in enumerate(vector_docs):
        # This will now work because doc is a dictionary, not a message object
        key = doc["content"][:120] 
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)
        chunk_map[key]  = doc


   for rank, item in enumerate(fts_docs):
       key = item["content"][:120]
       rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)
       chunk_map[key]  = {"content": item["content"], "metadata": item["metadata"]}


   ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
   return [chunk_map[key] for key, _ in ranked[:k]]
=======
from src.api.v1.tools.fts_search_tool import fts_search
from src.api.v1.tools.vector_search_tool import vector_search


def hybrid_search(query: str, k: int = 10) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) of vector-search and full-text-search results.

    Each result list contributes a score of 1/(60 + rank+1) per document.
    Documents that appear in both lists accumulate scores from both sources,
    so genuinely relevant chunks bubble to the top regardless of search type.

    Returns at most `k` dicts with keys:
        content, chunk_type, metadata, image_base64, page_number
    """
    vector_docs = vector_search(query, k=k)
    fts_docs = fts_search(query, k=k)

    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    # --- Vector results ---
    for rank, doc in enumerate(vector_docs):
        key = doc["content"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)
        chunk_map[key] = {
            "content": doc["content"],
            "chunk_type": doc.get("chunk_type"),
            "metadata": doc.get("metadata", {}),
            "image_path": doc.get("image_path"),
            "page_number": doc.get("page_number"),
        }

    # --- FTS results ---
    for rank, item in enumerate(fts_docs):
        key = item["content"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)
        if key not in chunk_map:
            chunk_map[key] = {
                "content": item["content"],
                "chunk_type": item["metadata"].get("type"),
                "metadata": item["metadata"],
                "image_path": item["metadata"].get("image_path"),
                "page_number": item["metadata"].get("page"),
            }

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_map[key] for key, _ in ranked[:k]]
>>>>>>> raghul
