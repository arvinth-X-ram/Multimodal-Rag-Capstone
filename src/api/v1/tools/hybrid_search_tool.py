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
