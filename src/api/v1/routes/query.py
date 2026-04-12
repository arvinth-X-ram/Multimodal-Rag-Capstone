from fastapi import APIRouter, HTTPException
from src.api.v1.services.query_service import query_documents
from src.api.v1.schemas.query_schema import QueryRequest, QueryResponse
from src.api.v1.tools.vector_search_tool import vector_search

router = APIRouter(prefix="/query",tags=["Query"])


@router.post("/")
def query_endpoint(request: QueryRequest):
 
    try:
        q = query_documents(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return q
