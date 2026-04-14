<<<<<<< HEAD
import os
from fastapi import APIRouter, UploadFile, File
from src.api.v1.services.query_services import query_documents
from src.api.v1.schema.query_schema import QueryRequest,QueryResponse

# Import your ingestion and query utilities
from src.ingestion.ingestion import run_ingestion
router = APIRouter()

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --- Upload Endpoint ---
@router.post("/admin/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Call ingestion pipeline (chunking + embedding into PGVector)
    run_ingestion(file_path)

    return {"file": file.filename, "message": "Upload and embedding successful"}

@router.post("/query")
def query_endpoint(request: QueryRequest):
    print(f"Received query: {request.query}")
    result = query_documents(request.query)  # Issue 17
    return result
=======
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
>>>>>>> raghul
