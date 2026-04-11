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