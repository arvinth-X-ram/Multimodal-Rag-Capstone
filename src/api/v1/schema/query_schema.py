from pydantic import BaseModel, Field
from typing import List, Optional

# ---- Request ----
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
   #  k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
   #  chunk_type: Optional[str] = Field(
   #      None, description="Filter by content type: 'text', 'table', or 'image'"
   #  )

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

class AIResponse(BaseModel):
   query: str = Field(description="The Given query by user must be present here")
   answer: str = Field(description="The generated response")
   policy_citations: str = Field(description="Give the Policy Citation")
   page_no: str = Field(description="The page number in the metadata")
   document_name: str = Field(description="Name of the document used")


