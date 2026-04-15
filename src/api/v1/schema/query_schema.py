from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
   query: str = Field(..., example="What is the minimum CIBIL score for personal loans?")


class QueryResponse(BaseModel):
   query: str
   answer: str
   policy_citations: str
   page_no: str
   document_name: str
   image_path: Optional[str] = Field(
       default=None,
       description="Local filesystem path to the image file when the answer involves a visual chunk (chart/figure). None for text-only answers.",
   )


class AIResponse(BaseModel):
   query: str = Field(description="The Given query by user must be present here")
   answer: str = Field(description="The generated response")
   policy_citations: str = Field(description="Give the Policy Citation")
   page_no: str = Field(description="The page number in the metadata")
   document_name: str = Field(description="Name of the document used")
   image_path: Optional[str] = Field(
       default=None,
       description="Local filesystem path to the image. Leave null — the pipeline injects this field automatically after generation.",
   )
