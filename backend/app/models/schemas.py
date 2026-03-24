"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# Request Models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)

    class Config:
        json_schema_extra = {
            "example": {"question": "What is the revenue for Q3?", "k": 5}
        }


class TextIngestionRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_name: str = Field(default="user_input")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Machine learning is a subset of AI...",
                "source_name": "ml_notes.txt"
            }
        }


# Response Models
class SourceMetadata(BaseModel):
    source: str
    page: Optional[Any] = None       # ✅ Any — handles int or string page numbers
    chunk_id: Optional[Any] = None   # ✅ Any — handles various chunk id types


class Source(BaseModel):
    id: int
    content: str
    metadata: SourceMetadata


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]    # ✅ flexible — no strict nested type validation
    confidence_score: float = Field(ge=0.0, le=1.0)
    retrieval_time: float
    generation_time: float
    total_time: float
    timestamp: datetime = Field(default_factory=datetime.now)


class IngestionResponse(BaseModel):
    message: str
    num_documents: int
    num_chunks: int
    ingestion_time: float
    timestamp: datetime = Field(default_factory=datetime.now)


class StatsResponse(BaseModel):
    total_documents: int
    embedding_dimension: int
    app_version: str


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)