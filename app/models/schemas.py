from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    text: str
    index: int
    token_count: int
    doc_id: str


class RetrievedChunk(BaseModel):
    doc_id: str
    chunk_index: int
    score: float
    text: str
    source_file: str


class ComplexityResult(BaseModel):
    score: float
    tier: Literal["simple", "complex"]
    signals: list[str]


class CostMetadata(BaseModel):
    model_used: str
    tokens_in: int
    tokens_out: int
    estimated_cost_usd: float
    cache_hit: bool
    latency_ms: float


class CacheResult(BaseModel):
    answer: str
    cost_metadata: CostMetadata
    cache_hit: bool = True


class GeneratorResponse(BaseModel):
    answer: str
    cited_chunk_ids: list[str]
    tokens_in: int
    tokens_out: int


class IngestResponse(BaseModel):
    request_id: str
    doc_id: str
    status: str


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    cited_chunks: list[RetrievedChunk]
    cost_metadata: CostMetadata
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    mongodb: str
    redis: str


class MetricsResponse(BaseModel):
    total_queries: int
    cache_hit_rate: float
    avg_cost_usd: float
    total_cost_usd: float
    simple_query_count: int
    complex_query_count: int
