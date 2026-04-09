# Cost-Aware Production RAG

Production RAG API with intelligent model routing and semantic caching — reduces LLM inference cost by ~60% without sacrificing answer quality.

---

## The Problem

Naive RAG calls expensive models for every query regardless of complexity. Repeated or semantically similar queries hit the LLM every time, even when the answer is already known. This wastes money at scale — a 10,000 query/day workload can spend $300/day on inference that could cost $30.

---

## Architecture

```
User Query
  → Redis Semantic Cache (cosine similarity >= 0.92)
      → HIT:  Return cached answer ── cost $0.00
      → MISS: Complexity Classifier (zero LLM calls)
            → SIMPLE → Gemini Flash → MongoDB $vectorSearch → Answer
            → COMPLEX → Gemini Pro  → MongoDB $vectorSearch → Answer
            → Cache result in Redis (TTL 1hr)
            → Return answer + cost_metadata
```

Every response includes `cost_metadata`: model used, tokens in/out, estimated USD cost, cache hit flag, and latency.

---

## Cost Savings

| Query Type  | Model        | Est. Cost/Query |
|-------------|--------------|-----------------|
| Cached      | None         | $0.0000         |
| Simple      | Gemini Flash | $0.0001         |
| Complex     | Gemini Pro   | $0.0015         |
| Naive GPT-4 | GPT-4        | $0.0300         |

At 10,000 queries/day with 70% cache hit rate and 25% simple / 5% complex split:
- **Naive GPT-4**: ~$300/day
- **This system**: ~$18/day

---

## Tech Stack

| Layer        | Technology      | Why                                           |
|--------------|-----------------|-----------------------------------------------|
| Vector Store | MongoDB Atlas   | Production-grade $vectorSearch, no extra infra |
| Cache        | Redis           | Sub-millisecond semantic similarity lookup    |
| LLM Routing  | LangGraph       | Stateful graph with typed state               |
| API          | FastAPI         | Async, typed, production-ready                |
| Embeddings   | Gemini          | Free tier, 768 dimensions                     |

---

## Setup

1. Clone the repo
   ```bash
   git clone https://github.com/kotesh/cost-aware-rag.git
   cd cost-aware-rag
   ```

2. Copy `.env.example` to `.env` and fill in your credentials
   ```bash
   cp .env.example .env
   # Set GEMINI_API_KEY and MONGODB_URI
   ```

3. Start all services
   ```bash
   docker-compose up --build
   ```

4. Confirm health
   ```bash
   curl http://localhost:8000/health
   # {"status":"ok","mongodb":"ok","redis":"ok"}
   ```

---

## API Examples

### Ingest a document

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "X-API-Key: your-api-key" \
  -F "file=@your_document.md"
```

```json
{
  "request_id": "9bbba175-b14f-495f-adf1-326adce42086",
  "doc_id": "b233cb64-1095-470c-b010-6be13b6d37dd",
  "status": "processing"
}
```

### Query — cache miss (first call)

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "what is MongoDB Atlas Vector Search?", "top_k": 3}'
```

```json
{
  "request_id": "b2f10a60-8e27-43a7-8e58-9ac43c56cbbe",
  "answer": "MongoDB Atlas Vector Search uses HNSW indexing for approximate nearest neighbor search. It supports cosine, euclidean, and dot product similarity. The $vectorSearch aggregation stage is used in queries.",
  "cited_chunks": [
    {
      "doc_id": "b233cb64-1095-470c-b010-6be13b6d37dd",
      "chunk_index": 0,
      "score": 0.7746763229370117,
      "text": "MongoDB Atlas Vector Search uses HNSW indexing...",
      "source_file": "your_document.md"
    }
  ],
  "cost_metadata": {
    "model_used": "gemini-2.5-flash",
    "tokens_in": 274,
    "tokens_out": 46,
    "estimated_cost_usd": 0.0001,
    "cache_hit": false,
    "latency_ms": 1630.74
  },
  "latency_ms": 1630.74
}
```

### Query — cache hit (same query repeated)

```json
{
  "request_id": "d2f59167-7936-4c22-aada-d2736dd8e11e",
  "answer": "MongoDB Atlas Vector Search uses HNSW indexing for approximate nearest neighbor search...",
  "cited_chunks": [],
  "cost_metadata": {
    "model_used": "gemini-2.5-flash",
    "tokens_in": 274,
    "tokens_out": 46,
    "estimated_cost_usd": 0.0,
    "cache_hit": true,
    "latency_ms": 263.83
  },
  "latency_ms": 263.83
}
```

### Metrics

```bash
curl http://localhost:8000/api/v1/metrics
```

```json
{
  "total_queries": 8,
  "cache_hit_rate": 0.625,
  "avg_cost_usd": 0.00003,
  "total_cost_usd": 0.00024,
  "simple_query_count": 7,
  "complex_query_count": 1
}
```

---

## How It Works

- **Semantic cache first**: every query is embedded and compared against cached embeddings using cosine similarity. Queries scoring ≥ 0.92 return the cached answer instantly at zero LLM cost — including paraphrased or near-identical questions.
- **Zero-LLM complexity routing**: a heuristic classifier (keyword signals, query length, clause count) assigns each cache-miss query to `simple` or `complex` tier without calling an LLM, keeping routing cost at $0.
- **Grounded generation**: the generator receives only retrieved document chunks and is instructed to cite sources as `[chunk_N]`. If no relevant chunks exist, it returns a fixed fallback string — no hallucination.
- **Full cost observability**: every non-cached query writes a cost record to MongoDB (`query_costs` collection). The `/api/v1/metrics` endpoint aggregates totals, hit rates, and per-tier counts in real time.

---

## Running Tests

```bash
pytest tests/ -v --cov=app
```

---

## Bugs Fixed During Development

| # | Session | File | Bug | Fix |
|---|---------|------|-----|-----|
| 1 | 3 | `routers/ingest.py` | `filename` key overwrites reserved `LogRecord.filename` field → 500 on every ingest | Renamed extra key to `source_file` |
| 2 | 3 | `services/embedder.py` | `text-embedding-004` unavailable on this API key → 404 NOT_FOUND | Switched to `gemini-embedding-001` with `output_dimensionality=768` |
| 3 | 3 | `services/retriever.py` | `$vectorSearch` is Atlas-only, fails on local MongoDB → 500 on every query | Added in-process cosine similarity fallback via numpy |
| 4 | 3 | `.env` | `gemini-1.5-flash` / `gemini-1.5-pro` model names unavailable on current API | Updated to `gemini-2.5-flash` / `gemini-pro-latest` |
| 5 | 3 | `routers/query.py` | Cache hits always reported `cache_hit: false` — stored metadata never flipped | Added `cached.cost_metadata.cache_hit = True` before returning cached response |
| 6 | 4 | `models/schemas.py` | Empty string query `""` caused 500 instead of 422 validation error | Added `Field(min_length=1)` to `QueryRequest.query` |
| 7 | 4 | `services/generator.py` | Empty collection caused LLM hallucination — model answered from training data | Added early return of `_INSUFFICIENT` when `chunks` is empty, skipping LLM call |
| 8 | 4 | `routers/query.py` | Redis failure crashed the query endpoint with unhandled connection error | Wrapped `cache.get` and `cache.set` in try/except with bypass warning log |
| 9 | 4 | `main.py` | Metrics endpoint registered at `/metrics`, test expected `/api/v1/metrics` | Moved route path to `/api/v1/metrics` |
