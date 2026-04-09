\# CLAUDE.md



\## Who I Am

Senior engineer. Java, Golang, Python, AWS, MongoDB, Kafka, Redis, Docker.

Every file must reflect production engineering — not tutorials, not demos.



\## Code Rules

\- Idiomatic Python. No Java-style Python.

\- Names: cfg, ctx, doc, msg fine. NEVER: data, result, temp, obj, handler.

\- Comments: WHY only. Code is the WHAT.

\- Guard clauses over nested if-else.

\- Errors specific: "failed to embed chunk: %s" not "error occurred"

\- Raise typed exceptions with context. Never bare except.

\- One-line docstring per public function. No more.

\- Realistic TODOs: # TODO: add circuit breaker once Redis SLA confirmed

\- No tutorial structure. No Step 1 / Step 2 comments.

\- Structured logs: logger.info("query routed", extra={"tier": tier})

\- Secrets from env vars only. Never hardcoded.

\- Docker: python:3.11-slim, non-root user, HEALTHCHECK.



\## Project: Cost-Aware RAG App



\### LLM Provider — Gemini ONLY

Package: google-generativeai

Key: GEMINI\_API\_KEY env var

Simple queries: gemini-1.5-flash

Complex queries: gemini-1.5-pro

Embeddings: text-embedding-004 (768 dimensions)

Do NOT use OpenAI or Anthropic inside this app.



\### Architecture

User Query

&#x20; → Redis Semantic Cache (cosine similarity >= 0.92)

&#x20;     → HIT:  return cached answer, cost = $0

&#x20;     → MISS: Complexity Classifier (NO LLM, pure heuristics)

&#x20;           → SIMPLE: Gemini Flash → MongoDB $vectorSearch → answer

&#x20;           → COMPLEX: Gemini Pro → MongoDB $vectorSearch → answer

&#x20;           → Cache result in Redis TTL 3600s

&#x20;           → Return answer + cost\_metadata



\### Every Response Must Include

cost\_metadata: {model\_used, tokens\_in, tokens\_out,

estimated\_cost\_usd, cache\_hit, latency\_ms}



\### File Structure

cost-aware-rag/

├── app/

│   ├── main.py

│   ├── config.py

│   ├── dependencies.py

│   ├── routers/

│   │   ├── ingest.py

│   │   └── query.py

│   ├── services/

│   │   ├── chunker.py

│   │   ├── embedder.py

│   │   ├── cache.py

│   │   ├── classifier.py

│   │   ├── retriever.py

│   │   ├── generator.py

│   │   └── cost\_tracker.py

│   ├── graph/

│   │   └── rag\_graph.py

│   ├── models/

│   │   └── schemas.py

│   └── utils/

│       └── logger.py

├── tests/

│   ├── test\_classifier.py

│   ├── test\_cache.py

│   └── test\_query\_endpoint.py

├── infra/

│   └── mongo\_index.js

├── docker-compose.yml

├── Dockerfile

├── .github/workflows/ci.yml

├── .env.example

├── requirements.txt

└── README.md



\### Env Vars

MONGODB\_URI, DB\_NAME=rag\_db, COLLECTION\_NAME=documents,

COSTS\_COLLECTION=query\_costs, GEMINI\_API\_KEY,

SIMPLE\_MODEL=gemini-1.5-flash, COMPLEX\_MODEL=gemini-1.5-pro,

COMPLEXITY\_THRESHOLD=0.4, REDIS\_URL=redis://redis:6379,

CACHE\_TTL=3600, CACHE\_SIMILARITY\_THRESHOLD=0.92,

API\_KEY, LOG\_LEVEL=INFO

