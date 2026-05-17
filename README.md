# 🔍 Advanced RAG Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-0.2-orange)
![FAISS](https://img.shields.io/badge/FAISS-1.8-red)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow)

**A production-grade Retrieval-Augmented Generation (RAG) platform with hybrid search, semantic embeddings, metadata filtering, and real-time LLM inference.**

</div>

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐  │
│  │  /query     │  │  /documents │  │  /health & metrics │  │
│  └──────┬──────┘  └──────┬──────┘  └────────────────────┘  │
│         │                │                                   │
│  ┌──────▼──────────────────────────────────────────────┐    │
│  │                  RAG Service                         │    │
│  │  ┌────────────┐  ┌──────────────┐  ┌─────────────┐ │    │
│  │  │  Retriever │  │ LLM Client   │  │   Context   │ │    │
│  │  │  (Hybrid)  │  │ (OpenAI)     │  │  Assembler  │ │    │
│  │  └────┬───────┘  └──────────────┘  └─────────────┘ │    │
│  └───────┼──────────────────────────────────────────────┘   │
│          │                                                   │
│  ┌───────▼──────────────────────────────────────────────┐   │
│  │               Retrieval Layer                         │   │
│  │  ┌─────────────────┐    ┌──────────────────────────┐ │   │
│  │  │  FAISS Vector   │    │      BM25 Keyword        │ │   │
│  │  │  Store (Semantic│◄───┤      Index (Lexical)     │ │   │
│  │  │  Search)        │    │                          │ │   │
│  │  └─────────────────┘    └──────────────────────────┘ │   │
│  │          ▲  Reciprocal Rank Fusion (RRF)  ▲           │   │
│  └──────────┼────────────────────────────────┼───────────┘   │
│             │                                │               │
│  ┌──────────┴────────────────────────────────┴───────────┐   │
│  │              Ingestion Pipeline                        │   │
│  │  PDF/DOCX/TXT → Text Extract → Chunk → Embed → Index  │   │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │                                 │
    ┌────▼────┐                      ┌─────▼──────┐
    │ Redis   │                      │ PostgreSQL  │
    │ Cache   │                      │  Metadata  │
    └─────────┘                      └────────────┘
```

---

## ✨ Key Features

| Feature | Details |
|---|---|
| **Hybrid Retrieval** | Semantic (FAISS) + Keyword (BM25) fusion via Reciprocal Rank Fusion |
| **Smart Chunking** | 3 strategies: Recursive, Semantic (sentence-boundary), Sliding Window |
| **Embedding Pipeline** | Batched OpenAI embeddings with in-memory caching & retry logic |
| **Context-Aware LLM** | Multi-turn conversation history, token budget management |
| **Streaming** | Server-Sent Events (SSE) for real-time token streaming |
| **Metadata Filtering** | Field-level filters on any document metadata attribute |
| **Reranking** | Cosine-similarity reranking (swap with cross-encoder in prod) |
| **Observability** | Structured JSON logging, Prometheus metrics, health endpoints |
| **Containerized** | Multi-stage Docker build, Docker Compose with Postgres + Redis |

---

## 🚀 Quick Start

### 1. Clone & Configure

```bash
git clone https://github.com/Rishabhkumar1714/advanced-rag-platform.git
cd advanced-rag-platform

cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Run with Docker Compose (Recommended)

```bash
docker-compose up --build
```

API available at: `http://localhost:8000`  
Swagger docs at: `http://localhost:8000/docs`

### 3. Run Locally

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload
```

---

## 📖 API Reference

### Query Endpoint

```bash
# Standard Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "question": "What are the key benefits of RAG systems?",
    "retrieval_mode": "hybrid",
    "top_k": 5,
    "rerank": true,
    "include_sources": true
  }'

# Streaming Query
curl -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain vector databases", "stream": true}'
```

### Document Ingestion

```bash
# Upload a file
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer your-token" \
  -F "file=@/path/to/document.pdf" \
  -F "collection_id=engineering-docs"

# Ingest raw text
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "RAG Overview",
    "content": "RAG combines retrieval with generation...",
    "collection_id": "ai-docs"
  }'

# List documents
curl http://localhost:8000/api/v1/documents?page=1&page_size=10
```

### Bulk Ingestion CLI

```bash
# Ingest an entire directory
python scripts/ingest_documents.py --dir ./data/raw --collection my-docs --strategy semantic

# Ingest a single file
python scripts/ingest_documents.py --file ./report.pdf
```

---

## 📁 Project Structure

```
advanced-rag-platform/
├── app/
│   ├── main.py                  # FastAPI app factory & lifespan
│   ├── config.py                # Centralized settings (Pydantic)
│   ├── api/
│   │   ├── routes/
│   │   │   ├── query.py         # RAG query endpoints
│   │   │   ├── documents.py     # Document CRUD & upload
│   │   │   └── health.py        # Health / readiness probes
│   │   └── dependencies.py      # DI: auth, services
│   ├── core/
│   │   ├── chunking.py          # Recursive, Semantic, SlidingWindow chunkers
│   │   ├── embeddings.py        # OpenAI embedding pipeline (batched + cached)
│   │   ├── vector_store.py      # FAISS index management
│   │   ├── retriever.py         # Hybrid retriever (BM25 + FAISS + RRF)
│   │   └── llm.py               # OpenAI LLM client (stream + non-stream)
│   ├── models/
│   │   ├── document.py          # Document schemas
│   │   └── query.py             # Query/response schemas
│   ├── services/
│   │   ├── ingestion_service.py # Full ingestion pipeline orchestrator
│   │   ├── rag_service.py       # RAG orchestration service
│   │   └── document_service.py  # Document CRUD service
│   └── utils/
│       ├── logger.py            # Structured logging (structlog)
│       └── helpers.py           # Tokenization, chunking, UUID utilities
├── tests/
│   ├── test_api.py              # API integration tests
│   └── test_retriever.py        # Unit tests for chunker & BM25
├── scripts/
│   └── ingest_documents.py      # CLI bulk ingestion script
├── data/
│   ├── raw/                     # Place source documents here
│   └── processed/               # FAISS index stored here
├── Dockerfile                   # Multi-stage production Docker build
├── docker-compose.yml           # Full stack: API + Postgres + Redis
├── requirements.txt
└── .env.example
```

---

## ⚙️ Configuration

All settings are in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | LLM model for generation |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model |
| `RETRIEVAL_TOP_K` | `5` | Number of chunks to retrieve |
| `HYBRID_SEARCH_ALPHA` | `0.5` | 1=pure semantic, 0=pure keyword |
| `CHUNK_SIZE` | `512` | Max chars per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `SIMILARITY_THRESHOLD` | `0.7` | Min score to include a chunk |
| `RERANKER_ENABLED` | `true` | Enable post-retrieval reranking |
| `DATABASE_URL` | PostgreSQL | Connection string |
| `REDIS_URL` | Redis | Cache connection string |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=app --cov-report=html
```

---

## 📊 Monitoring

Launch with the monitoring profile for Prometheus + Grafana:

```bash
docker-compose --profile monitoring up
```

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3001` (admin / admin)
- Metrics endpoint: `http://localhost:8000/metrics`

---

## 🔧 Production Checklist

- [ ] Set `APP_ENV=production` and a strong `SECRET_KEY`
- [ ] Configure real JWT auth in `app/api/dependencies.py`
- [ ] Replace in-memory `DocumentService` with PostgreSQL via SQLAlchemy
- [ ] Swap reranker with a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- [ ] Configure Redis for persistent embedding cache
- [ ] Set up log aggregation (ELK / Datadog)
- [ ] Add rate limiting middleware (e.g., `slowapi`)

---

## 👤 Author

**Rishabh Kumar**  
GitHub: [@Rishabhkumar1714](https://github.com/Rishabhkumar1714)

---

## 📄 License

This project is licensed under the MIT License.
