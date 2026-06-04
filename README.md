# LLM Router

An intelligent query routing system for financial workloads that reduces LLM inference costs by routing queries to the appropriate model tier based on complexity, with a semantic cache layer to eliminate redundant API calls.

**Live Demo:** `https://your-render-url.onrender.com` ← replace after deployment

---

## Results (Synthetic Benchmark — 100 Queries)

| Metric | Value |
|---|---|
| Cost Reduction vs All-PRO Baseline | **93.19%** |
| Cache Hit Rate | **87%** |
| Routing Accuracy | **100%** |
| Failed Queries | **0 / 100** |

> Benchmark run on a synthetic dataset of 100 financial queries (80 simple, 20 complex). The 87% cache hit rate is high due to template-based query repetition — in production with diverse real user queries, cache hit rates would likely be 20-30%. The 93.19% cost reduction reflects ideal cache conditions; realistic production savings would primarily come from the routing layer, estimated at 40-60%. Labels generated via weak supervision using a rule-based classifier with manual spot-check validation.

---

## Architecture

```
Query
  │
  ▼
Semantic Cache (FAISS + Redis)
  │ HIT → return cached response (0 tokens, 0 cost)
  │ MISS ↓
  ▼
TF-IDF + Logistic Regression Classifier
  │ simple (score < 0.45) → gemini-2.5-flash-lite
  │ complex (score ≥ 0.45) → gemini-2.5-flash
  ▼
Gemini API (threading timeout + fallback chain)
  │
  ▼
Store response in cache → return telemetry
```

---

## Motivation

LLM API costs scale linearly with usage. Most production query workloads are highly repetitive and span a wide range of complexity — a simple data extraction query costs the same as a multi-document synthesis task if routed naively to a single model.

This project implements three cost-reduction strategies in a single pipeline:

1. **Semantic caching** — identical or semantically similar queries are served from cache at zero token cost
2. **Complexity-based routing** — a lightweight ML classifier routes simple queries to a cheaper model tier
3. **Fallback chain** — if the primary model times out, the system automatically retries with the alternate model instead of failing

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM API | Google Gemini (gemini-2.5-flash, gemini-2.5-flash-lite) |
| Classifier | TF-IDF + Logistic Regression (scikit-learn) |
| Semantic Cache | FAISS + Redis |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| API Framework | FastAPI + Uvicorn |
| Experiment Tracking | MLflow |
| Deployment | Docker + Render |

---

## API Endpoints

### `GET /health`
Returns system status and current cache size.

```json
{
  "status": "ok",
  "cache_size": 12,
  "model_lite": "gemini-2.5-flash-lite",
  "model_pro": "gemini-2.5-flash",
  "version": "1.0.0"
}
```

### `POST /route`
Routes a financial query through the full pipeline.

**Request:**
```json
{
  "query": "Compare operating cash flows across all fiscal quarters and pinpoint anomalies."
}
```

**Response:**
```json
{
  "query": "Compare operating cash flows across all fiscal quarters and pinpoint anomalies.",
  "route": "PRO",
  "model": "gemini-2.5-flash",
  "complexity_score": 0.8776,
  "router_latency_ms": 15.87,
  "llm_latency_ms": 5190.01,
  "total_latency_ms": 5309.84,
  "input_tokens": 19,
  "output_tokens": 65,
  "total_tokens": 84,
  "response": "..."
}
```

### `POST /benchmark`
Runs the full 100-query synthetic benchmark and returns aggregated metrics.

---

## Demo Queries

Try these against the `/route` endpoint:

**Routes to LITE:**
```
What is the billing date for transaction ID-4412?
Extract the current outstanding balance from this ledger snippet.
What are the payment terms specified on the vendor invoice?
```

**Routes to PRO:**
```
Compare operating cash flows across all fiscal quarters and pinpoint anomalies.
Synthesize cross-quarter exposure risks and generate a risk-mitigation summary.
Evaluate the structural impact of the shifting debt-to-equity ratio in section 4.
```

**Second call routes to CACHE:**
```
Any of the above queries sent a second time
```

---

## Project Structure

```
llm-router/
├── src/
│   ├── classifier.py      # TF-IDF + LogReg classifier, weak supervision training
│   ├── router.py          # Core routing logic, fallback chain, cache integration
│   ├── cache.py           # SemanticCache — FAISS vector store + Redis KV store
│   ├── api.py             # FastAPI endpoints — /health /route /benchmark
│   ├── benchmark.py       # 100-query synthetic benchmark, MLflow logging
│   └── cache_store/       # Persisted FAISS index and metadata
├── .env.example
├── .gitignore
├── .dockerignore
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Running Locally

**Prerequisites:** Docker and Docker Compose installed.

```bash
# Clone the repo
git clone https://github.com/your-username/llm-router
cd llm-router

# Set up environment
cp .env.example .env
# Add your GEMINI_API_KEY to .env

# Start all services
docker-compose up --build

# Test health
curl http://localhost:8000/health

# Test routing
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the billing date for transaction ID-4412?"}'

# Run benchmark
curl -X POST http://localhost:8000/benchmark
```

---

## Environment Variables

See `.env.example` for the full list. Required:

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio API key |
| `COMPLEXITY_THRESHOLD` | Routing threshold (default: 0.45) |
| `CACHE_SIMILARITY_THRESHOLD` | Cache hit threshold (default: 0.88) |
| `REDIS_HOST` | Redis hostname (default: localhost) |
| `GEMINI_TIMEOUT` | LLM timeout in seconds (default: 10) |

---

## Classifier Design

The routing classifier is a TF-IDF vectorizer + Logistic Regression pipeline trained on 450 weakly-supervised synthetic financial queries (300 simple, 150 complex). Labels were generated using a rule-based keyword classifier as a weak supervisor, with manual spot-check validation on edge cases — a technique similar to programmatic labeling used in production ML systems.

The complexity threshold is configurable via `COMPLEXITY_THRESHOLD` in `.env`. Lowering it routes more queries to PRO (higher accuracy, higher cost). Raising it routes more to LITE (lower cost, lower accuracy).

The classifier achieves 100% accuracy on its held-out synthetic test set. Real-world performance would require labeled queries from actual financial workflows.

---

## Honest Limitations

- **Cache hit rate** — 87% on the synthetic benchmark is misleading for production estimates. Template-based query generation creates high similarity between queries. Real-world cache hit rates for diverse financial workloads would likely be 20-30%.
- **Cost savings** — The 93.19% figure reflects ideal cache conditions. Realistic production savings would primarily come from the routing layer, estimated at 40-60% depending on query complexity distribution.
- **Classifier training data** — 450 synthetic samples is sufficient for demonstration but a production classifier would require labeled real-world queries from actual financial workflows.
- **Dataset** — Queries are programmatically generated for benchmarking purposes and do not represent real financial data.

---

*Built as a portfolio project demonstrating cost-optimized LLM infrastructure for financial query workloads.*
```

---
