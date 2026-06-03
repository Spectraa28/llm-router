# Rule-Based LLM Cost Router

A high-performance, deterministic traffic cop designed to intercept incoming financial queries, calculate structural complexity, and dynamically route requests to the most optimal model tier. By deflecting simple point-lookups to **Gemini Flash-Lite** and reserving heavy analytical reasoning for **Gemini Flash**, this engine achieves significant production cost reductions without sacrificing data integrity or response quality.

---

## 📊 Core Performance Metrics

The engine was validated using a balanced 100-query workload on an unthrottled paid tier tier.

| Metric | Baseline (All-PRO) | Blended Engine (Router) | Performance Multiplier |
| --- | --- | --- | --- |
| **Total Cost** | $0.016839 | $0.010753 | **36.14% Cost Saved** |
| **Routing Overhead** | 0.00ms | **< 0.08ms** | Near-Zero Latency Cost |
| **Complex Query Recall** | 30% (Initial) | **100% (Refined)** | Zero Leakage / Zero Hallucinations |
| **Traffic Split** | 100% PRO | **77% LITE / 20% PRO** | 3 Server-Side Drops (Gracefully Handled) |

---

## 🧠 Architectural Overview

The engine utilizes an additive heuristic matrix that evaluates text profiles across three isolated structural signals before deciding on a model path.

```
                  [ Incoming User Query ]
                             │
                             ▼
               ┌───────────────────────────┐
               │ Regular Expression Filter │ (Strip Punctuation)
               └─────────────┬─────────────┘
                             │
                             ▼
               ┌───────────────────────────┐
               │ Additive Scoring Matrix   │
               │  - Signal 1: Word Count   │
               │  - Signal 2: Intent Verbs │
               │  - Signal 3: Cardinality  │
               └─────────────┬─────────────┘
                             │
                             ├─ Complexity Score > 0.35 ──► [ Gemini Pro ]
                             │
                             └─ Complexity Score <= 0.35 ─► [ Gemini Lite ]

```

### 1. Text Sanitization Layer

To protect word boundary execution, a regex layer purges all punctuation trailing tokens (e.g., matching `"trend?"` cleanly to `"trend"`):

```python
clean_query = re.sub(r'[^\w\s]', ' ', q_lower)

```

### 2. Evaluated Signals Matrix

* **Signal 1: Length Factor (Weight: 20%)** — Normalizes query word count against a maximum complexity baseline of 20 words.
* **Signal 2: Reasoning Intent (Weight: 50%)** — Scans token streams for exact matches against custom analytical verb lookup sets (`HashSet` with $O(1)$ lookup complexity).
* **Signal 3: Additive Cardinality Analytics (Weight: 30%)** — Tallies individual entity frequencies (years and financial terms) independently, breaking brittle strict `AND` gate conditions.

---

## 📁 Repository Structure

* `main.py`: Contains the core heuristic configuration array, regex text-stripping blocks, and the live Google GenAI integration loop.
* `benchmark_100.py`: The test harness execution file that triggers the 100-query load test, manages financial accumulation math, and pushes live telemetry data out to MLflow.
* `mlflow.db`: Local SQLite instance storing granular run entries.

---

## 🚀 Getting Started

### 1. Environment Setup

Clone the repository and configure your environment variables:

```bash
cp .env.example .env
# Open .env and populate your GEMINI_API_KEY

```

### 2. Stand up Telemetry Dashboard

Start your local MLflow tracking server on port 5001:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 5001

```

### 3. Run the Benchmark

Execute the unthrottled load test to populate the performance curves:

```bash
python benchmark_100.py

```
