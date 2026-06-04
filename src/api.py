import os 
import logging 
from dotenv import load_dotenv
from fastapi import FastAPI , HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel , field_validator
from router import simulate_routing , cache

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Router API",
    description="Intelligent query router -- routes financial queries to LITE or PRO models",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origin=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# MOdels 
class RouteRequest(BaseModel):
    query: str
    
    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("query must not be empty")
        return v.strip()
    
class RouteResponse(BaseModel):
    query: str
    route: str
    model: str
    complexity_score: float | None
    router_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    response: str

class HealthResponse(BaseModel):
    status:str
    cached_size:int
    model_lite:str
    model_pro:str
    version:str
    
class BenchmarkSummary(BaseModel):
    total_queries: int
    lite_count: int
    pro_count: int
    cache_count: int
    failed_count: int
    routing_accuracy_pct: float
    cache_hit_rate_pct: float
    cost_saved_pct: float
    total_tokens: int
    
    
# Routes 
@app.get("/health",response_model=HealthResponse)
def health():
    """Returns API Status and cache size"""
    return HealthResponse(
        status="ok",
        cached_size=cache.cache_size,
        model_lite="gemini-2.5-flash-lite",
        model_pro="gemini-2.5-flash",
        version="1.0.0"
    )
    
@app.post("/route",response_model=RouteResponse)
def route_query(request: RouteRequest):
    """
    ROutes a financial query to LITE , PRO or return a CACHE HIT 
    """
    
    try: 
        telemetry = simulate_routing(request.query)
        return RouteResponse(
            query=telemetry["query"],
            route=telemetry["route"],
            model=telemetry["model"],
            complexity_score=telemetry["score"],
            router_latency_ms=telemetry["router_latency_ms"],
            llm_latency_ms=telemetry["llm_latency_ms"],
            total_latency_ms=telemetry["total_latency_ms"],
            input_tokens=telemetry["input_tokens"],
            output_tokens=telemetry["output_tokens"],
            total_tokens=telemetry["total_tokens"],
            response=telemetry["response"]
        )
    except RuntimeError as e:
        logger.error(f"/route failed : {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"/route unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/benchmark", response_model=BenchmarkSummary)
def run_benchmark():
    """
    Runs the full 100-query synthetic benchmark inline.
    Returns aggregated metrics. Also logs to MLflow if tracking server is up.
    """
    from benchmark import generate_financial_dataset, GROUND_TRUTH, PRICING

    queries = generate_financial_dataset()
    assert len(queries) == len(GROUND_TRUTH)

    lite_count = pro_count = cache_count = failed_count = 0
    accumulated_cost = 0.0
    theoretical_baseline_cost = 0.0
    total_tokens = 0
    correct_routes = 0
    total_gradable = 0

    for i, (query, ground_truth_label) in enumerate(zip(queries, GROUND_TRUTH)):
        try:
            telemetry = simulate_routing(query)
            route = telemetry["route"]
            in_tokens = telemetry["input_tokens"]
            out_tokens = telemetry["output_tokens"]

            if route == "LITE":
                lite_count += 1
            elif route == "PRO":
                pro_count += 1
            elif route == "CACHE":
                cache_count += 1

            actual_cost = (
                (in_tokens * PRICING[route]["input"]) +
                (out_tokens * PRICING[route]["output"])
            )
            accumulated_cost += actual_cost

            naive_pro_cost = (
                (in_tokens * PRICING["PRO"]["input"]) +
                (out_tokens * PRICING["PRO"]["output"])
            )
            theoretical_baseline_cost += naive_pro_cost
            total_tokens += telemetry["total_tokens"]

            if route != "CACHE":
                predicted_complex = 1 if route == "PRO" else 0
                if predicted_complex == ground_truth_label:
                    correct_routes += 1
                total_gradable += 1

        except Exception as e:
            failed_count += 1
            logger.error(f"Benchmark query {i+1} failed: {e}")

    efficiency_savings = (
        ((theoretical_baseline_cost - accumulated_cost) / theoretical_baseline_cost) * 100
        if theoretical_baseline_cost > 0 else 0.0
    )
    routing_accuracy = (
        (correct_routes / total_gradable) * 100
        if total_gradable > 0 else 0.0
    )
    cache_hit_rate = (cache_count / len(queries)) * 100

    return BenchmarkSummary(
        total_queries=len(queries),
        lite_count=lite_count,
        pro_count=pro_count,
        cache_count=cache_count,
        failed_count=failed_count,
        routing_accuracy_pct=round(routing_accuracy, 2),
        cache_hit_rate_pct=round(cache_hit_rate, 2),
        cost_saved_pct=round(efficiency_savings, 2),
        total_tokens=total_tokens
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)