import time
import mlflow
from router import simulate_routing

#  MLflow Setup 
import os
from dotenv import load_dotenv
load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "Simple Deterministic Router"))

#  Pricing
PRICING = {
    "LITE": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
    "PRO":  {"input": 0.30 / 1_000_000, "output": 2.50 / 1_000_000},
    "CACHE": {"input": 0.0,              "output": 0.0}
}

#  Ground Truth Labels (for routing accuracy) 
# 0 = simple, 1 = complex — matches the dataset structure exactly
GROUND_TRUTH = [0] * 80 + [1] * 20


def generate_financial_dataset():
    simple_lookups = [
        "What is the billing date for transaction ID-4412?",
        "Extract the current outstanding balance from this ledger snippet.",
        "What was the total operating expense listed on page 12?",
        "Find the net profit metric for Q1 2026.",
        "Identify the merchant category code for this vendor transaction.",
        "What is the corporate tax identification number listed in Section C?",
        "Read the statement and extract the total credit adjustment value.",
        "What are the payment terms specified on the vendor invoice?"
    ] * 10  # 80 queries

    complex_analysis = [
        "Synthesize cross-quarter exposure risks and generate a risk-mitigation summary.",
        "Evaluate the structural impact of the shifting debt-to-equity ratio in section 4.",
        "Compare operating cash flows across all fiscal quarters and pinpoint anomalies.",
        "Perform a comprehensive breakdown of balance sheet changes relative to macro inflation.",
        "Analyze compliance footnotes to assess liability risks across regional subsidiaries."
    ] * 4  # 20 queries

    return simple_lookups + complex_analysis


def run_performance_benchmark():
    queries = generate_financial_dataset()
    assert len(queries) == len(GROUND_TRUTH), "Dataset and ground truth length mismatch"

    print(f"\n{'═'*55}")
    print(f"  SYNTHETIC DATASET BENCHMARK — {len(queries)} queries")
    print(f"  Note: Queries are programmatically generated for demo purposes")
    print(f"{'═'*55}\n")

    with mlflow.start_run(run_name="Full_Benchmark_Run"):

        #  Counters 
        lite_count = 0
        pro_count = 0
        cache_count = 0
        failed_count = 0

        accumulated_cost = 0.0
        theoretical_baseline_cost = 0.0
        total_tokens = 0

        correct_routes = 0
        total_gradable = 0  # excludes CACHE hits from accuracy calc

        complexity_scores = []

        for i, (query, ground_truth_label) in enumerate(zip(queries, GROUND_TRUTH)):
            print(f"🔄 [{i+1}/{len(queries)}] Processing...", flush=True)

            try:
                telemetry = simulate_routing(query)

                route = telemetry["route"]
                in_tokens = telemetry["input_tokens"]
                out_tokens = telemetry["output_tokens"]

                #  Route Counters 
                if route == "LITE":
                    lite_count += 1
                elif route == "PRO":
                    pro_count += 1
                elif route == "CACHE":
                    cache_count += 1

                #  Cost Tracking 
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

                #  Routing Accuracy (skip CACHE hits) 
                if route != "CACHE":
                    predicted_complex = 1 if route == "PRO" else 0
                    if predicted_complex == ground_truth_label:
                        correct_routes += 1
                    total_gradable += 1

                #  Complexity Score Distribution 
                if telemetry["score"] is not None:
                    complexity_scores.append(telemetry["score"])

                mlflow.log_metric("router_overhead_ms", telemetry["router_latency_ms"], step=i)
                mlflow.log_metric("model_inference_latency_ms", telemetry["llm_latency_ms"], step=i)
                mlflow.log_metric("total_turnaround_latency_ms", telemetry["total_latency_ms"], step=i)
                mlflow.log_metric("cumulative_blended_spend_usd", accumulated_cost, step=i)

                print(
                    f"  ✅ Route: {route} | "
                    f"Router: {telemetry['router_latency_ms']}ms | "
                    f"LLM: {telemetry['llm_latency_ms']}ms | "
                    f"Tokens: {telemetry['total_tokens']}",
                    flush=True
                )

            except Exception as e:
                failed_count += 1
                mlflow.log_metric("failed_queries", failed_count, step=i)
                print(f"  ❌ [{i+1}] Failed: {e}", flush=True)

        #  Final Aggregations 
        efficiency_savings = (
            ((theoretical_baseline_cost - accumulated_cost) / theoretical_baseline_cost) * 100
            if theoretical_baseline_cost > 0 else 0.0
        )

        routing_accuracy = (
            (correct_routes / total_gradable) * 100
            if total_gradable > 0 else 0.0
        )

        cache_hit_rate = (cache_count / len(queries)) * 100

        score_min = min(complexity_scores) if complexity_scores else 0.0
        score_mean = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0.0
        score_max = max(complexity_scores) if complexity_scores else 0.0

        #  Final MLflow Metrics 
        mlflow.log_metric("total_processed_tokens", total_tokens)
        mlflow.log_metric("lite_deflection_ratio", lite_count / len(queries))
        mlflow.log_metric("final_tracked_spend_usd", accumulated_cost)
        mlflow.log_metric("calculated_cost_reduction_pct", efficiency_savings)
        mlflow.log_metric("routing_accuracy_pct", routing_accuracy)
        mlflow.log_metric("cache_hit_rate_pct", cache_hit_rate)
        mlflow.log_metric("failed_queries_total", failed_count)
        mlflow.log_metric("complexity_score_min", score_min)
        mlflow.log_metric("complexity_score_mean", score_mean)
        mlflow.log_metric("complexity_score_max", score_max)

        mlflow.log_param("dataset_type", "synthetic")
        mlflow.log_param("total_queries", len(queries))
        mlflow.log_param("simple_queries", 80)
        mlflow.log_param("complex_queries", 20)
        mlflow.log_param("model_lite", "gemini-2.5-flash-lite-preview-06-17")
        mlflow.log_param("model_pro", "gemini-2.5-flash")

        #  Final Print 
        print(f"\n{'═'*55}")
        print("  📊 BENCHMARK RESULTS — SYNTHETIC DATASET")
        print(f"{'═'*55}")
        print(f"  LITE  routes        : {lite_count}")
        print(f"  PRO   routes        : {pro_count}")
        print(f"  CACHE hits          : {cache_count}")
        print(f"  Failed queries      : {failed_count}")
        print(f"{'─'*55}")
        print(f"  Routing accuracy    : {routing_accuracy:.2f}%")
        print(f"  Cache hit rate      : {cache_hit_rate:.2f}%")
        print(f"{'─'*55}")
        print(f"  Blended cost        : ${accumulated_cost:.6f}")
        print(f"  All-PRO baseline    : ${theoretical_baseline_cost:.6f}")
        print(f"  Cost saved          : {efficiency_savings:.2f}%")
        print(f"{'─'*55}")
        print(f"  Complexity scores   : min={score_min:.4f} mean={score_mean:.4f} max={score_max:.4f}")
        print(f"  Total tokens used   : {total_tokens}")
        print(f"{'═'*55}\n")


if __name__ == "__main__":
    run_performance_benchmark()