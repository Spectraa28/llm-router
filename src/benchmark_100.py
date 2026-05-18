import time
import mlflow
from main import simulate_routing 

# Connect to your local persistent SQLite tracking backend
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("Simple Deterministic router")

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
    ] * 10  # 80 Queries

    complex_analysis = [
        "Synthesize cross-quarter exposure risks and generate a risk-mitigation summary.",
        "Evaluate the structural impact of the shifting debt-to-equity ratio in section 4.",
        "Compare operating cash flows across all fiscal quarters and pinpoint anomalies.",
        "Perform a comprehensive breakdown of balance sheet changes relative to macro inflation.",
        "Analyze compliance footnotes to assess liability risks across regional subsidiaries."
    ] * 4  # 20 Queries

    return simple_lookups + complex_analysis

def run_performance_benchmark():
    queries = generate_financial_dataset()
    
    # Official Paid Tier Prices per 1 Million Tokens
    PRICING = {
        "LITE": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
        "PRO": {"input": 0.30 / 1_000_000, "output": 2.50 / 1_000_000}
    }

    print(f"Launching Unthrottled Load Test: Processing {len(queries)} back-to-back runs...")

    with mlflow.start_run(run_name="Paid_Tier_Full_Speed_Run"):
        lite_routes_triggered = 0
        pro_routes_triggered = 0
        accumulated_blended_cost = 0.0
        theoretical_baseline_cost = 0.0
        total_tokens_run = 0

        for i, query in enumerate(queries):
            # All pacing delays are completely removed to execute at full machine capability
            print(f"🔄 [{i+1}/100] Processing active query node...", flush=True)
            
            try:
                telemetry = simulate_routing(query)
                
                route_taken = telemetry["route"] 
                in_tokens = telemetry["input_tokens"]
                out_tokens = telemetry["output_tokens"]
                
                if route_taken == "LITE":
                    lite_routes_triggered += 1
                else:
                    pro_routes_triggered += 1

                actual_run_cost = (in_tokens * PRICING[route_taken]["input"]) + (out_tokens * PRICING[route_taken]["output"])
                accumulated_blended_cost += actual_run_cost

                naive_pro_cost = (in_tokens * PRICING["PRO"]["input"]) + (out_tokens * PRICING["PRO"]["output"])
                theoretical_baseline_cost += naive_pro_cost
                
                total_tokens_run += telemetry["total_tokens"]

                # Log to local tracking instance
                mlflow.log_metric("router_overhead_ms", telemetry["router_latency_ms"], step=i)
                mlflow.log_metric("model_inference_latency_ms", telemetry["llm_latency_ms"], step=i)
                mlflow.log_metric("total_turnaround_latency_ms", telemetry["total_latency_ms"], step=i)
                mlflow.log_metric("cumulative_blended_spend_usd", accumulated_blended_cost, step=i)
                
                print(f"Success! Tier: {route_taken} | Decision: {telemetry['router_latency_ms']}ms | Core LLM: {telemetry['llm_latency_ms']}ms", flush=True)

            except Exception as e:
                print(f"Unexpected script execution error at index {i+1}: {str(e)}", flush=True)

        efficiency_savings = ((theoretical_baseline_cost - accumulated_blended_cost) / theoretical_baseline_cost) * 100

        # Terminate Run Metrics Aggregations
        mlflow.log_metric("total_processed_tokens", total_tokens_run)
        mlflow.log_metric("lite_deflection_ratio", lite_routes_triggered / len(queries))
        mlflow.log_metric("final_tracked_spend_usd", accumulated_blended_cost)
        mlflow.log_metric("calculated_cost_reduction_percentage", efficiency_savings)

        print("\n" + "═"*50)
        print("📊 FINAL INFRASTRUCTURE SYSTEM METRICS")
        print("═"*50)
        print(f"• Fast Tier (LITE) Activations  : {lite_routes_triggered} runs")
        print(f"• Heavy Tier (PRO) Activations  : {pro_routes_triggered} runs")
        print(f"• Total Cost (Blended Engine)   : ${accumulated_blended_cost:.6f}")
        print(f"• Naive Baseline (All-PRO Cost) : ${theoretical_baseline_cost:.6f}")
        print(f"• Verified ROI Percentage       : {efficiency_savings:.2f}% Cost Saved")
        print("═"*50)

if __name__ == "__main__":
    run_performance_benchmark()