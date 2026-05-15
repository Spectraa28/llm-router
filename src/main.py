import time 
from google import genai
from classifier import classify_query

LITE_MODEL = "gemini-2.5-flash-lite"
PRO_MODEL = "gemini-2.5-flash"

client = genai.Client(api_key="AIzaSyA2yfML4-nWHHQca7GbB0dUDZbzYODM2bY")

def simulate_routing(query: str) -> dict:
    """
    Simulates the lifecycle of a query through the routing engine.
    Returns full telemetry for cost and latency analysis.
    """
    start_time = time.time()
    
    # DEcision phase 
    routing = classify_query(query)
    target_model = PRO_MODEL if routing["is_complex"] else LITE_MODEL
    
    # Execution phase 
    response = client.models.generate_content(
        model=target_model,
        contents= f"Answer THis financial query concisely: {query}",
        config={"temperature":0.0}
    )
    
    end_time = time.time()
    
    return {
        "query":query,
        "route": "PRO" if routing["is_complex"] else "LITE",
        "model":target_model,
        "score": routing["complexity_score"],
        "latency_ms": round((end_time - start_time) * 1000,2),
        "tokens": {
            "input": response.usage_metadata.prompt_token_count,
            "output": response.usage_metadata.candidates_token_count
        },
        "response": response.text.strip()
    }
    
if __name__ == "__main__":
    # A mix of simple and complex queries to test the decision boundary
    test_suite = [
        "What is the CEO's name?", 
        "Compare the revenue growth of 2023 vs 2024 and explain the trend.",
        "What is the par value per share?",
        "Analyze the impact of high interest rates on the 2023 net income."
    ]
    
    print(f"{'QUERY':<50} | {'ROUTE':<6} | {'MS':<8}")
    print("-" * 70)
    
    for q in test_suite:
        data = simulate_routing(q)
        print(f"{q[:48]:<50} | {data['route']:<6} | {data['latency_ms']:<8}")