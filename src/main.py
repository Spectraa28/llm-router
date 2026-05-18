import time 
from google import genai
from classifier import classify_query

LITE_MODEL = "gemini-2.5-flash-lite"
PRO_MODEL = "gemini-2.5-flash"

client = genai.Client(api_key="####")

def simulate_routing(query: str) -> dict:
    """
    Simulates the lifecycle of a query through the routing engine.
    Returns isolated telemetry for classification overhead and inference execution.
    """
    router_start = time.time()
    routing = classify_query(query)
    router_latency = (time.time() - router_start) * 1000
    
    target_model = PRO_MODEL if routing["is_complex"] else LITE_MODEL
    
    llm_start = time.time()
    response = client.models.generate_content(
        model=target_model,
        contents=f"Answer This financial query concisely: {query}",
        config={"temperature": 0.0}
    )
    llm_latency = (time.time() - llm_start) * 1000
    
    return {
        "query": query,
        "route": "PRO" if routing["is_complex"] else "LITE",
        "model": target_model,
        "score": routing["complexity_score"],
        "router_latency_ms": round(router_latency, 2),
        "llm_latency_ms": round(llm_latency, 2),
        "total_latency_ms": round(router_latency + llm_latency, 2),
        "input_tokens": response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count,
        "total_tokens": response.usage_metadata.prompt_token_count + response.usage_metadata.candidates_token_count,
        "response": response.text.strip()
    }