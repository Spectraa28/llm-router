import os
import logging
import time 
from google import genai
from classifier import classify_query
from cache import SemanticCache
from dotenv import load_dotenv
import threading

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LITE_MODEL = "gemini-2.5-flash-lite"
PRO_MODEL = "gemini-2.5-flash"
GEMINI_TIMEOUT = 10


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
_cache = SemanticCache()
cache = _cache

def simulate_routing(query: str) -> dict:
    """
    Simulates the lifecycle of a query through the routing engine.
    Returns isolated telemetry for classification overhead and inference execution.
    
    Full query cycle :
    1. check semantic cache 
    2. classify query complexity
    3.route to Lite or pro model
    4. store result in cache 
    return telemetry dict for benchmarkig and api response 
    """
    logger.info(f"Cache size at request time: {_cache.cache_size}")
    total_start = time.time()
    cached_response = _cache.get(query)
    logger.info(f"Cache get result: {cached_response}")
    total_start = time.time()
    
    # Semantic cache lookup
    cached_response = _cache.get(query)
    if cached_response:
        total_latency  = (time.time() - total_start) * 1000
        logger.info(f"CACHE HIT | query = '{query[:60]}' | latency={total_latency:.2f} ms")
        return {
            "query":query,
            "route":"CACHE",
            "model":"semantic_cache",
            "score":None,
            "router_latency_ms":0.0,
            "llm_latency_ms":0.0,
            "total_latency_ms": round(total_latency,2),
            "input_tokens":0,
            "output_tokens":0,
            "total_tokens": 0,
            "response":cached_response
        }
    
    
    # Step 2 classify query 
    router_start = time.time()
    routing = classify_query(query)
    router_latency = (time.time() - router_start) * 1000
    
    target_model = PRO_MODEL if routing["is_complex"] else LITE_MODEL
    route = "PRO" if routing["is_complex"] else "LITE"
    
    # LLM inference without timeout 
    #  Step 3: LLM Inference with Threading Timeout + Fallback
    FALLBACK_ORDER = [target_model, PRO_MODEL if target_model == LITE_MODEL else LITE_MODEL]

    response_text = None
    input_tokens = 0
    output_tokens = 0
    llm_latency = 0.0
    successful_model = None

    for attempt_model in FALLBACK_ORDER:
        result_container = {}
        error_container = {}

        def call_llm():
            try:
                resp = client.models.generate_content(
                    model=attempt_model,
                    contents=f"Answer this financial query concisely: {query}",
                    config={"temperature": 0.0}
                )
                result_container["response"] = resp
            except Exception as e:
                error_container["error"] = e

        llm_start = time.time()
        thread = threading.Thread(target=call_llm)
        thread.start()
        thread.join(timeout=GEMINI_TIMEOUT)
        llm_latency = (time.time() - llm_start) * 1000

        if thread.is_alive():
            logger.warning(f"Timeout on {attempt_model} after {GEMINI_TIMEOUT}s — trying fallback.")
            continue

        if "error" in error_container:
            logger.warning(f"Error on {attempt_model}: {error_container['error']} — trying fallback.")
            continue

        if "response" in result_container:
            resp = result_container["response"]
            response_text = resp.text.strip()
            input_tokens = resp.usage_metadata.prompt_token_count
            output_tokens = resp.usage_metadata.candidates_token_count
            successful_model = attempt_model
            break

    if response_text is None:
        raise RuntimeError(f"All models failed for query: '{query[:60]}'")
            
    # Store in cache 
    _cache.set(query,response_text)
    
    total_latency = (time.time() - total_start) * 1000
    
    logger.info(
        f"{route} | model={target_model}  | "
        f"router = {router_latency:.2f}ms | llm={llm_latency:.2f}ms | "
        f"total={total_latency:.2f}ms | tokens = {input_tokens + output_tokens}"
    )
    
    return {
        "query": query,
        "route": route,
        "model": successful_model,
        "score": routing["complexity_score"],
        "router_latency_ms": round(router_latency, 2),
        "llm_latency_ms": round(llm_latency, 2),
        "total_latency_ms": round(total_latency, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "response": response_text
    }