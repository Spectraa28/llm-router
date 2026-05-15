PRICING = {
    "gemini-2.5-flash-lite": {
        "input": 0.10, 
        "output": 0.40
    },
    "gemini-2.5-flash": {
        "input": 0.30, 
        "output": 2.50
    }
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """ 
    Calculates the USD cost for a specific model invocation
    """
    
    if model not in PRICING:
        raise ValueError(f"Unknown model identifier: {model}")
    
    rates = PRICING[model]
    
    cost_in = (input_tokens / 1_000_000) * rates["input"]
    cost_out = (output_tokens / 1_000_000) * rates["output"]
    
    return cost_in + cost_out

def simulate_savings(results: list) -> dict:
    """ 
    Calculates ROI by comparing the actual routed costs against a 'naive Baseline'
    """
    
    actual_total = 0.0
    baseline_total = 0.0
    
    for res in results:
        actual_cost = calculate_cost(
            res["model_used"],
            res["tokens"]["input"],
            res["tokens"]["output"]
        )
        
        actual_total  += actual_cost
        
        baseline_cost = calculate_cost(
            "gemini-2.5-flash",
            res["tokens"]["input"],
            res["tokens"]["output"]
        )
        
        baseline_total += baseline_cost
        
    saving_usd = baseline_total - actual_total
    
    saving_percent = (saving_usd / baseline_total * 100) if baseline_total > 0 else 0
    
    return {
        "actual_total_cost": actual_total,
        "baseline_total_cost": baseline_total,
        "total_savings_usd": saving_usd,
        "savings_percentage": round(saving_percent,2)
    }
    
