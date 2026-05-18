import re

# Expanded vocabulary to cover deep analytical verbs
REASONING_KEYWORDS = {
    "why", "compare", "driven", "impact", "trend", "difference", "explain", 
    "reason", "analysis", "versus", "vs", "evaluate", "analyze", "synthesize", 
    "breakdown", "assess", "review", "summarize", "exposure", "risks"
}

# Added structural metadata targets like balance sheets and statements
FINANCIAL_METRICS = {
    "revenue", "sales", "r&d", "research", "expense", "income", "margin", 
    "debt", "asset", "liability", "equity", "cash", "flow", "profit", "tax",
    "balance sheet", "ledger", "statement", "footnote", "compliance"
}

def classify_query(query: str) -> dict:
    """
    Analyzes a raw financial query to determine its routing path.
    Features additive cardinality scoring and punctuation-strip filters.
    """
    q_lower = query.lower()
    
    # Strip punctuation characters to protect word boundary checks
    clean_query = re.sub(r'[^\w\s]', ' ', q_lower)
    words = clean_query.split()
    word_count = len(words)
    
    # Signal 1: Length Factor (Weight: 20%)
    length_score = min(word_count / 20.0, 1.0)
    
    # Signal 2: Intent Keywords (Weight: 50%)
    found_intent_token = [word for word in words if word in REASONING_KEYWORDS]
    intent_match = len(found_intent_token) > 0
    # Scale intent score based on intensity, maxing out at 1.0
    intent_score = min(len(found_intent_token) * 0.5, 1.0) if intent_match else 0.0
    
    # Signal 3: Cardinality Analytics (Weight: 30%)
    years = set(re.findall(r"\b(20\d{2})\b", q_lower))
    found_metrics = {m for m in FINANCIAL_METRICS if m in q_lower}
    
    is_multi_year = len(years) > 1
    is_multi_metric = len(found_metrics) > 1
    
    year_points = 0.5 if is_multi_year else (0.2 if len(years) == 1 else 0.0)
    metric_points = 0.5 if is_multi_metric else (0.2 if len(found_metrics) == 1 else 0.0)
    cardinality_score = min(year_points + metric_points, 1.0)
    
    complexity_score = (intent_score * 0.5) + (cardinality_score * 0.3) + (length_score * 0.2)
    
    is_complex = complexity_score > 0.35
    
    return {
        "is_complex": is_complex,
        "complexity_score": round(complexity_score, 2),
        "metadata": {
            "word_count": word_count,
            "years_detected": list(years),
            "metrics_detected": list(found_metrics),
            "intent_tokens": found_intent_token
        }
    }
    
result = classify_query("Perform a structural breakdown of the balance sheet changes relative to macro inflation.")

print(result)