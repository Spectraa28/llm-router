import re

# Constants for the logic  
REASONING_KEYWORDS = {
    "why" ,"compare", "driven", "impact" , "trend",
    "difference","explain","reason","analysis","versus","vs"
}


FINANCIAL_METRICS = {
    "revenue", "sales", "r&d", "research", "expense", 
    "income", "margin", "debt", "asset", "liability", 
    "equity", "cash", "flow", "profit", "tax"
}

def classify_query(query: str) -> dict:
    """
    Analyzes a raw financial query to  determine  its routing path based on 
    Deterministic structural and semantic signals 
    
    This function acts as a  low-latency 'Traffic Cop,' whether a 
    query is a 'Point Lookup' (Simple)  or a 'Reasoning Task' (Complex) .
    It biases towards complexity to  prevent hallucinations in simple models.
    
    Args:  
        query(str): the raw user query string (e.g. "What was Apples 2023 R&D?"
        
    Returns :
        dict: A routing payload containing:
            - is_complex(bool): The final decision (True = Route to Pro model)
            - complexity score(float): the calculated complexity coefficient 
            - metadata(dict): breakdown of three evaluated signals 
                    - 'length_factor': Normalized word count signal.
                    - 'cardinality_count': Count of unique years (YYYY) and metrics.
                    - 'intent_match': Boolean indicating presence of analyst keywords.
        
        Heuristics & Signals:
    1. Query Length: Short queries (< 10 words) are biased toward the 'Lite' path.
    2. Temporal/Entity Cardinality: Uses regex to detect multiple years or 
       competing financial metrics (e.g., '2022 and 2023' or 'R&D and Net Sales').
    3. Reasoning Intent: Scan for analyst-specific verbs ('why', 'compare', 
       'driven', 'impact', 'trend') that signal a need for prose-heavy MD&A synthesis.
    
    """
    
    q_lower = query.lower()
    
    # Signal 1 -> length factor 
    words  = q_lower.split()
    word_count = len(words)
    length_score = min(word_count / 20.0 ,1.0)
    
    # Signal 2 : Intent Keywords
    found_intent_token = [word for word in words if word in REASONING_KEYWORDS]
    intent_match = len(found_intent_token) > 0
    intent_score = 1.0 if intent_match else 0.0
    
    # Signal 3 : cardinality  
    years = set(re.findall(r"\b(20\d{2})\b",q_lower))
    
    found_metrics = {m for m in FINANCIAL_METRICS if m in q_lower}
    

    is_multi_year = len(years) >1
    is_multi_metric = len(found_metrics) > 1
        
    cardinality_score = 1.0 if (is_multi_year and is_multi_metric) else 0.0
    
    
    complexity_score = (intent_score * 0.5) + (cardinality_score * 0.4) + length_score *0.1

    is_complex = complexity_score > 0.3
    
    return {
        "is_complex" : is_complex,
        "complexity_score": round(complexity_score,2),
        "metadata": {
            "word_count":  word_count,
            "years_detected": list(years),
            "metrics_detected": list(found_metrics),
            "intent_tokens": found_intent_token
        }
    }
    
result =  classify_query("Explain the impact of rising interest rates on Apple's cash flow and marketable securities between 2022 and 2023")

print(result)
