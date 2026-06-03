import os 
import re 
import pickle 
import logging
import numpy as np 
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COMPLEXITY_THRESHOLD = float(os.getenv("COMPLEXITY_THRESHOLD", "0.45"))
MODEL_PATH = "classifier.pkl"


# Weaak supervisor Vocabulary 
REASONING_KEYWORDS = {
    "why", "compare", "driven", "impact", "trend" , "difference", "explain","reason", "analysis", "versus", "vs", "evaluate", "analyze", "synthesize",
    "breakdown", "assess", "review", "summarize", "exposure", "risks" }

FINANCIAL_METRICS = {
    "revenue", "sales", "r&d", "research", "expense", "income", "margin",
    "debt", "asset", "liability", "equity", "cash", "flow", "profit", "tax",
    "balance sheet", "ledger", "statement", "footnote", "compliance"
}

def weak_label(query:str) -> int:
    """
    Old Keyword heuristic repurposed as a weak supervisor 
    Return 1 (complex) or 0 (simple) . Used only during training data generation.
    """
    
    q_lower = query.lower()
    clean = re.sub(r'[^\w\s]', ' ',q_lower)
    words = clean.split()
    
    found_intent = [w for w in words if w in REASONING_KEYWORDS]
    intent_score = min(len(found_intent) * 0.5 , 1.0) if found_intent else 0.0
    
    years = set(re.findall(r"\b(20\d{2})\b",q_lower))
    found_metrics = {m for m in  FINANCIAL_METRICS if m in q_lower}
    year_points = 0.5 if len(years) > 1 else (0.2 if len(years) == 1 else 0.0)
    metrics_points = 0.5 if len(found_metrics) > 1 else (0.2  if len(found_metrics) ==  1 else 0.0)
    cardinality_score = min(year_points + metrics_points  , 1.0)
    
    length_score = min(len(words)/20.0 , 1.0)
    score = (intent_score * 0.5) + (cardinality_score * 0.3)  + (length_score *0.2)
    return 1 if score  > 0.35 else 0


# Training dataset generator 
def _generate_training_data():
    simple_templates = [
        "What is the billing date for transaction ID-{n}?",
        "Extract the current outstanding balance from this ledger snippet.",
        "What was the total operating expense listed on page {n}?",
        "Find the net profit metric for Q{q} {year}.",
        "Identify the merchant category code for this vendor transaction.",
        "What is the corporate tax identification number listed in Section {n}?",
        "Read the statement and extract the total credit adjustment value.",
        "What are the payment terms specified on the vendor invoice?",
        "What is the closing balance for account ID-{n}?",
        "List the line items under operating expenses for this quarter.",
        "What is the invoice total for vendor {n}?",
        "Extract the depreciation value from the fixed assets table.",
        "What is the net revenue for Q{q}?",
        "Find the tax rate applied to this transaction.",
        "What is the account number associated with this ledger entry?",
        "Retrieve the total liabilities from the balance sheet.",
        "What is the stated interest rate on this loan agreement?",
        "Extract the gross margin from this income statement.",
        "What is the payment due date on invoice {n}?",
        "Identify the currency type used in this transaction record.",
    ]

    complex_templates = [
        "Synthesize cross-quarter exposure risks and generate a risk-mitigation summary.",
        "Evaluate the structural impact of the shifting debt-to-equity ratio in section {n}.",
        "Compare operating cash flows across all fiscal quarters and pinpoint anomalies.",
        "Perform a comprehensive breakdown of balance sheet changes relative to macro inflation.",
        "Analyze compliance footnotes to assess liability risks across regional subsidiaries.",
        "Why has the revenue trend diverged from the expense growth in {year} vs {year2}?",
        "Evaluate the impact of rising interest rates on the debt structure across Q1 and Q2.",
        "Compare the profit margins of all subsidiaries and identify underperforming segments.",
        "Synthesize the cash flow statement and income statement to assess liquidity risk.",
        "Analyze the multi-year trend in R&D expense relative to revenue growth.",
        "Assess the exposure risk in the derivatives portfolio across all fiscal periods.",
        "Break down the balance sheet changes and explain the drivers behind equity erosion.",
        "Why is the operating margin declining despite stable revenue in {year}?",
        "Compare working capital ratios across Q1, Q2, Q3, and Q4 and flag anomalies.",
        "Evaluate the structural compliance risks flagged in the footnotes of the annual report.",
    ]
    
    queries , labels  = [] , []
    
    for i in range(300):
        t = simple_templates[i%len(simple_templates)]
        q = t.format(n=i+1 , q=(i%4)+1, year=2020+(i%6))
        queries.append(q)
        labels.append(0)
        
    for  i in range(150):
        t = complex_templates[i%len(complex_templates)]
        q = t.format(n=i+1,year=2022+(i%4),  year2=2023+(i%3))
        queries.append(q)
        labels.append(1)
        
    # Human loop in  validation on edge cases 
    verified_overrides= {
        "What is the net revenue for Q3": 0,
        "Retrieve the total liabilities formthe balance sheet.":0,
        "Analyze the multi-year trend in R&D expense relative to revenue growth.": 1,      
    }
    
    for idx , q in enumerate(queries):
        if q in verified_overrides:
            labels[idx] = verified_overrides[q]
            
    logger.info(f"Generated {len(queries)} training sample -"
                f"{labels.count(0)} simple, {labels.count(1)} complex")
    
    return queries, labels

# MOdel Training 
def _train_model() -> Pipeline:
    logger.info("Training IF-IDF + Logistic Regression Classifier ... ")
    queries , labels = _generate_training_data()
    
    X_train , X_test, y_train, y_test = train_test_split(
        queries,labels, test_size=0.2,random_state=42,stratify=labels
    )
    
    pipeline = Pipeline([
        ("tfidf",TfidfVectorizer(
            ngram_range=(1,2),
            max_features=5000,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    
    pipeline.fit(X_train,y_train)
    
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test,y_pred,target_names=["simple","complex"])
    logger.info(f"Classifier evaluation on held-out test set:\n{report}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline,f)
        
    logger.info(f"Model saved to {MODEL_PATH}")
    
    return pipeline

# MOdel looader 
def _load_model() -> Pipeline:
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading classifier from {MODEL_PATH}")
        with open(MODEL_PATH, "rb")  as f:
            return pickle.load(f)
    logger.info("No saved model  found - Training from scratch")
    return _train_model()


# Load once at import time 
_model: Pipeline = _load_model()

# INferance 
def classify_query(query:str) -> dict:
    """
    Classifies a financial query as simple or complex 
    Return same dict shape as the old keyword classifier for  drop in compatibility
    """
    words = query.strip().split()
    
    # Guard : Under 4 words -> simple by default 
    if len(words) < 4:
        logger.debug(f"Query too short ({len(words)} words ) - defaulting to simple")
        return {
            "is_complex": False,
            "complexity_score": 0.0,
            "metadata": {
                "word_count":len(words),
                "reason": "below_minimum_length",
                "model": "default"
            }
        }
        
    proba = _model.predict_proba([query])[0]
    complexity_score = float(proba[1]) # probability of class 1  (complex)
    is_complex = complexity_score > COMPLEXITY_THRESHOLD
    
    return {
        "is_complex": is_complex,
        "complexity_score": round(complexity_score,4),
        "metadata": {
            "word_count": len(words),
            "simple_proba":round(float(proba[0]),4),
            "complex_proba": round(float(proba[1]),4),
            "threshold": COMPLEXITY_THRESHOLD,
            "model": "tfidf_logreg"
        }
    }