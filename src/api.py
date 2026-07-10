"""FastAPI service for fraud detection."""
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import config

# --- Load the model artifact once, at startup ---
try:
    ARTIFACT = joblib.load(config.MODELS_DIR / "fraud_model.joblib")
    MODEL = ARTIFACT["model"]
    CAT_COLS = ARTIFACT["cat_cols"]
    FEATURE_COLS = ARTIFACT["feature_cols"]
    THRESHOLD = ARTIFACT["threshold"]
except Exception as e:
    raise RuntimeError(f"Failed to load model artifact: {e}")

app = FastAPI(
    title="Fraud Detection API",
    description="Scores transactions for fraud probability using a LightGBM model.",
    version="1.1.0",
)


class Transaction(BaseModel):
    """A single transaction. Any subset of features may be provided;
    omitted features are treated as missing (NaN)."""
    features: Dict[str, Any]

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": {
                    "TransactionAmt": 150.0,
                    "ProductCD": "W",
                    "card4": "visa",
                    "card6": "debit",
                    "P_emaildomain": "gmail.com",
                }
            }
        }
    }


class TransactionBatch(BaseModel):
    """A list of transactions to score in one request."""
    transactions: List[Dict[str, Any]]


def build_feature_frame(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Reconstruct a DataFrame matching the model's exact training format,
    for one or many transaction records."""
    rows = [{col: rec.get(col, None) for col in FEATURE_COLS} for rec in records]
    df = pd.DataFrame(rows, columns=FEATURE_COLS)  # enforce exact column order

    # Restore categorical dtype
    for col in CAT_COLS:
        df[col] = df[col].astype("category")

    # Coerce numeric columns; stray strings -> NaN
    for col in FEATURE_COLS:
        if col not in CAT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def score_frame(df: pd.DataFrame) -> np.ndarray:
    """Return fraud probabilities for a prepared feature frame."""
    try:
        return MODEL.predict_proba(df)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")


@app.get("/health")
def health():
    """Liveness check — used by cloud platforms."""
    return {"status": "ok", "model_features": len(FEATURE_COLS), "threshold": THRESHOLD}


@app.post("/predict")
def predict(txn: Transaction):
    """Score a single transaction."""
    if not txn.features:
        raise HTTPException(status_code=422, detail="No features provided.")
    df = build_feature_frame([txn.features])
    prob = float(score_frame(df)[0])
    return {
        "fraud_probability": round(prob, 4),
        "is_fraud": prob >= THRESHOLD,
        "threshold": THRESHOLD,
        "n_features_provided": len(txn.features),
    }


@app.post("/predict_batch")
def predict_batch(batch: TransactionBatch):
    """Score many transactions in one request."""
    if not batch.transactions:
        raise HTTPException(status_code=422, detail="Empty transaction list.")
    if len(batch.transactions) > 1000:
        raise HTTPException(status_code=413, detail="Batch too large (max 1000).")

    df = build_feature_frame(batch.transactions)
    probs = score_frame(df)
    results = [
        {
            "fraud_probability": round(float(p), 4),
            "is_fraud": bool(p >= THRESHOLD),
            "n_features_provided": len(rec),
        }
        for p, rec in zip(probs, batch.transactions)
    ]
    return {"count": len(results), "threshold": THRESHOLD, "results": results}  