# ============================================================
# CREDIT RISK API — FastAPI Serving
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import logging
import time
import os
import json

from src.model_loader         import load_latest_model
from services.prediction_service import predict_applicant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Credit Risk Prediction API")

# ── Load model on startup ─────────────────────────────────────
try:
    model, threshold = load_latest_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error("Model loading failed: %s", e)
    model     = None
    threshold = 0.5


# ============================================================
# INPUT SCHEMA
# ============================================================

class ApplicantInput(BaseModel):
    age:        float
    income:     float       # annual income (000s)
    family:     int         # family size
    ccavg:      float       # avg monthly credit card spend (000s)
    education:  int         # 1=undergrad, 2=grad, 3=advanced
    mortgage:   float       # mortgage value (000s)
    securities_account: int = 0
    cd_account:         int = 0
    online:             int = 0
    creditcard:         int = 0


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Credit Risk Prediction API is live 🚀",
        "docs":    "/docs",
        "health":  "/health"
    }

@app.get("/health")
def health():
    return {"status": "running", "model_loaded": model is not None}

@app.get("/model_info")
def model_info():
    registry_path = "risk_models/latest_model.json"
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            return json.load(f)
    return {"error": "Model registry not found"}


# ── Prediction endpoint ───────────────────────────────────────

@app.post("/predict")
def predict(applicant: ApplicantInput):

    start      = time.time()
    input_data = applicant.dict()

    result     = predict_applicant(model, input_data, threshold)

    # ── Log prediction ────────────────────────────────────────
    log_record = {
        "timestamp":        time.time(),
        "age":              input_data["age"],
        "income":           input_data["income"],
        "risk_probability": result["risk_probability"],
        "risk_band":        result["risk_band"],
        "decision":         result["decision"],
        "rule_triggered":   result.get("rule_triggered"),
    }

    log_path = "logs/prediction_logs.csv"
    os.makedirs("logs", exist_ok=True)

    log_df = pd.DataFrame([log_record])
    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)

    result["latency_seconds"] = round(time.time() - start, 4)

    return result