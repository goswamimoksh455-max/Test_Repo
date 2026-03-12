# api/main.py
"""
CreditAssist ML API - PyCaret Version
Simple, clean, works perfectly.
"""

import os
import sys
import json
import uuid
import hashlib
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================
# Simple API Key Auth
# ============================================
from fastapi import Security
from fastapi.security import APIKeyHeader

API_KEY = "creditassist-demo-key-2024"
ADMIN_KEY = "creditassist-admin-key-2024"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def check_key(key: str = Security(api_key_header)):
    if key not in [API_KEY, ADMIN_KEY]:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return key

# ============================================
# Request Schema
# ============================================
class ApplicantInput(BaseModel):
    age: int = Field(..., ge=18, le=65)
    gender: int = Field(..., ge=0, le=1)
    education_level: int = Field(..., ge=1, le=4)
    marital_status: int = Field(..., ge=0, le=2)
    dependents: int = Field(..., ge=0)
    city_tier: int = Field(..., ge=1, le=3)
    monthly_income: int = Field(..., ge=0)
    monthly_expenses: int = Field(..., ge=0)
    existing_emi: int = Field(..., ge=0)
    savings_balance: int = Field(..., ge=0)
    bank_account_age_months: int = Field(..., ge=0)
    loan_amount_requested: int = Field(..., ge=1000)
    loan_tenure_months: int = Field(..., ge=1)
    loan_purpose: int = Field(..., ge=1, le=5)
    mobile_verified: int = Field(..., ge=0, le=1)
    upi_transactions_monthly: int = Field(..., ge=0)
    avg_upi_amount: int = Field(..., ge=0)
    social_media_score: int = Field(..., ge=0, le=100)
    phone_os: int = Field(..., ge=0, le=1)
    prev_loans_count: int = Field(..., ge=0)
    prev_defaults: int = Field(..., ge=0)
    credit_utilization_pct: float = Field(..., ge=0, le=100)

    class Config:
        json_schema_extra = {
            "example": {
                "age": 28, "gender": 1, "education_level": 3,
                "marital_status": 0, "dependents": 0, "city_tier": 1,
                "monthly_income": 35000, "monthly_expenses": 18000,
                "existing_emi": 3000, "savings_balance": 25000,
                "bank_account_age_months": 36, "loan_amount_requested": 20000,
                "loan_tenure_months": 12, "loan_purpose": 1,
                "mobile_verified": 1, "upi_transactions_monthly": 25,
                "avg_upi_amount": 600, "social_media_score": 65,
                "phone_os": 1, "prev_loans_count": 1,
                "prev_defaults": 0, "credit_utilization_pct": 35.0
            }
        }

# ============================================
# App Setup
# ============================================
app = FastAPI(
    title="CreditAssist ML API",
    description="Fair & Explainable Microcredit Scoring",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global
explainer = None
audit_log = []
model_meta = {}

@app.on_event("startup")
async def startup():
    global explainer, model_meta
    from model.explain_easy import EasyExplainer
    explainer = EasyExplainer()
    
    with open("artifacts/model_metadata.json") as f:
        model_meta = json.load(f)
    print("API Ready ✓")


def mask_pii(data):
    masked = data.copy()
    for field in ["age", "gender"]:
        if field in masked:
            masked[field] = "***"
    return masked


# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {"service": "CreditAssist ML", "status": "running", "docs": "/docs"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": explainer is not None,
        "model_version": model_meta.get("model_version"),
        "model_hash": model_meta.get("model_hash"),
    }


@app.post("/predict")
async def predict(applicant: ApplicantInput, key: str = Depends(check_key)):
    """Get credit decision + score."""
    data = applicant.model_dump()
    result = explainer.predict(data)

    # Hash for blockchain
    decision_hash = hashlib.sha256(
        json.dumps({"data": str(data), "result": str(result),
                     "model": model_meta.get("model_hash"),
                     "time": datetime.utcnow().isoformat()},
                    sort_keys=True).encode()
    ).hexdigest()

    # Audit log (PII masked)
    audit_log.append({
        "id": str(uuid.uuid4()),
        "time": datetime.utcnow().isoformat(),
        "masked_input": mask_pii(data),
        "decision": result["decision"],
        "score": result["credit_score"],
        "decision_hash": decision_hash,
    })

    return {**result, "decision_hash": decision_hash,
            "model_version": model_meta.get("model_version")}


@app.post("/explain")
async def explain(applicant: ApplicantInput, key: str = Depends(check_key)):
    """Get SHAP + LIME explanations."""
    data = applicant.model_dump()
    return {
        "prediction": explainer.predict(data),
        "shap": explainer.explain_shap(data),
        "lime": explainer.explain_lime(data),
    }


@app.post("/counterfactual")
async def counterfactual(applicant: ApplicantInput, key: str = Depends(check_key)):
    """Get suggestions to flip a rejection."""
    data = applicant.model_dump()
    return explainer.counterfactuals(data)


@app.get("/fairness")
async def fairness(key: str = Depends(check_key)):
    """Get fairness/bias report."""
    return explainer.fairness_check()


@app.get("/model-info")
async def model_info(key: str = Depends(check_key)):
    """Model metadata for blockchain team."""
    return model_meta


@app.get("/audit-log")
async def get_audit(limit: int = 50, key: str = Depends(check_key)):
    """Recent decisions (PII masked)."""
    return {"total": len(audit_log), "entries": audit_log[-limit:]}


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)