from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import date
import inference as inf

app = FastAPI(
    title="Motor Insurance Premium API",
    description="API for calculating motor insurance premium using Hybrid GLM + ANFIS model."
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PremiumRequest(BaseModel):
    birth_date: date
    exp: int
    matriculation_year: int
    vehicle_value: float
    cc: int
    loading: float = 20.0

@app.post("/predict")
def predict_premium(req: PremiumRequest):
    try:
        # Calculate derived fields for the model based on current date (or 2026 reference)
        # Match training script math exactly (simple year subtraction)
        current_date = date.today()
        
        age = max(18, current_date.year - req.birth_date.year)
        v_age = max(0, current_date.year - req.matriculation_year)

        # Request predictions from the model script
        result = inf.predict_pure_premium(
            age, req.exp, v_age, req.vehicle_value, req.cc
        )
        pure_premium, glm_pp, nf_pp, glm_freq, nf_freq = result
        
        # Calculate final premium based on loading
        loading_pct = req.loading / 100.0
        final_premium = pure_premium * (1 + loading_pct)
        loading_amount = final_premium - pure_premium
        
        # Get top fuzzy rule combination and its intrinsic risk score
        x_base = __import__('numpy').array([[age, req.exp, v_age, req.vehicle_value, req.cc]], dtype=__import__('numpy').float32)
        rule_name, rule_weight, rule_score = inf.get_rule_info(x_base)
        
        # Evaluate risk tier strictly from the active fuzzy rule's trained parametric constraints
        if rule_score < 0.85:
            risk_tier = "low"
            risk_label = "LOW RISK"
        elif rule_score < 1.15:
            risk_tier = "medium"
            risk_label = "MEDIUM RISK"
        else:
            risk_tier = "high"
            risk_label = "HIGH RISK"

        return {
            "status": "success",
            "premiums": {
                "pure_premium": pure_premium,
                "final_premium": final_premium,
                "loading_amount": loading_amount
            },
            "risk": {
                "tier": risk_tier,
                "label": risk_label,
                "ratio": rule_score,
                "rule_name": rule_name,
                "rule_weight": rule_weight
            },
            "breakdown": {
                "glm_pp": glm_pp,
                "nf_pp": nf_pp,
                "portfolio_avg": inf.obs_pure_prem,
                "glm_freq": glm_freq,
                "nf_freq": nf_freq
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Motor Insurance Premium API is running!"}
