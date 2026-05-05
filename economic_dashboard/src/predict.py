"""
predict.py
----------
Confidence-aware prediction engine.
"""

import numpy as np
import pandas as pd

def build_input_vector(latest_row: pd.Series, year: int, features: list) -> np.ndarray:
    """
    Creates a prediction vector for a future year based on the most recent historical data.
    """
    # Map historical row values to expected features
    row_data = {
        "gdp_lag1": latest_row.get("gdp", 0),
        "gdp_lag2": latest_row.get("gdp_lag1", 0),
        "gdp_lag3": latest_row.get("gdp_lag2", 0),
        "gdp_rolling_avg3": (latest_row.get("gdp", 0) + latest_row.get("gdp_lag1", 0) + latest_row.get("gdp_lag2", 0)) / 3,
        "growth_lag1": latest_row.get("gdp_growth", 0),
        "manufacturing_share": latest_row.get("manufacturing_share", 0.1),
        "vitality_index": latest_row.get("vitality_index", 0.5),
        "growth_volatility": latest_row.get("growth_volatility", 0.0),
        "year": year
    }
    
    return np.array([row_data.get(f, 0) for f in features]).reshape(1, -1)

def get_prediction(model_res: dict, X: np.ndarray, target: str) -> dict:
    """Calculates prediction and confidence metadata."""
    model = model_res["model"]
    pred = float(model.predict(X)[0])
    
    # Confidence Score: Based on R2 (clamped)
    confidence = max(0.0, min(1.0, model_res.get("cv_r2", 0.5)))
    
    # Status Assessment
    if target == "gdp":
        if pred > 1e13: status = "🏆 Global Superpower"
        elif pred > 1e12: status = "🟢 Major Economy"
        elif pred > 1e11: status = "🟡 Emerging Market"
        else: status = "⚪ Developing Economy"
    else:
        if pred > 5: status = "🚀 High Growth"
        elif pred > 2: status = "📈 Steady Expansion"
        elif pred > 0: status = "🟡 Slow Growth"
        else: status = "🚨 Contraction Risk"

    return {
        "value": pred,
        "confidence": confidence,
        "status": status
    }
