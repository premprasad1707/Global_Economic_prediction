"""
train_models.py
---------------
Persistent ML pipeline for economic forecasting. Supports Turbo and Ensemble modes.
"""

import os
import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Feature sets
CORE_FEATURES = [
    "gdp_lag1", "gdp_lag2", "gdp_lag3", "gdp_rolling_avg3",
    "growth_lag1", "manufacturing_share", "vitality_index",
    "growth_volatility", "year"
]

def train_all(df: pd.DataFrame, target: str = "gdp", fast_mode: bool = False, models_dir: str = "models") -> dict:
    """Trains and persists an ensemble of models."""
    os.makedirs(models_dir, exist_ok=True)
    
    feats = [c for c in CORE_FEATURES if c in df.columns]
    data = df.dropna(subset=feats + [target])
    
    if len(data) < 10:
        logger.warning("Insufficient data for robust training.")
        return {}

    X, y = data[feats].values, data[target].values
    results = {}
    
    models = {
        "Linear Ridge": Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))])
    }
    
    if not fast_mode:
        models.update({
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
        })

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        try:
            cv = cross_validate(model, X, y, cv=kfold, scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"})
            model.fit(X, y)
            
            res = {
                "model": model,
                "features": feats,
                "cv_r2": float(cv["test_r2"].mean()),
                "cv_rmse": float(-cv["test_rmse"].mean())
            }
            results[name] = res
            
            # Persistence
            with open(os.path.join(models_dir, f"{name.lower().replace(' ', '_')}_{target}.pkl"), "wb") as f:
                pickle.dump(res, f)
                
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")

    return results

def load_all_models(target: str, models_dir: str = "models") -> dict:
    """Loads all persisted models for a target."""
    results = {}
    if not os.path.exists(models_dir): return results
    
    for f in os.listdir(models_dir):
        if f.endswith(f"_{target}.pkl"):
            try:
                name = f.replace(f"_{target}.pkl", "").replace("_", " ").title()
                with open(os.path.join(models_dir, f), "rb") as pf:
                    results[name] = pickle.load(pf)
            except: continue
    return results

def get_best_model(results: dict) -> tuple:
    if not results: return None, None
    best_name = max(results.keys(), key=lambda k: results[k].get("cv_r2", 0))
    return best_name, results[best_name]
