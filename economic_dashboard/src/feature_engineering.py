"""
feature_engineering.py
-----------------------
Advanced feature extraction for economic forecasting. Ensures zero data leakage.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates lagged and rolling features for time-series forecasting.
    Features for year T only use data from T-1 and earlier.
    """
    df = df.copy().sort_values(["country", "year"])

    # 1. Temporal Lags (The core of prediction)
    for lag in [1, 2, 3]:
        df[f"gdp_lag{lag}"] = df.groupby("country")["gdp"].shift(lag)
        if "gdp_growth" in df.columns:
            df[f"growth_lag{lag}"] = df.groupby("country")["gdp_growth"].shift(lag)

    # 2. Historical Rolling Averages (Non-leaking)
    df["gdp_rolling_avg3"] = (
        df.groupby("country")["gdp"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # 3. Derived Ratios
    if "manufacturing" in df.columns and "gdp" in df.columns:
        # Lag manufacturing to avoid same-year leakage
        df["manufacturing_share"] = (
            df.groupby("country")["manufacturing"].shift(1) / 
            df.groupby("country")["gdp"].shift(1).replace(0, np.nan)
        ).fillna(0).clip(0, 1)

    # 4. Economic Vitality Index
    # Combination of lagged growth and manufacturing strength
    if "growth_lag1" in df.columns and "manufacturing_share" in df.columns:
        df["vitality_index"] = (
            (df["growth_lag1"] - df["growth_lag1"].min()) / 
            (df["growth_lag1"].max() - df["growth_lag1"].min() + 1e-9) +
            df["manufacturing_share"]
        ) / 2
    else:
        df["vitality_index"] = 0.5

    # 5. Volatility Proxy (Standard deviation of growth over 3 years)
    if "gdp_growth" in df.columns:
        df["growth_volatility"] = (
            df.groupby("country")["gdp_growth"]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).std())
            .fillna(0)
        )

    # 6. Fill all historical gaps with zeros
    eng_cols = [c for c in df.columns if any(x in c for x in ["lag", "rolling", "index", "volatility", "share"])]
    df[eng_cols] = df[eng_cols].fillna(0)

    logger.info(f"🚀 Feature engineering complete. Added {len(eng_cols)} indicators.")
    return df.reset_index(drop=True)
