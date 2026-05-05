"""
preprocessing.py
----------------
Advanced cleaning and numerical standardization for economic datasets.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

NUMERIC_COLS = ["gdp", "gdp_growth", "gdp_per_capita", "gni", "gnp", "inflation", "manufacturing"]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw data: strips symbols, handles NaNs via interpolation, and clamps outliers.
    """
    df = df.copy()
    
    # 1. Clean numeric strings
    for col in [c for c in NUMERIC_COLS if c in df.columns]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = (
                df[col].astype(str)
                .str.replace(r"[^\d.-]", "", regex=True)
                .pipe(pd.to_numeric, errors="coerce")
            )

    # 2. Year standardization
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # 3. Drop critical missingness
    df = df.dropna(subset=["country", "year"])
    df["country"] = df["country"].astype(str).str.strip()

    # 4. Intra-country interpolation
    df = df.sort_values(["country", "year"])
    numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
    df[numeric_present] = (
        df.groupby("country")[numeric_present]
        .transform(lambda x: x.ffill().bfill())
    )

    # 5. Outlier management (4-sigma clamping)
    for col in numeric_present:
        mu, sigma = df[col].mean(), df[col].std()
        if sigma > 0:
            df[col] = df[col].clip(mu - 4 * sigma, mu + 4 * sigma)

    # 6. Global zero-handling (GDP cannot be zero)
    for col in ["gdp", "gni", "gnp"]:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan).ffill().bfill()

    logger.info(f"✨ Data cleaning complete. Final shape: {df.shape}")
    return df.reset_index(drop=True)
