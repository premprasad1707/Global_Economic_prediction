"""
data_loader.py
--------------
Handles robust data loading with multi-path discovery and canonical mapping.
"""

import os
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Canonical Column Mapping ──────────────────────────────────────────────────
COLUMN_MAP = {
    "country": ["country", "nation", "economy", "Country"],
    "year": ["year", "Year", "period"],
    "gdp": ["gdp", "GDP", "gross domestic product"],
    "gdp_growth": ["gdp growth rate", "GDP Growth Rate", "economic growth"],
    "gdp_per_capita": ["gdp per capita", "GDP Per Capita"],
    "gni": ["gni", "GNI", "gross national income"],
    "gnp": ["gnp", "GNP", "gross national product"],
    "inflation": ["inflation rate", "Inflation Rate", "inflation"],
    "manufacturing": ["manufacturing output", "Manfacturing Output Rate", "manufacturing"],
}

def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Discovery-based data loader with synthetic fallback.
    """
    search_paths = [
        filepath,
        "data/World Economy Statistics.csv",
        "World Economy Statistics.csv",
        os.path.join(os.path.dirname(__file__), "..", "data", "World Economy Statistics.csv")
    ]

    df = None
    for path in [p for p in search_paths if p]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, encoding="utf-8")
                logger.info(f"✅ Successfully loaded data from {path}")
                break
            except Exception as e:
                logger.error(f"❌ Failed to load {path}: {e}")

    if df is None:
        logger.warning("⚠️ No data source found. Initializing synthetic benchmark data.")
        df = _generate_synthetic()

    return _standardize_columns(df)

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Maps varied column names to canonical internal names."""
    rename_map = {}
    cols = {c.lower().strip(): c for c in df.columns}
    
    for canonical, aliases in COLUMN_MAP.items():
        for alias in aliases:
            if alias.lower() in cols:
                rename_map[cols[alias.lower()]] = canonical
                break
                
    return df.rename(columns=rename_map)

def _generate_synthetic() -> pd.DataFrame:
    """Standard synthetic dataset generation for fallback."""
    np.random.seed(42)
    countries = ["USA", "China", "India", "Germany", "UK", "Brazil", "Japan"]
    years = range(2010, 2024)
    data = []
    for c in countries:
        base_gdp = np.random.uniform(1e12, 2e13)
        for y in years:
            growth = np.random.uniform(-0.02, 0.08)
            base_gdp *= (1 + growth)
            data.append({
                "country": c, "year": y, "gdp": base_gdp,
                "gdp_growth": growth * 100, "inflation": np.random.uniform(1, 10),
                "manufacturing": base_gdp * np.random.uniform(0.1, 0.3)
            })
    return pd.DataFrame(data)

COUNTRY_CAPITALS = {
    "United States": "Washington D.C.", "China": "Beijing", "Japan": "Tokyo",
    "Germany": "Berlin", "India": "New Delhi", "United Kingdom": "London",
    "France": "Paris", "Brazil": "Brasilia", "Italy": "Rome", "Canada": "Ottawa",
    "Russia": "Moscow", "Australia": "Canberra", "Spain": "Madrid",
    "Mexico": "Mexico City", "Indonesia": "Jakarta", "Netherlands": "Amsterdam",
    "Argentina": "Buenos Aires", "Afghanistan": "Kabul", "Pakistan": "Islamabad",
    "Bangladesh": "Dhaka", "Nigeria": "Abuja", "Egypt": "Cairo"
}
