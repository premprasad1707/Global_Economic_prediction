# 🌍 EconAI: Global Economic Intelligence Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**EconAI** is a premium, state-of-the-art AI-driven dashboard designed for analyzing and predicting global economic trends. Built with a sleek Glassmorphism design and powered by advanced machine learning models, it provides deep insights into GDP growth, inflation, and manufacturing output.

---

## ✨ Key Features

- **🔮 Predictive Analytics**: Forecast GDP and Growth rates using an ensemble of Ridge, Random Forest, XGBoost, and LightGBM models.
- **🚀 Turbo Mode**: Switch between high-precision ensembles and near-instant linear models for rapid analysis.
- **📊 Interactive Visualizations**: Deep-dive into economic correlations, residuals, and feature importance with Plotly.
- **🌐 Live Proxies**: Fetch real-time environmental data as proxies for localized economic activity using the OpenWeatherMap API.
- **💾 Persistent Intelligence**: Trained models are automatically persisted to disk for instant startup.
- **📥 Data Export**: Standardized CSV exports of cleaned and engineered datasets.

## 🛠️ Technology Stack

- **Core**: Python 3.9+
- **Frontend**: Streamlit (Premium Custom CSS)
- **Data Science**: Pandas, NumPy, Scikit-Learn
- **ML Models**: XGBoost, LightGBM
- **Visuals**: Plotly Express & Graph Objects
- **Live Data**: Requests (REST API Integration)

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python installed. We recommend creating a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Launch the Dashboard
```bash
streamlit run app.py
```

## 📂 Project Structure

- `app.py`: Main entry point and UI logic.
- `src/`: Core logic modules.
  - `data_loader.py`: Robust CSV handling and synthetic fallback.
  - `preprocessing.py`: Multi-stage data cleaning and standardization.
  - `feature_engineering.py`: Advanced economic indicator creation (Leakage-free).
  - `train_models.py`: Automated ensemble training and persistence.
  - `predict.py`: Real-time prediction and confidence scoring.
  - `evaluate.py`: Model benchmarking and visualization utilities.
- `data/`: Location for `World Economy Statistics.csv`.
- `models/`: Persistent storage for serialized ML models.
- `logs/`: Application execution logs.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for suggestions.

---
*Created with ❤️ by the EconAI Team*
