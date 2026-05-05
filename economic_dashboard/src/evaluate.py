"""
evaluate.py
-----------
Visualization and benchmarking utilities for economic models.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def plot_feature_importance(model_res: dict) -> go.Figure:
    """Extracts and plots importance from linear or tree models."""
    model = model_res["model"]
    features = model_res["features"]
    
    # Extract importance
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps["m"]
        
    if hasattr(estimator, "feature_importances_"):
        imps = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        imps = np.abs(estimator.coef_)
    else:
        imps = [1/len(features)] * len(features)
        
    df = pd.DataFrame({"Feature": features, "Importance": imps}).sort_values("Importance", ascending=True)
    
    fig = px.bar(df, x="Importance", y="Feature", orientation="h", 
                 title="Key Economic Drivers", color="Importance",
                 color_continuous_scale="RdBu")
    
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def plot_residuals(y_true, y_pred) -> go.Figure:
    """Plots prediction errors vs truth."""
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', 
                             marker=dict(color='#58a6ff', opacity=0.6)))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(title="Residual Analysis", xaxis_title="Predicted", yaxis_title="Error",
                      template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def get_metrics_df(results: dict) -> pd.DataFrame:
    """Summarizes all models into a comparison table."""
    data = []
    for name, res in results.items():
        data.append({
            "Model": name,
            "R² Score": f"{res.get('cv_r2', 0):.4f}",
            "RMSE": f"{res.get('cv_rmse', 0):.4g}"
        })
    return pd.DataFrame(data).sort_values("R² Score", ascending=False)

def generate_pdf_report(df, models, target):
    """Generates a simple PDF summary of the analysis."""
    from fpdf import FPDF
    import tempfile
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, "EconAI: Global Intelligence Report", ln=True, align="C")
    
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 10, f"Target Metric: {target.upper()}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Global Summary Statistics", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Average {target}: {df[target].mean():.2f}", ln=True)
    pdf.cell(0, 10, f"Total records analyzed: {len(df)}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Model Performance Benchmark", ln=True)
    pdf.set_font("Arial", "", 12)
    for name, res in models.items():
        pdf.cell(0, 8, f"- {name}: R2 = {res.get('cv_r2', 0):.4f}", ln=True)
        
    pdf.ln(20)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, "CONFIDENTIAL - EconAI Intelligence Engine", ln=True, align="C")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        return tmp.name
