"""
app.py
------
EconAI: Premium Global Economic Intelligence Dashboard.
"""

import os
import logging
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

# Local modules
from src.data_loader import load_data, COUNTRY_CAPITALS
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features
from src.train_models import train_all, load_all_models, get_best_model
from src.predict import build_input_vector, get_prediction
from src.evaluate import plot_feature_importance, plot_residuals, get_metrics_df, generate_pdf_report

# ── Configuration ────────────────────────────────────────────────────────────
st.set_page_config(page_title="EconAI Dashboard", page_icon="🌍", layout="wide")

# ── UI Customization (Glassmorphism 2.0) ──────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    .stApp {
        background: radial-gradient(circle at 20% 20%, #1a1c23 0%, #0d1117 100%);
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }
    
    /* Premium Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(88, 166, 255, 0.4);
        background: rgba(255, 255, 255, 0.05);
    }
    
    /* Metrics Text */
    div[data-testid="stMetricValue"] { color: #58a6ff !important; font-weight: 800 !important; }
    div[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 1rem !important; }

    /* Headers */
    h1, h2, h3 { color: #ffffff !important; letter-spacing: -0.5px; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #30363d;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #1f6feb 0%, #3b82f6 100%);
        color: white; border: none; border-radius: 10px;
        padding: 0.75rem 2rem; font-weight: 600; width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.6);
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# ── Data & Model Engine ───────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_processed_data():
    raw_df = load_data()
    clean_df = clean_data(raw_df)
    return engineer_features(clean_df)

@st.cache_resource
def get_models(target, turbo_mode=False, force_retrain=False):
    if not force_retrain:
        saved = load_all_models(target)
        if saved and (not turbo_mode or "Linear Ridge" in saved):
            return saved
            
    df = get_processed_data()
    return train_all(df, target, fast_mode=turbo_mode)

# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(df):
    st.sidebar.title("🌍 EconAI")
    st.sidebar.caption("v2.0 Premium Intelligence")
    st.sidebar.divider()
    
    page = st.sidebar.selectbox("Navigation", ["Overview", "Predictions", "Analytics", "Live Insights", "Export"])
    
    st.sidebar.divider()
    st.sidebar.subheader("Configuration")
    
    countries = sorted(df["country"].unique())
    selected_countries = st.sidebar.multiselect("Active Economies", countries, default=countries[:5])
    
    turbo = st.sidebar.toggle("🚀 Turbo Mode", value=True)
    force = st.sidebar.button("🔄 Force Retrain")
    
    st.sidebar.divider()
    api_key = st.sidebar.text_input("OpenWeather API Key", type="password")
    
    return page, selected_countries, turbo, force, api_key

# ── Page: Overview ────────────────────────────────────────────────────────────
def page_overview(df):
    st.title("📊 Global Economic Pulse")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Global Avg GDP", f"${df['gdp'].mean()/1e12:.2f}T")
    col2.metric("Avg Growth Rate", f"{df['gdp_growth'].mean():.2f}%")
    col3.metric("Inflation index", f"{df['inflation'].mean():.1f}%")
    col4.metric("Active Regions", len(df['country'].unique()))

    st.divider()
    
    # Global Map
    st.subheader("🌐 Global Economic Distribution")
    fig_map = px.choropleth(df, locations="country", locationmode="country names",
                            color="gdp", hover_name="country",
                            color_continuous_scale="Viridis",
                            title="GDP Distribution by Country")
    fig_map.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", 
                          plot_bgcolor="rgba(0,0,0,0)", geo=dict(bgcolor='rgba(0,0,0,0)'),
                          height=500)
    st.plotly_chart(fig_map, use_container_width=True)

    st.divider()
    
    # GDP Trend Plot
    gdp_trend = df.groupby("year")["gdp"].sum().reset_index()
    fig = px.area(gdp_trend, x="year", y="gdp", title="Total GDP Growth (Aggregated)",
                  line_shape='spline', color_discrete_sequence=['#58a6ff'])
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

# ── Page: Predictions ─────────────────────────────────────────────────────────
def page_predictions(df, turbo):
    st.title("🔮 Predictive Intelligence")
    
    with st.container():
        c1, c2, c3 = st.columns(3)
        target = c1.selectbox("Forecast Target", ["gdp", "gdp_growth"])
        country = c2.selectbox("Economy", sorted(df["country"].unique()))
        future_year = c3.slider("Target Year", 2024, 2035, 2026)
    
    if st.button("Generate Forecast"):
        with st.spinner("Analyzing temporal patterns..."):
            models = get_models(target, turbo_mode=turbo, force_retrain=st.session_state.get('force_retrain', False))
            best_name, best_res = get_best_model(models)
            
            if not best_res:
                st.error("Engine failure: Could not train or load models.")
                return
            
            latest = df[df["country"] == country].iloc[-1]
            X = build_input_vector(latest, future_year, best_res["features"])
            pred_data = get_prediction(best_res, X, target)
            
            st.success(f"Forecast complete using **{best_name}**")
            
            m1, m2, m3 = st.columns(3)
            val = f"${pred_data['value']/1e9:.2f}B" if target == "gdp" else f"{pred_data['value']:.2f}%"
            m1.metric(f"Predicted {target.upper()}", val)
            m2.metric("Confidence Level", f"{pred_data['confidence']:.1%}")
            m3.metric("Economic Status", pred_data['status'])
            
            st.progress(pred_data['confidence'])

# ── Page: Analytics ───────────────────────────────────────────────────────────
def page_analytics(df, turbo):
    st.title("📈 Model Benchmarking")
    
    target = st.segmented_control("Evaluation Base", ["gdp", "gdp_growth"], default="gdp")
    models = get_models(target, turbo_mode=turbo, force_retrain=st.session_state.get('force_retrain', False))
    
    if not models:
        st.warning("No models available. Try disabling Turbo Mode for full ensemble training.")
        return
        
    st.dataframe(get_metrics_df(models), use_container_width=True)
    
    b_name, b_res = get_best_model(models)
    
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_feature_importance(b_res), use_container_width=True)
    with c2:
        # Residuals require actual test run
        data_plot = df[b_res["features"] + [target]].dropna().tail(100)
        y_true = data_plot[target].values
        y_pred = b_res["model"].predict(data_plot[b_res["features"]].values)
        st.plotly_chart(plot_residuals(y_true, y_pred), use_container_width=True)

# ── Page: Live Insights ───────────────────────────────────────────────────────
def page_live(df, api_key):
    st.title("🌐 Live Economic Proxies")
    st.info("Using real-time environmental data as a proxy for localized economic activity levels.")
    
    country = st.selectbox("Market Selection", sorted(df["country"].unique()))
    capital = COUNTRY_CAPITALS.get(country, country)
    
    if not api_key:
        st.warning("Please provide an OpenWeather API Key in the sidebar to activate live streams.")
        return
        
    url = f"https://api.openweathermap.org/data/2.5/weather?q={capital}&appid={api_key}&units=metric"
    try:
        data = requests.get(url).json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Local Temp", f"{data['main']['temp']}°C")
        c2.metric("Humidity", f"{data['main']['humidity']}%")
        c3.metric("Visibility", f"{data['visibility']/1000}km")
    except:
        st.error("Failed to connect to OpenWeather. Verify API key and connectivity.")

# ── Main Loop ─────────────────────────────────────────────────────────────────
def main():
    df = get_processed_data()
    page, countries, turbo, force, api_key = render_sidebar(df)
    
    # Filter by countries
    if countries:
        df = df[df["country"].isin(countries)]
        
    if page == "Overview":
        page_overview(df)
    elif page == "Predictions":
        page_predictions(df, turbo)
    elif page == "Analytics":
        page_analytics(df, turbo)
    elif page == "Live Insights":
        page_live(df, api_key)
    elif page == "Export":
        st.title("📥 Export Center")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Data Export")
            st.write(f"Standardizing dataset with {len(df)} records.")
            st.download_button("🚀 Download Cleaned Dataset (CSV)", df.to_csv(index=False).encode('utf-8'), 
                               "econai_dataset.csv", "text/csv")
        
        with c2:
            st.subheader("Intelligence Report")
            st.write("Generate a high-level executive summary in PDF format.")
            target = st.selectbox("Report Base", ["gdp", "gdp_growth"], key="report_target")
            if st.button("📄 Generate PDF Report"):
                models = get_models(target)
                if models:
                    pdf_path = generate_pdf_report(df, models, target)
                    with open(pdf_path, "rb") as f:
                        st.download_button("📥 Download PDF Report", f, f"econai_report_{target}.pdf", "application/pdf")
                else:
                    st.error("No models available to generate report.")

        st.divider()
        st.info("💡 **Model Persistence**: All trained models are automatically saved to the `models/` directory for instant access on your next session.")

if __name__ == "__main__":
    main()