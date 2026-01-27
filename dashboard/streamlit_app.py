import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import os

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
st.set_page_config(page_title="FraudGuard | Monitoring Dashboard", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .kpi-card {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }
    .kpi-label { color: #9CA3AF; font-size: 0.85rem; margin-bottom: 0.25rem; }
    .kpi-value { color: #F9FAFB; font-size: 1.9rem; font-weight: 700; line-height: 1; }
    .kpi-sub { color: #D1D5DB; font-size: 0.78rem; margin-top: 0.35rem; }
    .section-title { font-size: 1.05rem; font-weight: 700; margin: 0.2rem 0 0.8rem 0; }
    .tiny-muted { color: #6B7280; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)

st_autorefresh(interval=10000, key="refresh")

@st.cache_data(ttl=10)
def fetch_alerts():
    try:
        res = requests.get(f"{API_BASE}/alerts", timeout=4)
        res.raise_for_status()
        data = res.json()
        return data.get("alerts", [])
    except Exception:
        return []

@st.cache_data(ttl=10)
def fetch_stats():
    try:
        res = requests.get(f"{API_BASE}/stats", timeout=4)
        res.raise_for_status()
        return res.json()
    except Exception:
        return {}

alerts = fetch_alerts()
stats = fetch_stats()

st.sidebar.title("Controls")
show_only_fraud = st.sidebar.checkbox("Show fraud only", value=False)
min_score = st.sidebar.slider("Minimum fraud score", 0.0, 1.0, 0.0, 0.01)
top_n = st.sidebar.slider("Rows to show", 10, 200, 50, 10)
auto_refresh = st.sidebar.toggle("Auto refresh", value=True)

if not auto_refresh:
    st_autorefresh(interval=0, key="pause_refresh")

st.title("🛡️ FraudGuard Monitoring Dashboard")
st.caption("Portfolio-grade real-time fraud monitoring built on FastAPI + Streamlit.")

if alerts:
    df = pd.DataFrame(alerts).copy()
    for col in ["amount", "fraud_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "amount", "fraud_score"])
    df = df.sort_values("timestamp", ascending=False)
else:
    df = pd.DataFrame(columns=["transaction_id", "user_id", "amount", "location", "fraud_score", "timestamp"])

if show_only_fraud:
    df = df[df["fraud_score"] >= 0.5]
df = df[df["fraud_score"] >= min_score]

total_alerts = stats.get("fraud_detected", len(df))
filtered_rows = len(df)
avg_score_api = stats.get("fraud_score_avg", None)
threshold_api = stats.get("threshold", None)

if isinstance(threshold_api, float):
    threshold_api = f"{threshold_api:.3f}"
elif threshold_api is None:
    threshold_api = "N/A"

if avg_score_api is None:
    avg_score_display = "N/A"
else:
    avg_score_display = f"{float(avg_score_api):.3f}"

api_status = "Online" if (stats or not df.empty) else "Offline"

c1, c2, c3, c4 = st.columns(4)

cards = [
    ("Total Alerts", total_alerts, "From backend stats"),
    ("Shown Rows", filtered_rows, "After active filters"),
    ("Avg Fraud Score", avg_score_display, "From backend stats"),
    ("Threshold", threshold_api, "From backend stats"),
]

for col, (label, value, sub) in zip([c1, c2, c3, c4], cards):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

left, right = st.columns([1.2, 0.8])

with left:
    st.markdown('<div class="section-title">Fraud score over time</div>', unsafe_allow_html=True)
    if not df.empty:
        time_df = df.sort_values("timestamp")
        fig = px.line(
            time_df,
            x="timestamp",
            y="fraud_score",
            color_discrete_sequence=["#ef4444"],
            markers=True,
            hover_data=["transaction_id", "user_id", "amount", "location"]
        )
        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Time",
            yaxis_title="Fraud Score",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No alerts available yet.")

with right:
    st.markdown('<div class="section-title">Alerts by location</div>', unsafe_allow_html=True)
    if not df.empty:
        loc_df = df.groupby("location", as_index=False).size().sort_values("size", ascending=False)
        fig2 = px.bar(
            loc_df,
            x="location",
            y="size",
            color="size",
            color_continuous_scale=["#93c5fd", "#1d4ed8", "#7f1d1d"],
            text="size"
        )
        fig2.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Location",
            yaxis_title="Alerts",
            template="plotly_dark",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No location data to show yet.")

st.write("")

st.markdown('<div class="section-title">Recent alerts</div>', unsafe_allow_html=True)

if not df.empty:
    table_df = df[["timestamp", "transaction_id", "user_id", "amount", "location", "fraud_score"]].head(top_n).copy()
    table_df["fraud_score"] = table_df["fraud_score"].map(lambda x: f"{x:.3f}")
    table_df["amount"] = table_df["amount"].map(lambda x: f"₹{x:,.2f}")
    table_df["timestamp"] = table_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(table_df, use_container_width=True, hide_index=True)
else:
    st.info("No alerts have been logged yet.")

st.write("")
st.markdown('<div class="tiny-muted">Designed for portfolio presentation: live API-backed monitoring, KPI cards, charts, filters, and responsive layout.</div>', unsafe_allow_html=True)