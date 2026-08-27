"""
Anomaly Detection — monitoring dashboard.

Streamlit + Plotly. Single-line HTML strings throughout to avoid Streamlit's
markdown renderer breaking on blank lines inside multi-line HTML blocks.
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
REFRESH_MS = 10_000

st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Theme ────────────────────────────────────────────────────────────────
C = {
    "bg":        "#0a0e14",
    "surface":   "#11161d",
    "surface_2": "#161c25",
    "border":    "#1f2733",
    "border_lo": "#171d26",
    "text":      "#e6edf3",
    "text_dim":  "#8b96a5",
    "text_mute": "#5c6675",
    "ok":        "#10b981",
    "low":       "#3b82f6",
    "med":       "#f59e0b",
    "high":      "#ef4444",
    "crit":      "#dc2626",
    "accent":    "#60a5fa",
    "muted_blue":"#1e3a5f",
}

# CSS — this is allowed to be multi-line, it's a style block
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', -apple-system, system-ui, sans-serif !important; }
.stApp { background: #0a0e14; }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1400px; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
div[data-testid="stToolbar"] { display: none; }

.topbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 22px; padding-bottom: 16px; border-bottom: 1px solid #171d26; }
.brand { display: flex; align-items: center; gap: 14px; }
.brand-mark { width: 36px; height: 36px; display: grid; place-items: center; background: linear-gradient(135deg, #ef4444 0%, #7f1d1d 100%); border-radius: 10px; font-size: 18px; box-shadow: 0 6px 16px rgba(239,68,68,0.25); }
.brand-title { color: #e6edf3; font-size: 1.15rem; font-weight: 700; letter-spacing: -0.01em; line-height: 1; }
.brand-sub { color: #5c6675; font-size: 0.78rem; font-weight: 500; letter-spacing: 0.02em; text-transform: uppercase; margin-top: 4px; }
.status-pill { display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 999px; background: #11161d; border: 1px solid #1f2733; color: #8b96a5; font-size: 0.78rem; font-weight: 500; }
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: #10b981; box-shadow: 0 0 0 0 rgba(16,185,129,0.7); animation: pulse 2s infinite; }
.status-dot.offline { background: #ef4444; animation: none; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(16,185,129,0.6); } 70% { box-shadow: 0 0 0 8px rgba(16,185,129,0); } 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); } }

.kpi { position: relative; background: #11161d; border: 1px solid #1f2733; border-radius: 14px; padding: 16px 18px 14px 22px; overflow: hidden; height: 100%; }
.kpi-accent { position: absolute; left: 0; top: 0; bottom: 0; width: 3px; }
.kpi-label { color: #5c6675; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 8px; }
.kpi-value { color: #e6edf3; font-size: 1.85rem; font-weight: 700; line-height: 1.05; font-feature-settings: "tnum"; letter-spacing: -0.02em; }
.kpi-sub { color: #8b96a5; font-size: 0.78rem; margin-top: 6px; font-weight: 500; }

.section-title { font-size: 0.95rem; font-weight: 700; color: #e6edf3; margin: 0 0 4px 0; letter-spacing: -0.01em; }
.section-sub { color: #5c6675; font-size: 0.78rem; margin: 0 0 12px 0; }

.panel { background: #11161d; border: 1px solid #1f2733; border-radius: 14px; padding: 18px 20px; }
.panel-empty { background: #11161d; border: 1px solid #1f2733; border-radius: 14px; height: 280px; display: grid; place-items: center; color: #5c6675; font-size: 0.85rem; }

.pill { display: inline-block; padding: 2px 9px; border-radius: 6px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
.pill-crit { background: rgba(220,38,38,0.18); color: #fca5a5; border: 1px solid rgba(220,38,38,0.35); }
.pill-high { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.30); }
.pill-med  { background: rgba(245,158,11,0.15); color: #fcd34d; border: 1px solid rgba(245,158,11,0.30); }
.pill-low  { background: rgba(59,130,246,0.15); color: #93c5fd; border: 1px solid rgba(59,130,246,0.30); }

.alerts-table { width: 100%; border-collapse: collapse; font-family: 'Inter', sans-serif; }
.alerts-table thead tr { background: #161c25; }
.alerts-table th { text-align: left; padding: 12px 14px; color: #5c6675; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; border-bottom: 1px solid #1f2733; }
.alerts-table th.num { text-align: right; }
.alerts-table tbody tr { border-bottom: 1px solid #171d26; }
.alerts-table tbody tr:last-child { border-bottom: none; }
.alerts-table td { padding: 11px 14px; vertical-align: middle; }
.mono { font-family: 'JetBrains Mono', monospace; }

.footer-bar { margin-top: 28px; padding-top: 14px; border-top: 1px solid #171d26; display: flex; justify-content: space-between; align-items: center; color: #5c6675; font-size: 0.75rem; }
.api-pill { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; padding: 3px 8px; border-radius: 6px; background: #161c25; border: 1px solid #1f2733; color: #8b96a5; }

section[data-testid="stSidebar"] { background: #11161d; border-right: 1px solid #1f2733; }
</style>
""", unsafe_allow_html=True)


# ─── Data fetch ───────────────────────────────────────────────────────────

@st.cache_data(ttl=8)
def fetch_alerts():
    try:
        r = requests.get(f"{API_BASE}/alerts", timeout=4)
        r.raise_for_status()
        return r.json().get("alerts", []), True
    except Exception:
        return [], False

@st.cache_data(ttl=8)
def fetch_stats():
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=4)
        r.raise_for_status()
        return r.json(), True
    except Exception:
        return {}, False

@st.cache_data(ttl=8)
def fetch_history():
    try:
        r = requests.get(f"{API_BASE}/score-history?limit=200", timeout=4)
        r.raise_for_status()
        return r.json().get("history", []), True
    except Exception:
        return [], False


# ─── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='color:#e6edf3;font-weight:700;font-size:0.9rem;letter-spacing:0.04em;text-transform:uppercase;margin-bottom:14px;'>Controls</div>", unsafe_allow_html=True)
    auto_refresh = st.toggle("Auto refresh", value=True, help="Reload every 10 seconds")
    st.divider()
    min_score = st.slider("Min fraud score", 0.0, 1.0, 0.0, 0.01)
    top_n = st.slider("Rows in table", 10, 200, 25, 5)
    show_only_fraud = st.checkbox("Fraud only", value=True)

if auto_refresh:
    st_autorefresh(interval=REFRESH_MS, key="autorefresh")


# ─── Pull data ────────────────────────────────────────────────────────────
alerts, alerts_ok = fetch_alerts()
stats, stats_ok = fetch_stats()
history, history_ok = fetch_history()
api_online = alerts_ok or stats_ok or history_ok
threshold = stats.get("threshold", 0.5) if isinstance(stats.get("threshold"), (int, float)) else 0.5

# alerts → df
if alerts:
    df = pd.DataFrame(alerts)
    for col in ("amount", "fraud_score"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "amount", "fraud_score"]).sort_values("timestamp", ascending=False)
else:
    df = pd.DataFrame(columns=["transaction_id", "user_id", "amount", "location", "fraud_score", "timestamp"])

if show_only_fraud and not df.empty:
    df = df[df["fraud_score"] >= threshold]
if not df.empty:
    df = df[df["fraud_score"] >= min_score]

# history → df
hist_df = pd.DataFrame(history) if history else pd.DataFrame(columns=["timestamp", "score", "is_fraud"])
if not hist_df.empty:
    hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"], errors="coerce")
    hist_df["score"] = pd.to_numeric(hist_df["score"], errors="coerce")
    hist_df = hist_df.dropna(subset=["timestamp", "score"]).sort_values("timestamp")


# ─── Header ───────────────────────────────────────────────────────────────
now_str = datetime.now().strftime("%H:%M:%S")
status_class = "" if api_online else "offline"
status_text = "Live" if api_online else "Offline"

# Single-line — Streamlit's markdown breaks on blank lines inside HTML
st.markdown(
    f'<div class="topbar"><div class="brand"><div class="brand-mark">🛡️</div><div><div class="brand-title">Anomaly Detection</div><div class="brand-sub">Real-Time Transaction Monitoring</div></div></div><div class="status-pill"><span class="status-dot {status_class}"></span>{status_text} · Updated {now_str}</div></div>',
    unsafe_allow_html=True,
)


# ─── KPI strip — use st.columns + single-line HTML per card ──────────────
total_txn = stats.get("total_transactions", 0) or 0
fraud_count = stats.get("fraud_detected", 0) or 0
fraud_rate = stats.get("fraud_rate")
avg_score = stats.get("fraud_score_avg")

fraud_rate_str = f"{fraud_rate*100:.2f}%" if isinstance(fraud_rate, (int, float)) else "—"
avg_score_str = f"{avg_score:.3f}" if isinstance(avg_score, (int, float)) else "—"

kpis = [
    ("Transactions", f"{total_txn:,}", "Processed by model", C["low"]),
    ("Fraud Alerts", f"{fraud_count:,}", f"Rate: {fraud_rate_str}", C["high"]),
    ("Avg Score", avg_score_str, "All transactions", C["med"]),
    ("Threshold", f"{threshold:.3f}", "Decision boundary", C["accent"]),
]

cols = st.columns(4, gap="small")
for col, (label, value, sub, accent) in zip(cols, kpis):
    with col:
        st.markdown(
            f'<div class="kpi"><div class="kpi-accent" style="background:{accent}"></div><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown('<div style="height:22px;"></div>', unsafe_allow_html=True)


# ─── Plotly base ──────────────────────────────────────────────────────────
def base_layout(height=320, **kwargs):
    layout = dict(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11, color=C["text_dim"]),
        xaxis=dict(gridcolor=C["border_lo"], zerolinecolor=C["border_lo"], linecolor=C["border"], tickfont=dict(color=C["text_dim"], size=10)),
        yaxis=dict(gridcolor=C["border_lo"], zerolinecolor=C["border_lo"], linecolor=C["border"], tickfont=dict(color=C["text_dim"], size=10)),
        hoverlabel=dict(bgcolor=C["surface_2"], bordercolor=C["border"], font=dict(family="Inter", color=C["text"], size=11)),
        showlegend=False,
    )
    layout.update(kwargs)
    return layout

def severity_color(score: float) -> str:
    if score >= 0.85: return C["crit"]
    if score >= 0.50: return C["high"]
    if score >= threshold: return C["med"]
    return C["low"]


# ─── Row 1: timeline + distribution ───────────────────────────────────────
col1, col2 = st.columns([1.55, 1])

with col1:
    st.markdown('<div class="section-title">Score timeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Each point is one transaction. Threshold shown as dashed line.</div>', unsafe_allow_html=True)
    if not hist_df.empty:
        colors = [severity_color(s) for s in hist_df["score"]]
        fig = go.Figure()
        fig.add_hline(y=threshold, line_dash="dash", line_color=C["text_mute"], line_width=1, annotation_text=f"threshold {threshold:.3f}", annotation_position="top left", annotation_font=dict(color=C["text_mute"], size=10))
        fig.add_trace(go.Scatter(
            x=hist_df["timestamp"], y=hist_df["score"], mode="markers",
            marker=dict(color=colors, size=8, line=dict(color=C["bg"], width=1), opacity=0.85),
            customdata=hist_df[["user_id", "amount", "location"]].values,
            hovertemplate="<b>Score %{y:.3f}</b><br>User %{customdata[0]} · ₹%{customdata[1]:,.2f}<br>%{customdata[2]} · %{x|%H:%M:%S}<extra></extra>",
        ))
        fig.update_layout(**base_layout(height=320, yaxis=dict(range=[-0.04, 1.04], gridcolor=C["border_lo"], linecolor=C["border"], tickfont=dict(color=C["text_dim"], size=10))))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown('<div class="panel-empty">Awaiting transactions…</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-title">Score distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Where the model places its confidence.</div>', unsafe_allow_html=True)
    if not hist_df.empty and len(hist_df) >= 5:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=hist_df["score"], xbins=dict(start=0, end=1, size=0.05),
            marker=dict(color=hist_df["score"], colorscale=[[0, C["low"]], [0.5, C["med"]], [1, C["high"]]], line=dict(color=C["bg"], width=0.5)),
            opacity=0.95, hovertemplate="Score range %{x}<br>%{y} txns<extra></extra>",
        ))
        fig.add_vline(x=threshold, line_dash="dash", line_color=C["text_mute"], line_width=1, annotation_text=f"{threshold:.3f}", annotation_position="top", annotation_font=dict(color=C["text_mute"], size=10))
        fig.update_layout(**base_layout(
            height=320,
            xaxis=dict(title="Fraud score", range=[0, 1], gridcolor=C["border_lo"], linecolor=C["border"], tickfont=dict(color=C["text_dim"], size=10), title_font=dict(color=C["text_mute"], size=10)),
            yaxis=dict(title="Count", gridcolor=C["border_lo"], linecolor=C["border"], tickfont=dict(color=C["text_dim"], size=10), title_font=dict(color=C["text_mute"], size=10)),
            bargap=0.04,
        ))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown('<div class="panel-empty">Need ≥5 transactions…</div>', unsafe_allow_html=True)


# ─── Row 2: locations + hours ─────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="section-title">Alerts by location</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Color-graded by average severity, not count.</div>', unsafe_allow_html=True)
    if not df.empty:
        loc_df = df.groupby("location", as_index=False).agg(count=("transaction_id", "size"), mean_score=("fraud_score", "mean")).sort_values("count", ascending=True)
        bar_colors = [severity_color(s) for s in loc_df["mean_score"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=loc_df["location"], x=loc_df["count"], orientation="h",
            marker=dict(color=bar_colors, line=dict(color=C["bg"], width=0.5)),
            text=loc_df["count"], textposition="outside",
            textfont=dict(color=C["text"], size=11, family="JetBrains Mono"),
            customdata=loc_df["mean_score"],
            hovertemplate="<b>%{y}</b><br>%{x} alerts · avg score %{customdata:.3f}<extra></extra>",
        ))
        fig.update_layout(**base_layout(
            height=320,
            xaxis=dict(showticklabels=False, gridcolor=C["border_lo"]),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)", tickfont=dict(color=C["text"], size=11)),
        ))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown('<div class="panel-empty">No alerts yet…</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="section-title">Alerts by hour</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Time-of-day pattern across all alerts.</div>', unsafe_allow_html=True)
    if not df.empty:
        hour_df = df.copy()
        hour_df["hour"] = hour_df["timestamp"].dt.hour
        hour_counts = hour_df.groupby("hour").size().reindex(range(24), fill_value=0).reset_index(name="count")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hour_counts["hour"], y=hour_counts["count"],
            marker=dict(color=hour_counts["count"], colorscale=[[0, C["muted_blue"]], [0.5, C["med"]], [1, C["high"]]], line=dict(color=C["bg"], width=0.5)),
            hovertemplate="<b>Hour %{x:02d}:00</b><br>%{y} alerts<extra></extra>",
        ))
        fig.update_layout(**base_layout(
            height=320,
            xaxis=dict(dtick=2, range=[-0.5, 23.5], gridcolor="rgba(0,0,0,0)", linecolor=C["border"], tickfont=dict(color=C["text_dim"], size=10)),
            yaxis=dict(gridcolor=C["border_lo"], linecolor=C["border"], tickfont=dict(color=C["text_dim"], size=10)),
            bargap=0.15,
        ))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown('<div class="panel-empty">No alerts yet…</div>', unsafe_allow_html=True)


# ─── Recent alerts table ──────────────────────────────────────────────────
st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Recent alerts</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-sub">Most recent {top_n} flagged transactions, sorted by time.</div>', unsafe_allow_html=True)


def severity_label(s: float) -> str:
    if s >= 0.85: return "CRITICAL"
    if s >= 0.50: return "HIGH"
    if s >= threshold: return "MEDIUM"
    return "LOW"


def severity_pill(s: float) -> str:
    label = severity_label(s)
    cls = {"CRITICAL": "pill-crit", "HIGH": "pill-high", "MEDIUM": "pill-med", "LOW": "pill-low"}[label]
    return f'<span class="pill {cls}">{label}</span>'


if not df.empty:
    table = df[["timestamp", "transaction_id", "user_id", "amount", "location", "fraud_score"]].head(top_n).copy()

    rows = []
    for _, r in table.iterrows():
        ts = r["timestamp"].strftime("%H:%M:%S") if pd.notna(r["timestamp"]) else "—"
        date = r["timestamp"].strftime("%Y-%m-%d") if pd.notna(r["timestamp"]) else ""
        tx = r["transaction_id"]
        tx_short = (tx[:8] + "…" + tx[-4:]) if isinstance(tx, str) and len(tx) > 14 else tx
        score = r["fraud_score"]
        score_color = severity_color(score)
        # IMPORTANT: build each row as a single line — no newlines in the f-string
        row = (
            f'<tr>'
            f'<td><div style="color:#e6edf3;font-weight:500;font-size:0.85rem;">{ts}</div><div style="color:#5c6675;font-size:0.72rem;">{date}</div></td>'
            f'<td class="mono" style="font-size:0.78rem;color:#8b96a5;">{tx_short}</td>'
            f'<td class="mono" style="text-align:right;color:#e6edf3;font-size:0.85rem;">{int(r["user_id"])}</td>'
            f'<td class="mono" style="text-align:right;color:#e6edf3;font-size:0.85rem;font-weight:500;">₹{r["amount"]:,.2f}</td>'
            f'<td style="color:#e6edf3;font-size:0.85rem;">{r["location"]}</td>'
            f'<td class="mono" style="text-align:right;color:{score_color};font-weight:600;font-size:0.88rem;">{score:.3f}</td>'
            f'<td>{severity_pill(score)}</td>'
            f'</tr>'
        )
        rows.append(row)

    header = (
        '<thead><tr>'
        '<th>Time</th>'
        '<th>Txn ID</th>'
        '<th class="num">User</th>'
        '<th class="num">Amount</th>'
        '<th>Location</th>'
        '<th class="num">Score</th>'
        '<th>Severity</th>'
        '</tr></thead>'
    )
    table_html = (
        '<div class="panel" style="padding:0;overflow-x:auto;">'
        '<table class="alerts-table">'
        f'{header}'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
        '</div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)
else:
    st.markdown('<div class="panel" style="height:120px;display:grid;place-items:center;color:#5c6675;">No alerts to display.</div>', unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="footer-bar"><div>Anomaly Detection · FastAPI + Streamlit + Kafka · Stateful behavioural features</div><div><span class="api-pill">API {API_BASE}</span></div></div>',
    unsafe_allow_html=True,
)
