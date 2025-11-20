# gann_dashboard_reliable_final.py
"""
GANN Pro — Final Reliable Dashboard (recommended tickers)
Run: streamlit run gann_dashboard_reliable_final.py
FIXED: Removed matplotlib dependency for background_gradient
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from fpdf import FPDF
import io, base64, time, traceback, math

# ---------------------------
# App config + CSS
# ---------------------------
st.set_page_config(page_title="GANN Pro — Reliable Final", layout="wide", page_icon="📈")
st.markdown("""
<style>
:root{--bg:#061026;--card:#0b1626;--muted:#94aace;--accent:#7dd3fc;--accent2:#a78bfa;}
body{background:linear-gradient(180deg,var(--bg),#020815); color:#eaf3ff;}
.block-container{padding-top:1rem;}
.stButton>button{background:linear-gradient(90deg,var(--accent),var(--accent2)); border:none; color:#012; font-weight:700;}
.card{background:rgba(255,255,255,0.03); padding:12px; border-radius:10px; box-shadow:0 6px 18px rgba(0,0,0,0.6);}
.small{color:var(--muted); font-size:13px;}
.download-btn {
    display: inline-block;
    padding: 10px 20px;
    background: linear-gradient(90deg, #7dd3fc, #a78bfa);
    color: #012;
    text-decoration: none;
    border-radius: 8px;
    font-weight: 700;
    text-align: center;
    margin: 10px 0;
    transition: all 0.3s ease;
}
.download-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(125, 211, 252, 0.4);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h2>📈 GANN Pro — Final Reliable Dashboard</h2>", unsafe_allow_html=True)
st.markdown("<div class='small'>Recommended tickers: Nifty (^NSEI), Dow (^DJI), Nasdaq (^IXIC). Robust fallbacks & safety checks included.</div>", unsafe_allow_html=True)
st.write("")

# ---------------------------
# Helpers
# ---------------------------
def safe_fmt(val, fmt="{
