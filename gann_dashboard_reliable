# gann_dashboard_reliable.py
"""
GANN Dashboard — Full PRO (Reliable)
Run: streamlit run gann_dashboard_reliable.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import time
from fpdf import FPDF
import io
import base64
import math
import traceback
from typing import Dict, List, Tuple

# ---------------------------
# ----- APP CONFIG ----------
# ---------------------------
st.set_page_config(page_title="GANN Pro — Reliable Dashboard", layout="wide", page_icon="📈", initial_sidebar_state="expanded")
# CSS: premium dark theme (compact)
st.markdown("""
<style>
:root{--bg:#071025;--card:#071827;--muted:#9fb6d5;--accent:#7dd3fc;--accent2:#a78bfa;}
body { background: linear-gradient(180deg,var(--bg),#020816) !important; color:#e6eef8; }
.block-container { padding-top: 1rem; }
.stButton>button { background: linear-gradient(90deg,var(--accent),var(--accent2)); border:none; color:#012; font-weight:700; }
.card { background: rgba(255,255,255,0.03); padding:12px; border-radius:12px; box-shadow: 0 6px 18px rgba(0,0,0,0.6); }
.small { color:var(--muted); font-size:13px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='margin:2px'>📈 GANN Pro — Reliable Dashboard</h2>", unsafe_allow_html=True)
st.markdown("<div class='small'>Full-featured GANN analysis with robust error handling and exports.</div>", unsafe_allow_html=True)
st.write("")

# ---------------------------
# ----- REQUIREMENTS NOTE ---
# ---------------------------
# (User should install required packages externally)
# pip install streamlit pandas numpy yfinance plotly fpdf openpyxl python-dateutil

# ---------------------------
# ----- UTILITY HELPERS -----
# ---------------------------
def safe_fmt(val, fmt="{:.2f}", na="N/A"):
    try:
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return na
        return fmt.format(val)
    except Exception:
        return na

def try_exec(func, fallback=None, *args, **kwargs):
    """Run func(*args, **kwargs) and return fallback on any exception (plus log)"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.debug if hasattr(st, "debug") else None
        # log trace in session state for debugging if enabled
        _ = traceback.format_exc()
        return fallback

def bytes_to_download_link(bytes_obj, filename, mimetype):
    b64 = base64.b64encode(bytes_obj).decode()
    return f"data:{mimetype};base64,{b64}"

# ---------------------------
# ----- RELIABILITY LAYER ---
# ---------------------------
# caching with manual clear button
@st.cache_data(ttl=3600, show_spinner=False)
def yf_download_safe(ticker: str, start: str, end: str, retries=2, pause=1.0) -> pd.DataFrame:
    """
    Robust wrapper around yfinance.download.
    Retries on failure. Returns empty DataFrame on failure.
    """
    last_exc = None
    for attempt in range(retries+1):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
            if df is None:
                df = pd.DataFrame()
            if not df.empty:
                df = df.reset_index()
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                # ensure columns
                for c in ['Open','High','Low','Close','Adj Close','Volume']:
                    if c not in df.columns:
                        df[c] = np.nan
                df['Return_Pct'] = df['Close'].pct_change()*100
            return df
        except Exception as e:
            last_exc = e
            time.sleep(pause)
    # if we reached here, all attempts failed
    st.warning(f"Failed to download {ticker}. Error: {last_exc}")
    return pd.DataFrame()

# ---------------------------
# ----- GANN GENERATION -----
# ---------------------------
SPRING_EQ_MONTH = 3
SPRING_EQ_DAY = 21

def generate_static_angles(years: List[int], angles: List[int]) -> pd.DataFrame:
    rows = []
    for y in years:
        base = date(y, SPRING_EQ_MONTH, SPRING_EQ_DAY)
        days_in_year = 365.25
        for a in angles:
            offset = int(round((a / 360.0) * days_in_year))
            d = base + timedelta(days=offset)
            rows.append({'GANN_Date': pd.to_datetime(d).date(), 'Type': f'{a}° from Equinox', 'Source': 'StaticAngle'})
    return pd.DataFrame(rows)

def equinox_solstice(years: List[int]) -> pd.DataFrame:
    mapping = {
        'Spring Equinox': (3,21),
        'Summer Solstice': (6,21),
        'Fall Equinox': (9,23),
        'Winter Solstice': (12,21)
    }
    rows = []
    for y in years:
        for name,(m,d) in mapping.items():
            rows.append({'GANN_Date': pd.to_datetime(date(y,m,d)).date(), 'Type': name, 'Source': 'EquinoxSolstice'})
    return pd.DataFrame(rows)

def pressure_dates(years: List[int], methods: List[str]) -> pd.DataFrame:
    rows = []
    for y in years:
        base = date(y, SPRING_EQ_MONTH, SPRING_EQ_DAY)
        quarter_points = [base + relativedelta(months=+q) for q in (3,6,9,12)]
        if 'simple' in methods:
            cycles = [7,14,28]
            for cp in [base] + quarter_points:
                for c in cycles:
                    for n in range(1,13):
                        rows.append({'GANN_Date': (cp + timedelta(days=c*n)), 'Type': f'Pressure_{c}d', 'Source':'Simple'})
        if 'advanced' in methods:
            cycles = [45,60,90,120]
            for cp in [base] + quarter_points:
                for c in cycles:
                    for n in range(1,10):
                        rows.append({'GANN_Date': (cp + timedelta(days=c*n)), 'Type': f'Pressure_{c}d', 'Source':'Advanced'})
        if 'astro' in methods:
            cycles = [19,33,51,72]
            for cp in [base] + quarter_points:
                for c in cycles:
                    for n in range(1,10):
                        rows.append({'GANN_Date': (cp + timedelta(days=c*n)), 'Type': f'Astro_{c}d', 'Source':'Astro'})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['GANN_Date'] = pd.to_datetime(df['GANN_Date']).dt.date
    df = df.drop_duplicates(subset=['GANN_Date','Type'])
    return df

def build_gann_master(years: List[int], angles: List[int], pressure_methods: List[str]) -> pd.DataFrame:
    s = generate_static_angles(years, angles)
    e = equinox_solstice(years)
    p = pressure_dates(years, pressure_methods)
    # concat and dedupe
    df = pd.concat([s,e,p], ignore_index=True, sort=False)
    df['GANN_Date'] = pd.to_datetime(df['GANN_Date']).dt.date
    df = df.drop_duplicates(subset=['GANN_Date','Type'])
    df = df.sort_values('GANN_Date').reset_index(drop=True)
    return df

# ---------------------------
# ----- GANN TOOLS (approx) -
# ---------------------------
def square_of_9(price: float, steps:int=12) -> List[float]:
    """Approx levels ±1..steps%"""
    levels = []
    for i in range(1, steps+1):
        levels.append(price * (1 + 0.01 * i))
        levels.append(price * (1 - 0.01 * i))
    return sorted(levels)

def support_resistance_basic(df: pd.DataFrame) -> Tuple[float,float]:
    if df is None or df.empty:
        return (float('nan'), float('nan'))
    s = df['Low'].rolling(20, min_periods=1).min().iloc[-1]
    r = df['High'].rolling(20, min_periods=1).max().iloc[-1]
    return (s, r)

def detect_moves(df: pd.DataFrame, thresholds: List[int]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df['MoveTag'] = df['Return_Pct'].apply(lambda x: next((f">{t}%" for t in sorted(thresholds, reverse=True) if abs(x) >= t), ''))
    return df

def detect_vol_spikes(df: pd.DataFrame, multiplier:float=2.0) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df['Vol_Mean_20'] = df['Volume'].rolling(20, min_periods=1).mean()
    df['VolSpike'] = df['Volume'] > (df['Vol_Mean_20'] * multiplier)
    return df

# ---------------------------
# ----- UI: Sidebar -------
# ---------------------------
with st.sidebar:
    st.header("Settings & Data")
    years = st.slider("GANN years (start,end)", 2023, 2035, (2023, 2025))
    start_y, end_y = years
    years_list = list(range(start_y, end_y+1))

    st.markdown("### Markets (Yahoo tickers)")
    # default tickers; fallback mapping will be used
    default_map = {
        "Nifty 50": "^NSEI",
        "Dow Jones": "^DJI",
        "Nasdaq": "^IXIC"
    }
    markets = st.multiselect("Select markets", options=list(default_map.keys()), default=list(default_map.keys()))
    custom_tickers = {}
    for m in markets:
        v = st.text_input(f"Ticker for {m}", value=default_map.get(m,''), key=f"ticker_{m}")
        custom_tickers[m] = v.strip()

    st.markdown("### Data Range (daily)")
    end_date = st.date_input("End date", value=date.today())
    start_date = st.date_input("Start date", value=end_date - relativedelta(years=+3))

    st.markdown("### GANN Angles")
    angle_list = [30,45,60,72,90,120,135,150,180,210,225,240,252,270,288,300,315,330]
    chosen_angles = st.multiselect("Choose angles", angle_list, default=angle_list)

    st.markdown("### Pressure date methods")
    pressure_methods = st.multiselect("Methods", ['simple','advanced','astro'], default=['simple','advanced','astro'])

    st.markdown("### Detection & thresholds")
    move_thresholds = st.multiselect("Mark moves larger than (%)", [1,2,3,5], default=[1,2])
    vol_multiplier = st.slider("Volume spike multiplier", 1.5, 5.0, 2.0, 0.1)

    st.markdown("### Export settings")
    enable_pdf = st.checkbox("Enable PDF export", value=True)
    enable_excel = st.checkbox("Enable Excel export", value=True)

    st.markdown("### Other")
    refresh_cache = st.button("Clear cached data (force refresh)")

# Clear cache if requested
if refresh_cache:
    st.cache_data.clear()
    st.success("Cache cleared. Re-run data fetch by interacting with the app.")

# ---------------------------
# ----- DATA FETCHING ------
# ---------------------------
st.markdown("---")
st.subheader("Data Fetch & Status")
col_fetch_left, col_fetch_right = st.columns([3,1])

with col_fetch_left:
    st.write(f"Fetching daily data for selected markets between {start_date} and {end_date}.")
with col_fetch_right:
    if st.button("Manual Refresh (clear cache)"):
        st.cache_data.clear()
        st.experimental_rerun()

# fetch markets robustly
market_data: Dict[str, pd.DataFrame] = {}
fetch_errors: Dict[str, str] = {}

for market_name, ticker in custom_tickers.items():
    if not ticker:
        fetch_errors[market_name] = "No ticker provided"
        continue
    # try primary ticker, then a few fallbacks (common variations)
    tried = []
    success_df = pd.DataFrame()
    candidates = [ticker, ticker + ".NS", ticker.replace("^",""), ticker.replace("^","") + ".NS"]
    candidates = list(dict.fromkeys(candidates))  # unique
    for t in candidates:
        tried.append(t)
        df = try_exec(yf_download_safe, fallback=pd.DataFrame(), ticker=t, start=start_date.strftime("%Y-%m-%d"), end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            success_df = df
            break
    if success_df.empty:
        fetch_errors[market_name] = f"All candidates failed for {market_name}. Tried: {tried}"
        market_data[market_name] = pd.DataFrame()  # empty
    else:
        market_data[market_name] = success_df

# show fetch summary
if fetch_errors:
    st.warning("Some markets failed to fetch. See details below.")
    for k,v in fetch_errors.items():
        st.text(f"{k}: {v}")

fetched_ok = [k for k,v in market_data.items() if (v is not None and not v.empty)]
st.success(f"Fetched data for: {', '.join(fetched_ok) or 'None'}")

# ---------------------------
# ----- BUILD GANN MASTER ---
# ---------------------------
gann_master = build_gann_master(years_list, chosen_angles, pressure_methods)
st.info(f"Generated {len(gann_master)} GANN date entries (deduplicated).")

# ---------------------------
# ----- SIGNALS & ALIGN ----
# ---------------------------
def find_nearest_trading_date(gdate: date, trading_dates: List[date], lookback_days=7) -> date:
    """Find latest trading date <= gdate within lookback_days, else return None"""
    for i in range(0, lookback_days+1):
        candidate = gdate - timedelta(days=i)
        if candidate in trading_dates:
            return candidate
    return None

primary_market = list(market_data.keys())[0] if market_data else None
primary_market = st.selectbox("Primary market for signal alignment", options=list(market_data.keys()), index=0 if market_data else -1)

signals_rows = []
if primary_market and not market_data[primary_market].empty:
    df_primary = market_data[primary_market].copy()
    trading_dates = list(df_primary['Date'].dt.date)
    for _, row in gann_master.iterrows():
        gd = row['GANN_Date']
        market_date = find_nearest_trading_date(gd, trading_dates, lookback_days=7)
        if market_date is None:
            continue
        market_row = df_primary[df_primary['Date'].dt.date == market_date]
        if market_row.empty:
            continue
        mr = market_row.iloc[-1]
        signals_rows.append({
            'GANN_Date': gd,
            'GANN_Type': row.get('Type',''),
            'Source': row.get('Source',''),
            'Market_Date': market_date,
            'Close': mr['Close'],
            'Change_Pct': mr['Return_Pct']
        })
signals_df = pd.DataFrame(signals_rows).sort_values('GANN_Date').reset_index(drop=True)
if not signals_df.empty:
    signals_df['MoveTag'] = signals_df['Change_Pct'].apply(lambda x: next((f">{t}%" for t in sorted(move_thresholds, reverse=True) if abs(x) >= t), ''))
else:
    st.warning("No aligned GANN signals found for the selected primary market and date range.")

# ---------------------------
# ----- MAIN LAYOUT TABS ---
# ---------------------------
tab_overview, tab_signals, tab_charts, tab_tools, tab_exports = st.tabs(["Overview","GANN Signals","Charts","Gann Tools","Export & Reports"])

# ---------- Overview ----------
with tab_overview:
    st.subheader("Overview Metrics")
    # safe retrieval
    dfp = market_data.get(primary_market, pd.DataFrame())
    if dfp is None or dfp.empty:
        st.warning("Primary market has no data to show metrics.")
    else:
        recent = dfp.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        close_val = recent.get('Close', np.nan)
        return_val = recent.get('Return_Pct', np.nan)
        col1.metric("Latest Close", safe_fmt(close_val, "{:.2f}"), delta=safe_fmt(return_val, "{:.2f}%"))
        vol30 = dfp['Volume'].tail(30).mean() if len(dfp) >= 1 else np.nan
        col2.metric("30d Avg Vol", safe_fmt(vol30, "{:.0f}"))
        hi52 = dfp['Close'].rolling(252, min_periods=1).max().iloc[-1]
        lo52 = dfp['Close'].rolling(252, min_periods=1).min().iloc[-1]
        col3.metric("52w High", safe_fmt(hi52, "{:.2f}"))
        col4.metric("52w Low", safe_fmt(lo52, "{:.2f}"))

    st.markdown("### Market comparison snapshot")
    rows = []
    for name, dfm in market_data.items():
        if dfm is None or dfm.empty:
            rows.append({'Market': name, 'Latest Close': None, '1d %': None, 'Status': 'No Data'})
            continue
        last = dfm.iloc[-1]
        rows.append({'Market': name, 'Latest Close': last['Close'], '1d %': last['Return_Pct'], 'Status':'OK'})
    comp = pd.DataFrame(rows)
    st.dataframe(comp.fillna("N/A"), use_container_width=True)

# ---------- Signals ----------
with tab_signals:
    st.subheader("GANN Dates aligned with Market Dates (Signals)")
    st.markdown("Showing nearest trading day <= GANN date (lookback 7 days). Move tags mark significant changes.")
    if signals_df.empty:
        st.info("No signals to display. Check primary market or date range.")
    else:
        # show filters
        min_date = st.date_input("Show GANN from", value=signals_df['GANN_Date'].min())
        max_date = st.date_input("to", value=signals_df['GANN_Date'].max())
        show_significant_only = st.checkbox("Show only significant moves", value=False)
        df_filtered = signals_df[(signals_df['GANN_Date'] >= min_date) & (signals_df['GANN_Date'] <= max_date)]
        if show_significant_only:
            df_filtered = df_filtered[df_filtered['MoveTag'] != ""]
        st.dataframe(df_filtered, use_container_width=True, height=420)

        st.markdown("#### Quick analytics")
        if not df_filtered.empty:
            avg_move = df_filtered['Change_Pct'].mean()
            wins = df_filtered[df_filtered['Change_Pct'] > 0].shape[0]
            losses = df_filtered[df_filtered['Change_Pct'] < 0].shape[0]
            st.write(f"Average move: {safe_fmt(avg_move, '{:.2f}%')}")
            st.write(f"Wins: {wins}  |  Losses: {losses}")

# ---------- Charts ----------
with tab_charts:
    st.subheader("Candlestick Charts, SMAs, GANN markers")
    # choose market for chart
    chart_market = st.selectbox("Chart market", options=list(market_data.keys()), index=0)
    chart_df = market_data.get(chart_market, pd.DataFrame())
    if chart_df is None or chart_df.empty:
        st.warning("No data available for chart market.")
    else:
        # date range input
        cm_from = st.date_input("Chart from", value=end_date - relativedelta(years=1), key="chart_from")
        cm_to = st.date_input("Chart to", value=end_date, key="chart_to")
        show_sma20 = st.checkbox("SMA 20", value=True, key="sma20")
        show_sma50 = st.checkbox("SMA 50", value=True, key="sma50")
        show_sma200 = st.checkbox("SMA 200", value=False, key="sma200")
        show_fan = st.checkbox("Show Gann Fan (approx)", value=False, key="gannfan")
        show_gann_markers = st.checkbox("Show GANN date markers", value=True, key="gannmarkers")

        plot_df = chart_df[(chart_df['Date'].dt.date >= cm_from) & (chart_df['Date'].dt.date <= cm_to)].copy()
        if plot_df.empty:
            st.warning("No data in selected chart range.")
        else:
            # compute SMAs
            if show_sma20: plot_df['SMA20'] = plot_df['Close'].rolling(20, min_periods=1).mean()
            if show_sma50: plot_df['SMA50'] = plot_df['Close'].rolling(50, min_periods=1).mean()
            if show_sma200: plot_df['SMA200'] = plot_df['Close'].rolling(200, min_periods=1).mean()

            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=plot_df['Date'], open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="Candles"))
            if show_sma20: fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['SMA20'], name='SMA20', line=dict(width=1.5, dash='dash')))
            if show_sma50: fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['SMA50'], name='SMA50', line=dict(width=1.5, dash='dot')))
            if show_sma200: fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['SMA200'], name='SMA200', line=dict(width=1.5)))

            # add GANN markers
            if show_gann_markers:
                gann_in_range = [d for d in gann_master['GANN_Date'] if (d >= cm_from and d <= cm_to)]
                # annotate nearest trading day markers
                trading_set = set(plot_df['Date'].dt.date.to_list())
                for gd in gann_in_range:
                    # find nearest trading day <= gd up to 5 days
                    nd = find_nearest_trading_date(gd, list(trading_set), lookback_days=5)
                    if nd:
                        yval = plot_df.loc[plot_df['Date'].dt.date == nd, 'Close']
                        if not yval.empty:
                            fig.add_trace(go.Scatter(x=[pd.to_datetime(nd)], y=[float(yval.iloc[0])], mode='markers+text',
                                                     marker=dict(size=9, symbol='diamond'), text=[f"GANN {gd}"], textposition='top center', showlegend=False))
            # optional fan approx (simple radial lines)
            if show_fan:
                # pick last close as origin
                origin_date = plot_df['Date'].iloc[-1]
                origin_price = plot_df['Close'].iloc[-1]
                # draw several slopes
                slopes = [1, 0.5, 0.25, 2]  # naive slopes
                xvals = plot_df['Date']
                for s in slopes:
                    yvals = [origin_price + ( ( (x - origin_date).days ) * s * 0.1 ) for x in xvals]
                    fig.add_trace(go.Scatter(x=xvals, y=yvals, mode='lines', line=dict(width=1, dash='dash'), showlegend=False))
            fig.update_layout(template='plotly_dark', height=650, margin=dict(l=30,r=10,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

            # volume chart
            vol = px.bar(plot_df, x='Date', y='Volume', title="Volume", labels={'Volume':'Volume'})
            vol.update_layout(template='plotly_dark', height=200)
            st.plotly_chart(vol, use_container_width=True)

# ---------- GANN Tools ----------
with tab_tools:
    st.subheader("Gann Tools & Diagnostics")
    # base price
    base_market = st.selectbox("Select market for tools", options=list(market_data.keys()), index=0)
    base_df = market_data.get(base_market, pd.DataFrame())
    if base_df is None or base_df.empty:
        st.warning("No data for Gann tools.")
    else:
        last_price = base_df['Close'].iloc[-1]
        st.markdown(f"**Last Close ({base_market})**: {safe_fmt(last_price, '{:.2f}')}")
        levels = square_of_9(last_price, steps=12)
        st.markdown("**Square of 9 levels (approx)**")
        st.write(pd.DataFrame({'level': levels}))
        s, r = support_resistance_basic(base_df)
        st.markdown(f"**20-day Support:** {safe_fmt(s, '{:.2f}')}  |  **20-day Resistance:** {safe_fmt(r, '{:.2f}')}")

        # volume spikes
        vdf = detect_vol_spikes(base_df, multiplier=vol_multiplier)
        spikes = vdf[vdf['VolSpike']].tail(10)
        st.markdown(f"**Recent volume spikes (last 10):** {len(spikes)}")
        if not spikes.empty:
            st.dataframe(spikes[['Date','Close','Volume']].tail(10), use_container_width=True)

# ---------- Exports & Reports ----------
with tab_exports:
    st.subheader("Export Data & Generate Report")
    # Excel export of primary market and signals
    export_market = st.selectbox("Export market", options=list(market_data.keys()), index=0)
    em_df = market_data.get(export_market, pd.DataFrame())
    export_buf = io.BytesIO()
    if enable_excel:
        if st.button("Prepare Excel (data + signals)"):
            if em_df is None or em_df.empty:
                st.warning("No data to export.")
            else:
                with pd.ExcelWriter(export_buf, engine='openpyxl') as writer:
                    em_df.to_excel(writer, sheet_name='market_data', index=False)
                    if not signals_df.empty:
                        signals_df.to_excel(writer, sheet_name='gann_signals', index=False)
                export_buf.seek(0)
                b64 = base64.b64encode(export_buf.read()).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="gann_export_{export_market}.xlsx">📥 Download Excel</a>'
                st.markdown(href, unsafe_allow_html=True)

    # PDF report
    if enable_pdf:
        if st.button("Generate PDF Summary"):
            if em_df is None or em_df.empty:
                st.warning("No data for PDF.")
            else:
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=12)
                pdf.add_page()
                pdf.set_font("Arial", size=16)
                pdf.cell(0, 8, f"GANN Dashboard Report - {export_market}", ln=True, align='C')
                pdf.set_font("Arial", size=10)
                pdf.ln(4)
                # summary
                last = em_df.iloc[-1]
                pdf.cell(0,6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
                pdf.cell(0,6, f"Latest Close: {safe_fmt(last['Close'],'{:.2f}')}, 1d%: {safe_fmt(last['Return_Pct'], '{:.2f}%')}", ln=True)
                pdf.ln(6)
                if not signals_df.empty:
                    pdf.cell(0,6, "Recent GANN signals (sample):", ln=True)
                    pdf.ln(2)
                    sample = signals_df.tail(12)
                    # header
                    colw = [30,30,30,30,30]
                    for h in ['GANN_Date','GANN_Type','Market_Date','Close','Change%']:
                        pdf.cell(36,6,h,border=1)
                    pdf.ln()
                    for _, r in sample.iterrows():
                        pdf.cell(36,6,str(r['GANN_Date']),border=1)
                        pdf.cell(36,6,str(r['GANN_Type'])[:10],border=1)
                        pdf.cell(36,6,str(r['Market_Date']),border=1)
                        pdf.cell(36,6,safe_fmt(r['Close'],'{:.0f}'),border=1)
                        pdf.cell(36,6,safe_fmt(r['Change_Pct'],'{:.2f}%'),border=1)
                        pdf.ln()
                else:
                    pdf.cell(0,6,"No GANN signals to include.", ln=True)
                pdf_bytes = pdf.output(dest='S').encode('latin-1')
                b64 = base64.b64encode(pdf_bytes).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="gann_report_{export_market}.pdf">📄 Download PDF</a>'
                st.markdown(href, unsafe_allow_html=True)

st.markdown("---")
st.markdown("© GANN Pro — Reliable Dashboard • Prototype. For production, secure API keys, respect API rate limits, and vet tickers.")
