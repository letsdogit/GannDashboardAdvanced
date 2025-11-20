# gann_dashboard_premium.py
"""
GANN Pro — Premium Edition (Full features, reliable)
Run: streamlit run gann_dashboard_premium.py
"""

# ---------------------------
# Imports & Requirements
# ---------------------------
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
from typing import List, Dict
# sklearn for simple ML predictor
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
except Exception:
    RandomForestClassifier = None
    train_test_split = None

# ---------------------------
# Page config & CSS
# ---------------------------
st.set_page_config(page_title="GANN Pro — Premium", layout="wide", page_icon="📈")
st.markdown("""
<style>
:root{--bg:#07101d;--card:#0e1a2b;--muted:#9bb0cc;--accent:#7dd3fc;--accent2:#a78bfa;}
body{background:linear-gradient(180deg,var(--bg),#030b18); color:#e8f0ff;}
.block-container{padding-top:1rem;}
.stButton>button{background:linear-gradient(90deg,var(--accent),var(--accent2)); border:none; color:#042236; font-weight:700; border-radius:8px;}
.card{background:rgba(255,255,255,0.04); padding:14px; border-radius:10px; box-shadow:0 6px 20px rgba(0,0,0,0.5);}
.small{color:var(--muted); font-size:12px;}
</style>
""", unsafe_allow_html=True)

st.title("📈 GANN Pro — Premium Edition (Reliable, Feature-rich)")
st.caption("GANN Grid, Fan, Swing detection, Pattern scanner, ML predictor, intraday + exports — with heavy reliability")

# ---------------------------
# Utilities & Safe helpers
# ---------------------------
def safe_fmt(v, fmt="{:.2f}", na="N/A"):
    try:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return na
        return fmt.format(v)
    except Exception:
        return na

def safe_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default

def to_base64_bytes(obj_bytes):
    return base64.b64encode(obj_bytes).decode()

def download_link_bytes(bts, filename, mimetype):
    b64 = to_base64_bytes(bts)
    return f"data:{mimetype};base64,{b64}"

# ---------------------------
# Robust Yahoo download with fallback candidates
# ---------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_with_candidates(candidates: List[str], start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Try several candidate tickers until one returns data.
    Returns DataFrame or empty df.
    """
    for t in candidates:
        try:
            df = yf.download(t, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
            if df is not None and not df.empty:
                df = df.reset_index()
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                for c in ['Open','High','Low','Close','Adj Close','Volume']:
                    if c not in df.columns:
                        df[c] = np.nan
                df['Return_Pct'] = df['Close'].pct_change()*100
                df['__TICKER__'] = t
                return df
        except Exception:
            time.sleep(0.5)
            continue
    return pd.DataFrame()

# ---------------------------
# GANN generation logic
# ---------------------------
SPRING_EQ = (3,21)
def generate_angles(years: List[int], angles: List[int]):
    rows=[]
    for y in years:
        base = date(y, SPRING_EQ[0], SPRING_EQ[1])
        for a in angles:
            offset = int(round((a/360.0)*365.25))
            rows.append({'GANN_Date': base + timedelta(days=offset), 'Type': f'{a}°', 'Source':'Angle'})
    return pd.DataFrame(rows)

def generate_equinox_solstice(years: List[int]):
    mapping={'Spring Equinox':(3,21),'Summer Solstice':(6,21),'Fall Equinox':(9,23),'Winter Solstice':(12,21)}
    rows=[]
    for y in years:
        for name,(m,d) in mapping.items():
            rows.append({'GANN_Date': date(y,m,d), 'Type': name, 'Source':'EqSol'})
    return pd.DataFrame(rows)

def generate_pressure_dates(years: List[int], methods: List[str]):
    rows=[]
    for y in years:
        base = date(y, SPRING_EQ[0], SPRING_EQ[1])
        quarters = [base + relativedelta(months=+q) for q in (3,6,9,12)]
        if 'simple' in methods:
            cycles=[7,14,28]
            for cp in [base]+quarters:
                for c in cycles:
                    for n in range(1,13):
                        rows.append({'GANN_Date': cp + timedelta(days=c*n), 'Type': f'P{c}', 'Source':'Simple'})
        if 'advanced' in methods:
            cycles=[45,60,90,120]
            for cp in [base]+quarters:
                for c in cycles:
                    for n in range(1,10):
                        rows.append({'GANN_Date': cp + timedelta(days=c*n), 'Type': f'P{c}', 'Source':'Advanced'})
        if 'astro' in methods:
            cycles=[19,33,51,72]
            for cp in [base]+quarters:
                for c in cycles:
                    for n in range(1,10):
                        rows.append({'GANN_Date': cp + timedelta(days=c*n), 'Type': f'A{c}', 'Source':'Astro'})
    df = pd.DataFrame(rows)
    if not df.empty:
        df['GANN_Date'] = pd.to_datetime(df['GANN_Date']).dt.date
        df = df.drop_duplicates(subset=['GANN_Date','Type'])
    return df

def build_gann_master(years: List[int], angles: List[int], methods: List[str]) -> pd.DataFrame:
    parts=[generate_angles(years, angles), generate_equinox_solstice(years), generate_pressure_dates(years, methods)]
    df = pd.concat(parts, ignore_index=True, sort=False)
    df['GANN_Date'] = pd.to_datetime(df['GANN_Date']).dt.date
    df = df.drop_duplicates(subset=['GANN_Date','Type']).sort_values('GANN_Date').reset_index(drop=True)
    return df

# ---------------------------
# GANN grid visualization (approx Square-of-9 heatmap)
# ---------------------------
def gann_square_grid(center_price, steps=9):
    """
    Build a square grid around center_price with percent steps.
    Returns pandas DataFrame for heatmap.
    """
    pct = np.linspace(-0.10, 0.10, steps)  # -10% to +10%
    grid = np.array([[center_price*(1+p) for p in pct] for _ in range(steps)])
    df = pd.DataFrame(grid)
    return df

# ---------------------------
# Swing high / low (zigzag-like)
# ---------------------------
def detect_swings(df: pd.DataFrame, window=5):
    """
    Detect local highs and lows using a rolling window.
    Returns lists of (date, price) for highs and lows.
    """
    highs=[]
    lows=[]
    if df is None or df.empty:
        return highs, lows
    series_high = df['High']
    series_low = df['Low']
    dates = df['Date'].dt.date
    for i in range(window, len(df)-window):
        hi = series_high[i]
        lo = series_low[i]
        if hi == series_high[i-window:i+window+1].max():
            highs.append((dates.iloc[i], hi))
        if lo == series_low[i-window:i+window+1].min():
            lows.append((dates.iloc[i], lo))
    return highs, lows

# ---------------------------
# Pattern scanner (basic heuristics)
# ---------------------------
def detect_double_top(df, lookback_days=60):
    """
    Very basic double-top detection: find two peaks within lookback window close in price.
    Returns list of tuples with dates and prices.
    """
    out=[]
    if df is None or df.empty:
        return out
    highs = df['High'].rolling(5, center=True).max().dropna()
    # find local peaks indices
    peaks = highs[ (highs.shift(1) < highs) & (highs.shift(-1) < highs) ]
    peak_idx = peaks.index.tolist()
    for i in range(len(peak_idx)):
        for j in range(i+1, len(peak_idx)):
            di = peak_idx[i]; dj = peak_idx[j]
            days = (df.loc[dj,'Date'] - df.loc[di,'Date']).days
            if days <= lookback_days and abs(df.loc[di,'High'] - df.loc[dj,'High'])/max(df.loc[di,'High'], df.loc[dj,'High']) < 0.03:
                out.append((df.loc[di,'Date'].date(), df.loc[dj,'Date'].date(), df.loc[di,'High'], df.loc[dj,'High']))
    return out

def detect_head_shoulders(df):
    """
    Extremely simplified: look for three peaks with middle peak higher and shoulders similar
    """
    out=[]
    if df is None or df.empty:
        return out
    highs = df['High'].rolling(5, center=True).max().dropna()
    peaks = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)]
    idx = peaks.index.tolist()
    for i in range(len(idx)-2):
        a,b,c = idx[i], idx[i+1], idx[i+2]
        ha = df.loc[a,'High']; hb = df.loc[b,'High']; hc = df.loc[c,'High']
        if hb > ha and hb > hc and abs(ha-hc)/max(ha,hc) < 0.05:
            out.append((df.loc[a,'Date'].date(), df.loc[b,'Date'].date(), df.loc[c,'Date'].date()))
    return out

# ---------------------------
# Simple ML turning-point predictor (RandomForest)
# ---------------------------
def prepare_ml_features(df: pd.DataFrame, horizon=3):
    """
    Create simple features: returns, SMA, vol, and label if price goes up after horizon days.
    """
    if df is None or df.empty:
        return None, None
    d = df.copy().reset_index(drop=True)
    d['ret1'] = d['Close'].pct_change()
    d['ret5'] = d['Close'].pct_change(5)
    d['sma5'] = d['Close'].rolling(5, min_periods=1).mean()
    d['sma10'] = d['Close'].rolling(10, min_periods=1).mean()
    d['vol20'] = d['Volume'].rolling(20, min_periods=1).mean()
    d['future_ret'] = d['Close'].pct_change(periods=horizon).shift(-horizon)
    d['label'] = (d['future_ret'] > 0).astype(int)
    feat_cols = ['ret1','ret5','sma5','sma10','vol20']
    X = d[feat_cols].fillna(0)
    y = d['label'].fillna(0).astype(int)
    return X, y

def train_ml_model(X, y):
    if RandomForestClassifier is None:
        return None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        if len(X_train) < 20:
            return None
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        return clf
    except Exception:
        return None

# ---------------------------
# Sidebar inputs (all features on)
# ---------------------------
with st.sidebar:
    st.header("Configuration — Premium")
    years = st.slider("GANN years (start,end)", 2023, 2035, (2023,2025))
    years_list = list(range(years[0], years[1]+1))

    st.markdown("### Markets & tickers")
    # Recommended with extended fallbacks to guarantee data
    recommended_candidates = {
        "Nifty 50": ["^NSEI","^NIFTY50","NSEI.NS","NIFTY50.NS","NIFTYBEES.NS"],
        "Dow Jones": ["DJI","^DJI","DIA"],   # DIA ETF fallback
        "Nasdaq": ["IXIC","^IXIC","QQQ"],    # QQQ ETF fallback
        "S&P 500 (ETF)": ["SPY"]
    }
    selected_markets = st.multiselect("Select markets to include", list(recommended_candidates.keys()),
                                      default=["Nifty 50","Dow Jones","Nasdaq","S&P 500 (ETF)"])

    # assemble candidates for each chosen market
    market_candidates = {}
    for m in selected_markets:
        user_tick = st.text_input(f"Primary ticker override for {m} (optional)", value=recommended_candidates[m][0])
        cand = [user_tick] + [c for c in recommended_candidates[m] if c != user_tick]
        # unique preserve order
        seen=set(); cand_clean=[x for x in cand if not (x in seen or seen.add(x))]
        market_candidates[m] = cand_clean

    st.markdown("### Date range & frequency")
    end_date = st.date_input("End date", value=date.today())
    start_date = st.date_input("Start date", value=end_date - relativedelta(years=3))

    st.markdown("### GANN settings")
    angle_choices = [30,45,60,72,90,120,135,150,180,210,225,240,252,270,288,300,315,330]
    chosen_angles = st.multiselect("Angles", angle_choices, default=angle_choices)
    pressure_methods = st.multiselect("Pressure methods", ['simple','advanced','astro'], default=['simple','advanced','astro'])

    st.markdown("### Detection & ML")
    move_thresholds = st.multiselect("Highlight moves larger than (%)", [1,2,3,5], default=[1,2])
    vol_multiplier = st.slider("Volume spike multiplier", 1.5, 5.0, 2.0, 0.1)
    run_ml = st.checkbox("Train ML turning-point predictor (RandomForest)", value=True)
    intraday_interval = st.selectbox("Intraday interval (if available)", options=["1d","60m","30m","15m","5m"], index=0)

    st.markdown("### Exports")
    enable_excel = st.checkbox("Enable Excel export", value=True)
    enable_pdf = st.checkbox("Enable PDF export", value=True)

    if st.button("Clear cache & refresh"):
        st.cache_data.clear()
        st.success("Cache cleared. Re-run to fetch fresh data.")
        st.rerun()

# ---------------------------
# Fetch market daily/intraday data (robust)
# ---------------------------
st.markdown("---")
st.subheader("Fetching market data (robust fallbacks)")

market_data = {}
fetch_messages = {}
for market, candidates in market_candidates.items():
    st.info(f"Fetching {market} (candidates: {', '.join(candidates)})")
    # attempt daily first (1d)
    df = fetch_with_candidates(candidates, start=start_date.strftime("%Y-%m-%d"), end=(end_date+timedelta(days=1)).strftime("%Y-%m-%d"), interval="1d")
    # if user requested intraday and daily returned but intraday desired, fetch intraday separately later
    if df is None or df.empty:
        # try ETFs fallback if not already included
        st.warning(f"No daily data found for {market} with candidates. Trying ETF fallbacks (SPY/QQQ/DIA)...")
        fallback_pool = ["SPY","QQQ","DIA"]
        df = fetch_with_candidates(fallback_pool, start=start_date.strftime("%Y-%m-%d"), end=(end_date+timedelta(days=1)).strftime("%Y-%m-%d"), interval="1d")
        if df is None or df.empty:
            fetch_messages[market] = f"No data found (tried candidates + ETFs)."
            market_data[market] = pd.DataFrame()
            continue
    market_data[market] = df
    fetch_messages[market] = f"OK (ticker used: {df['__TICKER__'].iloc[0] if '__TICKER__' in df.columns else 'unknown'})"

# display fetch results
if fetch_messages:
    for k,v in fetch_messages.items():
        if "OK" in v:
            st.success(f"{k}: {v}")
        else:
            st.warning(f"{k}: {v}")

ok_markets = [m for m,df in market_data.items() if not df.empty]
if not ok_markets:
    st.error("No market data could be fetched for any selected market. Try different tickers or check network access.")
    st.stop()

# ---------------------------
# Build GANN master
# ---------------------------
gann_master = build_gann_master(years_list, chosen_angles, pressure_methods)
st.info(f"Generated {len(gann_master)} unique GANN dates (deduped).")

# ---------------------------
# Align GANN dates to a primary market (nearest trading day)
# ---------------------------
primary_market = ok_markets[0]
primary_market = st.selectbox("Primary market for alignment & advanced analytics", ok_markets, index=0)

# build signals aligned with primary market
signals = []
if primary_market:
    primary_df = market_data[primary_market]
    trading_dates = primary_df['Date'].dt.date.tolist()
    for _, r in gann_master.iterrows():
        gd = r['GANN_Date']
        # find nearest trading day <= gd within 7 days
        found = None
        for i in range(0,8):
            candidate = gd - timedelta(days=i)
            if candidate in trading_dates:
                found = candidate
                break
        if not found:
            continue
        mr = primary_df[primary_df['Date'].dt.date == found]
        if mr.empty:
            continue
        last = mr.iloc[-1]
        signals.append({
            'GANN_Date': gd,
            'Type': r.get('Type',''),
            'Source': r.get('Source',''),
            'Market_Date': found,
            'Close': last.get('Close', np.nan),
            'Change_Pct': last.get('Return_Pct', np.nan)
        })

signals_df = pd.DataFrame(signals).sort_values('GANN_Date').reset_index(drop=True)
# classify moves safely
def classify_move_safe(x, thresholds):
    try:
        if x is None or pd.isna(x):
            return ""
        xv = float(x)
        for t in sorted(thresholds, reverse=True):
            if abs(xv) >= t:
                return f">{t}%"
        return ""
    except:
        return ""

if not signals_df.empty:
    signals_df['MoveTag'] = signals_df['Change_Pct'].apply(lambda x: classify_move_safe(x, move_thresholds))

# ---------------------------
# ML Predictor training (safe)
# ---------------------------
ml_model = None
ml_status = "Not trained"
if run_ml and primary_market and not market_data[primary_market].empty and RandomForestClassifier is not None:
    try:
        X, y = prepare_ml_features(market_data[primary_market], horizon=3)
        if X is not None and len(X) > 30:
            ml_model = train_ml_model(X, y)
            if ml_model is not None:
                ml_status = "Trained"
            else:
                ml_status = "Not enough data to train or training failed"
        else:
            ml_status = "Insufficient feature rows for training"
    except Exception as e:
        ml_status = f"Training failed: {e}"
else:
    if run_ml and RandomForestClassifier is None:
        ml_status = "scikit-learn not available; ML disabled"

# ---------------------------
# Swing detection & pattern scan for each market (on-demand)
# ---------------------------
def detect_all_patterns(df: pd.DataFrame):
    swings = {}
    patterns = {}
    try:
        highs, lows = detect_swings(df, window=5)
        swings['highs'] = highs
        swings['lows'] = lows
        patterns['double_tops'] = detect_double_top(df, lookback_days=120)
        patterns['head_shoulders'] = detect_head_shoulders(df)
    except Exception:
        swings = {'highs': [], 'lows': []}
        patterns = {'double_tops': [], 'head_shoulders': []}
    return swings, patterns

# ---------------------------
# UI Tabs
# ---------------------------
tab_overview, tab_signals, tab_charts, tab_tools, tab_ml, tab_exports = st.tabs(["Overview","Signals","Charts","GANN Tools","ML Predictor","Exports"])

# ------- Overview -------
with tab_overview:
    st.subheader("Overview & Quick Snapshot")
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown(f"**Primary market:** {primary_market}")
        if primary_market:
            dfp = market_data[primary_market]
            last = dfp.iloc[-1]
            st.metric("Latest Close", safe_fmt(last.get('Close', np.nan)), delta=safe_fmt(last.get('Return_Pct', np.nan), "{:.2f}%"))
            st.markdown(f"Data source ticker: **{last.get('__TICKER__','unknown')}**")
    with col2:
        st.markdown("GANN master stats")
        st.write(f"Total GANN entries: {len(gann_master)}")
        st.write(f"Aligned signals: {len(signals_df)}")
        st.write(f"ML status: {ml_status}")

    st.markdown("### Market snapshot (selected markets)")
    snap_rows = []
    for m,df in market_data.items():
        if df is None or df.empty:
            snap_rows.append({'Market':m, 'Latest Close':"N/A", '1d%':"N/A", 'Status':'No Data'})
        else:
            last = df.iloc[-1]
            snap_rows.append({'Market':m, 'Latest Close': safe_fmt(last.get('Close', np.nan)), '1d%': safe_fmt(last.get('Return_Pct',np.nan), "{:.2f}%"), 'Status':'OK'})
    st.dataframe(pd.DataFrame(snap_rows), use_container_width=True)

# ------- Signals -------
with tab_signals:
    st.subheader("GANN Dates aligned with Market Dates (Signals)")
    if signals_df.empty:
        st.info("No aligned GANN signals for the selected primary market and date range.")
    else:
        min_d = st.date_input("From", value=signals_df['GANN_Date'].min())
        max_d = st.date_input("To", value=signals_df['GANN_Date'].max())
        only_sig = st.checkbox("Show only significant moves", value=False)
        view = signals_df[(signals_df['GANN_Date'] >= min_d) & (signals_df['GANN_Date'] <= max_d)]
        if only_sig:
            view = view[view['MoveTag'] != ""]
        st.dataframe(view, use_container_width=True, height=420)

        # safe win/loss
        cp = pd.to_numeric(view['Change_Pct'], errors='coerce')
        wins = int((cp > 0).sum())
        losses = int((cp < 0).sum())
        st.write("Wins:", wins, "Losses:", losses)
        st.write("Average move:", safe_fmt(cp.mean(), "{:.2f}%"))

# ------- Charts -------
with tab_charts:
    st.subheader("Interactive Candles with GANN overlays")
    chart_market = st.selectbox("Chart market", ok := [m for m in market_data.keys() if not market_data[m].empty], index=0)
    chart_df = market_data[chart_market]
    ch_from = st.date_input("Chart from", value=end_date - relativedelta(years=1), key="cf1")
    ch_to = st.date_input("Chart to", value=end_date, key="ct1")
    plot_df = chart_df[(chart_df['Date'].dt.date >= ch_from) & (chart_df['Date'].dt.date <= ch_to)].copy()
    if plot_df.empty:
        st.warning("No data in selected chart range.")
    else:
        sma20 = st.checkbox("SMA 20", value=True)
        sma50 = st.checkbox("SMA 50", value=True)
        sma200 = st.checkbox("SMA 200", value=False)
        show_gann_markers = st.checkbox("Show GANN markers", value=True)
        show_fan = st.checkbox("Show GANN fan (approx)", value=False)
        # compute SMAs safely
        if sma20: plot_df['SMA20'] = plot_df['Close'].rolling(20, min_periods=1).mean()
        if sma50: plot_df['SMA50'] = plot_df['Close'].rolling(50, min_periods=1).mean()
        if sma200: plot_df['SMA200'] = plot_df['Close'].rolling(200, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=plot_df['Date'], open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="Candles"))
        if sma20: fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['SMA20'], name='SMA20'))
        if sma50: fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['SMA50'], name='SMA50'))
        if sma200: fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['SMA200'], name='SMA200'))
        # GANN markers
        if show_gann_markers:
            tset = set(plot_df['Date'].dt.date.tolist())
            for gd in gann_master['GANN_Date']:
                if gd >= ch_from and gd <= ch_to:
                    nd = None
                    for i in range(0,6):
                        d = gd - timedelta(days=i)
                        if d in tset:
                            nd = d; break
                    if nd:
                        yv = plot_df.loc[plot_df['Date'].dt.date == nd, 'Close']
                        if not yv.empty:
                            fig.add_trace(go.Scatter(x=[pd.to_datetime(nd)], y=[float(yv.iloc[0])], mode='markers+text', marker=dict(size=8,symbol='diamond'), text=[f"GANN {gd}"], textposition='top center', showlegend=False))
        # Gann fan approximate
        if show_fan:
            origin_date = plot_df['Date'].iloc[-1]
            origin_price = plot_df['Close'].iloc[-1]
            slopes = [1,0.5,0.25,2]
            for s in slopes:
                yvals = [origin_price + ((x-origin_date).days)*s*0.1 for x in plot_df['Date']]
                fig.add_trace(go.Scatter(x=plot_df['Date'], y=yvals, mode='lines', line=dict(width=1,dash='dash'), showlegend=False))
        fig.update_layout(template='plotly_dark', height=650)
        st.plotly_chart(fig, use_container_width=True)
        # volume chart
        vol_fig = px.bar(plot_df, x='Date', y='Volume', title='Volume')
        vol_fig.update_layout(template='plotly_dark', height=200)
        st.plotly_chart(vol_fig, use_container_width=True)

# ------- GANN Tools & Patterns -------
with tab_tools:
    st.subheader("GANN Tools, Square-of-9, Swings & Pattern Scanner")
    tool_market = st.selectbox("Select market for tools", ok)
    df_tool = market_data[tool_market]
    if df_tool.empty:
        st.warning("No data for tool market.")
    else:
        last_price = df_tool['Close'].iloc[-1]
        st.markdown(f"**Last Close ({tool_market}):** {safe_fmt(last_price)}")
        st.markdown("### Square-of-9 (approx) grid")
        grid = gann_square_grid(last_price, steps=9)
        st.dataframe(grid.round(2))
        st.markdown("### Swing highs/lows (local detection)")
        highs, lows = detect_swings(df_tool, window=5)
        st.write("Highs (sample):", highs[:8])
        st.write("Lows (sample):", lows[:8])
        st.markdown("### Pattern scanner (basic heuristics)")
        dtops = detect_double_top(df_tool, lookback_days=120)
        hs = detect_head_shoulders(df_tool)
        st.write("Double tops found (sample):", dtops[:5])
        st.write("Head & Shoulders patterns (sample):", hs[:5])
        st.markdown("### Volume spikes (last 10)")
        vdf = detect_vol_spikes(df_tool, vol_multiplier)
        spikes = vdf[vdf['VolSpike']].tail(10)
        st.dataframe(spikes[['Date','Close','Volume']].tail(10))

# ------- ML Predictor -------
with tab_ml:
    st.subheader("Simple ML Turning-point Predictor (RandomForest)")
    st.markdown("This is a simple proof-of-concept: features are returns, SMAs, vol; label is whether price rises after horizon.")
    st.write("ML status:", ml_status)
    if ml_model is not None:
        st.success("Model trained — showing latest prediction on primary market")
        # prepare most recent feature row and predict
        X_all, y_all = prepare_ml_features(market_data[primary_market], horizon=3)
        latest = X_all.iloc[-1:].fillna(0)
        pred = ml_model.predict(latest)[0]
        proba = ml_model.predict_proba(latest)[0][1] if hasattr(ml_model, "predict_proba") else None
        st.write("Prediction (1 = price up after horizon):", int(pred))
        if proba is not None:
            st.write("Probability:", f"{proba:.2f}")
    else:
        st.info("ML model not available (insufficient data or sklearn not installed).")

# ------- Exports -------
with tab_exports:
    st.subheader("Export Data & Reports")
    export_m = st.selectbox("Export market", ok, index=0)
    edf = market_data[export_m]
    if edf.empty:
        st.warning("No data to export for selected market.")
    else:
        if enable_excel:
            if st.button("Generate Excel (market + signals)"):
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                    edf.to_excel(writer, sheet_name='market_data', index=False)
                    if not signals_df.empty:
                        signals_df.to_excel(writer, sheet_name='gann_signals', index=False)
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode()
                st.markdown(f'<a download="gann_export_{export_m}.xlsx" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}">📥 Download Excel</a>', unsafe_allow_html=True)

        if enable_pdf:
            if st.button("Generate PDF Summary (text-only)") :
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=14)
                pdf.cell(0,8, f"GANN Premium Report - {export_m}", ln=True, align='C')
                pdf.ln(4)
                pdf.set_font("Arial", size=10)
                last = edf.iloc[-1]
                pdf.cell(0,6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
                pdf.cell(0,6, f"Last Close: {safe_fmt(last.get('Close',np.nan))}", ln=True)
                pdf.cell(0,6, f"1d %: {safe_fmt(last.get('Return_Pct',np.nan), '{:.2f}%')}", ln=True)
                pdf.ln(6)
                if not signals_df.empty:
                    pdf.cell(0,6, "Recent GANN Signals (sample):", ln=True)
                    sample = signals_df.tail(10)
                    for _, r in sample.iterrows():
                        pdf.cell(0,6, f"{r['GANN_Date']} | {r['Type'][:12]:12} | Close {safe_fmt(r['Close'])} | {safe_fmt(r['Change_Pct'],'{:.2f}%')}", ln=True)
                pdf_bytes = pdf.output(dest='S').encode('latin-1')
                b64 = base64.b64encode(pdf_bytes).decode()
                st.markdown(f'<a download="gann_report_{export_m}.pdf" href="data:application/pdf;base64,{b64}">📄 Download PDF</a>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div class='small'>GANN Pro — Premium Edition • Approximate GANN tools. For production, verify tickers and optionally supply News API keys for event linking.</div>", unsafe_allow_html=True)
