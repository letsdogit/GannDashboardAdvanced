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
def safe_fmt(val, fmt="{:.2f}", na="N/A"):
    try:
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return na
        return fmt.format(val)
    except Exception:
        return na

@st.cache_data(ttl=3600, show_spinner=False)
def yf_download_robust(ticker, start, end, max_retries=3):
    """
    Robust Yahoo Finance download with proper error handling
    """
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            
            if df is None or df.empty:
                st.warning(f"Attempt {attempt + 1}: No data returned for {ticker}")
                time.sleep(1)
                continue
            
            df = df.reset_index()
            
            if 'Date' not in df.columns:
                st.error(f"Date column missing for {ticker}")
                continue
            
            df['Date'] = pd.to_datetime(df['Date'])
            if df['Date'].dt.tz is not None:
                df['Date'] = df['Date'].dt.tz_localize(None)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    for alt in [col.lower(), col.upper(), f'Adj {col}']:
                        if alt in df.columns:
                            df[col] = df[alt]
                            break
                    else:
                        df[col] = np.nan
            
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']
            
            df['Return_Pct'] = df['Close'].pct_change() * 100
            df['__SOURCE_TICKER'] = ticker
            
            st.success(f"✓ Successfully downloaded {len(df)} rows for {ticker}")
            return df
            
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
            time.sleep(1)
            continue
    
    st.error(f"Failed to download {ticker} after {max_retries} attempts")
    return pd.DataFrame()

# ---------------------------
# GANN generation functions
# ---------------------------
SPRING_EQ = (3, 21)

def generate_static_angles(years, angles):
    rows=[]
    for y in years:
        base = date(y, SPRING_EQ[0], SPRING_EQ[1])
        for a in angles:
            offset = int(round((a/360.0)*365.25))
            rows.append({'GANN_Date': (base + timedelta(days=offset)), 'Type': f"{a}° from Equinox", 'Source':'Angle'})
    return pd.DataFrame(rows)

def generate_equinox_solstice(years):
    mapping = {'Spring Equinox':(3,21),'Summer Solstice':(6,21),'Fall Equinox':(9,23),'Winter Solstice':(12,21)}
    rows=[]
    for y in years:
        for name,(m,d) in mapping.items():
            rows.append({'GANN_Date': date(y,m,d), 'Type': name, 'Source':'EquinoxSolstice'})
    return pd.DataFrame(rows)

def generate_pressure(years, methods):
    rows=[]
    for y in years:
        base = date(y, SPRING_EQ[0], SPRING_EQ[1])
        quarters = [base + relativedelta(months=+q) for q in (3,6,9,12)]
        if 'simple' in methods:
            cycles=[7,14,28]
            for cp in [base]+quarters:
                for c in cycles:
                    for n in range(1,13):
                        rows.append({'GANN_Date': cp + timedelta(days=c*n), 'Type':f'Pressure_{c}d','Source':'Simple'})
        if 'advanced' in methods:
            cycles=[45,60,90,120]
            for cp in [base]+quarters:
                for c in cycles:
                    for n in range(1,10):
                        rows.append({'GANN_Date': cp + timedelta(days=c*n), 'Type':f'Pressure_{c}d','Source':'Advanced'})
        if 'astro' in methods:
            cycles=[19,33,51,72]
            for cp in [base]+quarters:
                for c in cycles:
                    for n in range(1,10):
                        rows.append({'GANN_Date': cp + timedelta(days=c*n), 'Type':f'Astro_{c}d','Source':'Astro'})
    df = pd.DataFrame(rows)
    if not df.empty:
        df['GANN_Date'] = pd.to_datetime(df['GANN_Date']).dt.date
        df = df.drop_duplicates(subset=['GANN_Date','Type'])
    return df

def build_gann_master(years, angles, methods):
    pieces = [generate_static_angles(years, angles), generate_equinox_solstice(years), generate_pressure(years, methods)]
    df = pd.concat(pieces, ignore_index=True, sort=False)
    df['GANN_Date'] = pd.to_datetime(df['GANN_Date']).dt.date
    df = df.drop_duplicates(subset=['GANN_Date','Type']).sort_values('GANN_Date').reset_index(drop=True)
    return df

# ---------------------------
# GANN tool helpers
# ---------------------------
def find_nearest_trading_date(gdate, trading_dates, lookback=7):
    for i in range(lookback+1):
        candidate = gdate - timedelta(days=i)
        if candidate in trading_dates:
            return candidate
    return None

def square_of_9(price, steps=12):
    levels=[]
    for i in range(1, steps+1):
        levels.append(price*(1+0.01*i))
        levels.append(price*(1-0.01*i))
    return sorted(levels)

def support_resistance(df):
    if df is None or df.empty:
        return (np.nan, np.nan)
    s = df['Low'].rolling(20, min_periods=1).min().iloc[-1]
    r = df['High'].rolling(20, min_periods=1).max().iloc[-1]
    return (s,r)

def detect_vol_spikes(df, mult=2.0):
    if df is None or df.empty:
        return df
    d = df.copy()
    d['VolMean20'] = d['Volume'].rolling(20, min_periods=1).mean()
    d['VolSpike'] = d['Volume'] > (d['VolMean20'] * mult)
    return d

# ---------------------------
# Safe comparison helpers
# ---------------------------
def safe_count_positive(series):
    try:
        return int(series.dropna().gt(0).sum())
    except Exception:
        return 0

def safe_count_negative(series):
    try:
        return int(series.dropna().lt(0).sum())
    except Exception:
        return 0

def safe_mean(series):
    try:
        clean = series.dropna()
        if len(clean) > 0:
            return clean.mean()
        return np.nan
    except Exception:
        return np.nan

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Settings — FINAL")
    years = st.slider("GANN years (start,end)", 2020, 2030, (2023, 2025))
    years_list = list(range(years[0], years[1]+1))

    st.markdown("### Markets")
    ticker_options = {
        "Nifty 50": "^NSEI",
        "Dow Jones": "^DJI",
        "Nasdaq": "^IXIC",
        "S&P 500": "^GSPC",
        "FTSE 100": "^FTSE",
        "DAX": "^GDAXI",
        "Nikkei 225": "^N225",
        "Custom": "CUSTOM"
    }
    
    selected_markets = st.multiselect(
        "Select Markets", 
        options=list(ticker_options.keys()), 
        default=["Nifty 50", "Dow Jones", "Nasdaq"]
    )
    
    market_tickers = {}
    for market in selected_markets:
        if market == "Custom":
            custom_ticker = st.text_input("Enter custom ticker (e.g., AAPL, TSLA)", value="AAPL")
            market_tickers[custom_ticker] = custom_ticker
        else:
            market_tickers[market] = ticker_options[market]

    st.markdown("### Date range")
    end_date = st.date_input("End date", value=date.today())
    start_date = st.date_input("Start date", value=end_date - relativedelta(years=2))

    st.markdown("### GANN Angles")
    all_angles = [30,45,60,72,90,120,135,150,180,210,225,240,252,270,288,300,315,330]
    angles_sel = st.multiselect("Angles", all_angles, default=[30,45,60,90,120,150,180,210,240,270,300,330])

    st.markdown("### Pressure date methods")
    pressure_methods = st.multiselect("Methods", ['simple','advanced','astro'], default=['simple'])

    st.markdown("### Detection & Export")
    move_thresholds = st.multiselect("Mark moves > (%)", [1,2,3,5], default=[1,2])
    vol_multiplier = st.slider("Volume spike multiplier", 1.5, 5.0, 2.0, 0.1)
    enable_excel = st.checkbox("Enable Excel export", value=True)
    enable_pdf = st.checkbox("Enable PDF export", value=False)

    if st.button("🔄 Clear cache and refresh"):
        st.cache_data.clear()
        st.success("Cache cleared!")
        st.rerun()

# ---------------------------
# Fetch data for each selected market
# ---------------------------
st.markdown("---")
st.subheader("📊 Fetching Market Data")

market_data = {}
fetch_status = {}

with st.spinner("Downloading market data..."):
    for market_name, ticker in market_tickers.items():
        with st.expander(f"Downloading {market_name} ({ticker})", expanded=False):
            df = yf_download_robust(
                ticker, 
                start=start_date.strftime("%Y-%m-%d"), 
                end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d")
            )
            
            if df is not None and not df.empty:
                market_data[market_name] = df
                fetch_status[market_name] = "✓ Success"
            else:
                market_data[market_name] = pd.DataFrame()
                fetch_status[market_name] = "✗ Failed"

status_df = pd.DataFrame([
    {"Market": k, "Status": v, "Rows": len(market_data.get(k, pd.DataFrame()))}
    for k, v in fetch_status.items()
])
st.dataframe(status_df, use_container_width=True)

ok_markets = [k for k, v in market_data.items() if not v.empty]
if not ok_markets:
    st.error("⚠️ No market data available. Please check your internet connection or try different tickers.")
    st.stop()

st.success(f"✓ Successfully loaded {len(ok_markets)} market(s): {', '.join(ok_markets)}")

# ---------------------------
# Build GANN master
# ---------------------------
with st.spinner("Generating GANN dates..."):
    gann_master = build_gann_master(years_list, angles_sel, pressure_methods)
    st.info(f"📅 Generated {len(gann_master)} GANN dates")

# ---------------------------
# Align GANN dates to primary market
# ---------------------------
primary_market = st.selectbox("Select primary market for analysis", ok_markets, index=0)

signals_df = pd.DataFrame()
if primary_market:
    dfp = market_data[primary_market]
    trading_dates = dfp['Date'].dt.date.tolist()
    rows=[]
    
    with st.spinner("Aligning GANN dates with trading data..."):
        for _, row in gann_master.iterrows():
            gd = row['GANN_Date']
            nd = find_nearest_trading_date(gd, trading_dates, lookback=7)
            if nd is None:
                continue
            mr = dfp[dfp['Date'].dt.date == nd]
            if mr.empty:
                continue
            last = mr.iloc[-1]
            rows.append({
                'GANN_Date': gd,
                'GANN_Type': row.get('Type',''),
                'Source': row.get('Source',''),
                'Market_Date': nd,
                'Close': last.get('Close', np.nan),
                'Change_Pct': last.get('Return_Pct', np.nan)
            })
        signals_df = pd.DataFrame(rows).sort_values('GANN_Date').reset_index(drop=True)

def classify_move_safe(x, thresholds):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return ""
        xv = float(x)
        for t in sorted(thresholds, reverse=True):
            if abs(xv) >= t:
                return f">{t}%"
        return ""
    except Exception:
        return ""

if not signals_df.empty:
    signals_df['MoveTag'] = signals_df['Change_Pct'].apply(lambda x: classify_move_safe(x, move_thresholds))

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview","🎯 GANN Signals","📈 Charts","🔧 Tools","📁 Exports"])

# Overview
with tab1:
    st.subheader("Market Overview")
    if primary_market and not market_data[primary_market].empty:
        dfp = market_data[primary_market]
        last = dfp.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest Close", safe_fmt(last.get('Close', np.nan), "{:.2f}"), 
                   delta=safe_fmt(last.get('Return_Pct', np.nan), "{:.2f}%"))
        col2.metric("30d Avg Vol", safe_fmt(dfp['Volume'].tail(30).mean(), "{:,.0f}"))
        col3.metric("52w High", safe_fmt(dfp['Close'].rolling(252, min_periods=1).max().iloc[-1], "{:.2f}"))
        col4.metric("52w Low", safe_fmt(dfp['Close'].rolling(252, min_periods=1).min().iloc[-1], "{:.2f}"))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dfp['Date'], 
            y=dfp['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#7dd3fc', width=2)
        ))
        fig.update_layout(
            title=f"{primary_market} Price History",
            template='plotly_dark',
            height=400,
            xaxis_title="Date",
            yaxis_title="Price"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### All Markets Snapshot")
    snapshot_rows = []
    for m, df in market_data.items():
        if df is None or df.empty:
            snapshot_rows.append({
                'Market': m, 
                'Latest Close': 'N/A', 
                '1d Change %': 'N/A', 
                'Status': 'No Data'
            })
        else:
            last = df.iloc[-1]
            snapshot_rows.append({
                'Market': m, 
                'Latest Close': safe_fmt(last.get('Close', np.nan)), 
                '1d Change %': safe_fmt(last.get('Return_Pct', np.nan), "{:.2f}%"), 
                'Status': '✓ Active'
            })
    st.dataframe(pd.DataFrame(snapshot_rows), use_container_width=True)

# GANN Signals
with tab2:
    st.subheader("GANN Signals Analysis")
    
    if signals_df.empty:
        st.warning("⚠️ No GANN signals aligned. Check date range or primary market.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            f_from = st.date_input("From Date", value=signals_df['GANN_Date'].min(), key="sig_from")
        with col2:
            f_to = st.date_input("To Date", value=signals_df['GANN_Date'].max(), key="sig_to")
        
        only_sig = st.checkbox("Show only significant moves", value=False)
        
        df_view = signals_df[
            (signals_df['GANN_Date'] >= f_from) & 
            (signals_df['GANN_Date'] <= f_to)
        ].copy()
        
        if only_sig:
            df_view = df_view[df_view['MoveTag'] != ""]
        
        # FIXED: Display without background_gradient to avoid matplotlib dependency
        st.dataframe(df_view, use_container_width=True, height=400)
        
        if not df_view.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            avg_change = safe_mean(df_view['Change_Pct'])
            wins = safe_count_positive(df_view['Change_Pct'])
            losses = safe_count_negative(df_view['Change_Pct'])
            win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
            
            col1.metric("Average Move", safe_fmt(avg_change, "{:.2f}%"))
            col2.metric("Wins", wins)
            col3.metric("Losses", losses)
            col4.metric("Win Rate", f"{win_rate:.1f}%")

# Charts
with tab3:
    st.subheader("Advanced Charts")
    
    if not ok_markets:
        st.warning("No market data available for charts.")
    else:
        chart_market = st.selectbox("Select market for charting", ok_markets, index=0, key="chart_market")
        dfc = market_data.get(chart_market, pd.DataFrame())
        
        if dfc.empty:
            st.warning("Selected market has no data.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                ch_from = st.date_input("Chart From", value=end_date - relativedelta(months=6), key="ch_from")
            with col2:
                ch_to = st.date_input("Chart To", value=end_date, key="ch_to")
            
            col1, col2, col3, col4 = st.columns(4)
            show20 = col1.checkbox("SMA 20", value=True)
            show50 = col2.checkbox("SMA 50", value=True)
            show200 = col3.checkbox("SMA 200", value=False)
            show_gann = col4.checkbox("GANN Markers", value=True)
            
            plot_df = dfc[
                (dfc['Date'].dt.date >= ch_from) & 
                (dfc['Date'].dt.date <= ch_to)
            ].copy()
            
            if plot_df.empty:
                st.warning("No data in selected range.")
            else:
                if show20:
                    plot_df['SMA20'] = plot_df['Close'].rolling(20, min_periods=1).mean()
                if show50:
                    plot_df['SMA50'] = plot_df['Close'].rolling(50, min_periods=1).mean()
                if show200:
                    plot_df['SMA200'] = plot_df['Close'].rolling(200, min_periods=1).mean()
                
                fig = go.Figure()
                
                fig.add_trace(go.Candlestick(
                    x=plot_df['Date'],
                    open=plot_df['Open'],
                    high=plot_df['High'],
                    low=plot_df['Low'],
                    close=plot_df['Close'],
                    name='Price'
                ))
                
                if show20:
                    fig.add_trace(go.Scatter(
                        x=plot_df['Date'], 
                        y=plot_df['SMA20'], 
                        name='SMA 20',
                        line=dict(color='orange', width=1.5, dash='dash')
                    ))
                
                if show50:
                    fig.add_trace(go.Scatter(
                        x=plot_df['Date'], 
                        y=plot_df['SMA50'], 
                        name='SMA 50',
                        line=dict(color='blue', width=1.5, dash='dot')
                    ))
                
                if show200:
                    fig.add_trace(go.Scatter(
                        x=plot_df['Date'], 
                        y=plot_df['SMA200'], 
                        name='SMA 200',
                        line=dict(color='red', width=2)
                    ))
                
                if show_gann and not signals_df.empty:
                    gann_in_range = signals_df[
                        (signals_df['GANN_Date'] >= ch_from) & 
                        (signals_df['GANN_Date'] <= ch_to)
                    ]
                    
                    for _, row in gann_in_range.iterrows():
                        market_date = row['Market_Date']
                        close_price = row['Close']
                        
                        if not math.isnan(close_price):
                            fig.add_trace(go.Scatter(
                                x=[pd.to_datetime(market_date)],
                                y=[close_price],
                                mode='markers+text',
                                marker=dict(size=10, symbol='diamond', color='yellow'),
                                text=[f"GANN"],
                                textposition='top center',
                                showlegend=False,
                                hovertext=f"{row['GANN_Type']}<br>{row['GANN_Date']}"
                            ))
                
                fig.update_layout(
                    title=f"{chart_market} - Candlestick Chart with GANN Dates",
                    template='plotly_dark',
                    height=600,
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### Volume Analysis")
                if 'Volume' in plot_df.columns:
                    vol_data = plot_df[['Date', 'Volume']].copy()
                    vol_data = vol_data[vol_data['Volume'].notna()]
                    
                    if not vol_data.empty:
                        vol_fig = go.Figure()
                        vol_fig.add_trace(go.Bar(
                            x=vol_data['Date'],
                            y=vol_data['Volume'],
                            name='Volume',
                            marker_color='rgba(125, 211, 252, 0.6)'
                        ))
                        
                        vol_fig.update_layout(
                            title="Trading Volume",
                            template='plotly_dark',
                            height=250,
                            xaxis_title="Date",
                            yaxis_title="Volume"
                        )
                        
                        st.plotly_chart(vol_fig, use_container_width=True)
                    else:
                        st.info("No volume data available for this period.")
                else:
                    st.info("Volume data not available for this market.")

# Tools
with tab4:
    st.subheader("GANN Analysis Tools")
    
    if not ok_markets:
        st.warning("No data available for tools.")
    else:
        tool_market = st.selectbox("Select market", ok_markets, key="tool_market")
        dfm = market_data.get(tool_market, pd.DataFrame())
        
        if dfm.empty:
            st.warning("No data available.")
        else:
            last_price = dfm['Close'].iloc[-1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Square of 9 Levels")
                st.markdown(f"**Last Close**: {safe_fmt(last_price, '{:.2f}')}")
                
                levels = square_of_9(last_price, steps=12)
                levels_df = pd.DataFrame({
                    'Level': [safe_fmt(l, '{:.2f}') for l in levels],
                    'Distance %': [safe_fmt((l - last_price) / last_price * 100, '{:.2f}%') for l in levels]
                })
                st.dataframe(levels_df, use_container_width=True, height=300)
            
            with col2:
                st.markdown("### Support & Resistance")
                S, R = support_resistance(dfm)
                
                st.metric("Support (20-day)", safe_fmt(S, '{:.2f}'))
                st.metric("Resistance (20-day)", safe_fmt(R, '{:.2f}'))
                st.metric("Current Price", safe_fmt(last_price, '{:.2f}'))
                
                if not math.isnan(S):
                    dist_s = ((last_price - S) / S) * 100
                    st.write(f"Distance from Support: {safe_fmt(dist_s, '{:.2f}%')}")
                
                if not math.isnan(R):
                    dist_r = ((R - last_price) / last_price) * 100
                    st.write(f"Distance to Resistance: {safe_fmt(dist_r, '{:.2f}%')}")
            
            st.markdown("### Volume Spike Analysis")
            vdf = detect_vol_spikes(dfm, vol_multiplier)
            
            if 'VolSpike' in vdf.columns:
                spikes = vdf[vdf['VolSpike'] == True].tail(20)
                
                if not spikes.empty:
                    st.write(f"Found {len(spikes)} volume spikes in recent history")
                    spike_display = spikes[['Date', 'Close', 'Volume', 'Return_Pct']].copy()
                    spike_display['Date'] = spike_display['Date'].dt.strftime('%Y-%m-%d')
                    st.dataframe(spike_display, use_container_width=True)
                else:
                    st.info("No recent volume spikes detected.")
            else:
                st.info("Volume spike analysis not available.")

# Exports
with tab5:
    st.subheader("Export Data & Reports")
    
    if not ok_markets:
        st.warning("No market data available for export.")
    else:
        export_market = st.selectbox("Select market to export", ok_markets, key="export_market")
        edf = market_data.get(export_market, pd.DataFrame())
        
        if edf.empty:
            st.warning("Selected market has no data.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Excel Export")
                if enable_excel:
                    if st.button("📥 Generate Excel Report", key="excel_btn"):
                        try:
                            buf = io.BytesIO()
                            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                                edf.to_excel(writer, sheet_name='Market_Data', index=False)
                                
                                if not signals_df.empty:
                                    signals_df.to_excel(writer, sheet_name='GANN_Signals', index=False)
                                
                                summary_data = pd.DataFrame({
                                    'Metric': ['Market', 'Latest Close', '1d Change %', 'Data Points', 'Date Range'],
                                    'Value': [
                                        export_market,
                                        safe_fmt(edf['Close'].iloc[-1]),
                                        safe_fmt(edf['Return_Pct'].iloc[-1], '{:.2f}%'),
                                        len(edf),
                                        f"{edf['Date'].min().strftime('%Y-%m-%d')} to {edf['Date'].max().strftime('%Y-%m-%d')}"
                                    ]
                                })
                                summary_data.to_excel(writer, sheet_name='Summary', index=False)
                            
                            buf.seek(0)
                            b64 = base64.b64encode(buf.read()).decode()
                            
                            download_link = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="GANN_Export_{export_market}_{datetime.now().strftime("%Y%m%d")}.xlsx" class="download-btn">📥 Download Excel File</a>'
                            st.markdown(download_link, unsafe_allow_html=True)
                            st.success("✓ Excel file generated successfully!")
                            
                        except Exception as e:
                            st.error(f"Error generating Excel: {str(e)}")
                else:
                    st.info("Excel export is disabled. Enable it in the sidebar settings.")
            
            with col2:
                st.markdown("### 📄 PDF Report")
                if enable_pdf:
                    if st.button("📥 Generate PDF Report", key="pdf_btn"):
                        try:
                            pdf = FPDF()
                            pdf.add_page()
                            pdf.set_font("Arial", 'B', size=16)
                            pdf.cell(0, 10, f"GANN Analysis Report", ln=True, align='C')
                            
                            pdf.set_font("Arial", 'B', size=12)
                            pdf.ln(5)
                            pdf.cell(0, 8, f"Market: {export_market}", ln=True)
                            
                            pdf.set_font("Arial", size=10)
                            pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
                            pdf.ln(3)
                            
                            last = edf.iloc[-1]
                            pdf.set_font("Arial", 'B', size=11)
                            pdf.cell(0, 7, "Market Statistics", ln=True)
                            pdf.set_font("Arial", size=10)
                            pdf.cell(0, 6, f"Latest Close: {safe_fmt(last.get('Close', np.nan), '{:.2f}')}", ln=True)
                            pdf.cell(0, 6, f"1-Day Change: {safe_fmt(last.get('Return_Pct', np.nan), '{:.2f}%')}", ln=True)
                            pdf.cell(0, 6, f"Data Points: {len(edf)}", ln=True)
                            pdf.cell(0, 6, f"Date Range: {edf['Date'].min().strftime('%Y-%m-%d')} to {edf['Date'].max().strftime('%Y-%m-%d')}", ln=True)
                            pdf.ln(5)
                            
                            if not signals_df.empty:
                                pdf.set_font("Arial", 'B', size=11)
                                pdf.cell(0, 7, "Recent GANN Signals", ln=True)
                                pdf.set_font("Arial", size=9)
                                
                                sample_signals = signals_df.tail(15)
                                for _, r in sample_signals.iterrows():
                                    gann_type = str(r.get('GANN_Type', ''))[:20]
                                    close_val = safe_fmt(r.get('Close', np.nan), '{:.2f}')
                                    change_val = safe_fmt(r.get('Change_Pct', np.nan), '{:.2f}%')
                                    
                                    line = f"{r['GANN_Date']} | {gann_type:20} | Close: {close_val:10} | Change: {change_val}"
                                    pdf.cell(0, 5, line, ln=True)
                            else:
                                pdf.set_font("Arial", size=10)
                                pdf.cell(0, 6, "No GANN signals available.", ln=True)
                            
                            pdf.ln(5)
                            pdf.set_font("Arial", 'I', size=8)
                            pdf.cell(0, 5, "This report is for informational purposes only and does not constitute financial advice.", ln=True)
                            
                            pdf_bytes = pdf.output(dest='S').encode('latin-1')
                            b64 = base64.b64encode(pdf_bytes).decode()
                            
                            download_link = f'<a href="data:application/pdf;base64,{b64}" download="GANN_Report_{export_market}_{datetime.now().strftime("%Y%m%d")}.pdf" class="download-btn">📄 Download PDF Report</a>'
                            st.markdown(download_link, unsafe_allow_html=True)
                            st.success("✓ PDF report generated successfully!")
                            
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
                else:
                    st.info("PDF export is disabled. Enable it in the sidebar settings.")
            
            st.markdown("---")
            st.markdown("### 📋 Data Preview")
            
            tab_preview1, tab_preview2 = st.tabs(["Market Data", "GANN Signals"])
            
            with tab_preview1:
                st.dataframe(
                    edf.tail(50)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Return_Pct']],
                    use_container_width=True,
                    height=300
                )
            
            with tab_preview2:
                if not signals_df.empty:
                    st.dataframe(
                        signals_df.tail(50),
                        use_container_width=True,
                        height=300
                    )
                else:
                    st.info("No GANN signals available.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: var(--muted); font-size: 12px; padding: 20px 0;'>
    <p><strong>© 2025 GANN Pro Dashboard</strong> • Advanced Technical Analysis Tool</p>
    <p>⚠️ Disclaimer: This tool is for educational and informational purposes only. Not financial advice.</p>
    <p>Built with Streamlit • Data from Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
