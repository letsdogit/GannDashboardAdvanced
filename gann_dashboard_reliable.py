# gann_dashboard_reliable_final.py
"""
GANN Pro — Final Reliable Dashboard (recommended tickers)
Run: streamlit run gann_dashboard_reliable_final.py
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
</style>
""", unsafe_allow_html=True)

st.markdown("<h2>📈 GANN Pro — Final Reliable Dashboard</h2>", unsafe_allow_html=True)
st.markdown("<div class='small'>Recommended tickers: Nifty (^NSEI), Dow (DJI), Nasdaq (IXIC). Robust fallbacks & safety checks included.</div>", unsafe_allow_html=True)
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

def try_run(func, fallback=None, *a, **kw):
    try:
        return func(*a, **kw)
    except Exception:
        traceback.print_exc()
        return fallback

@st.cache_data(ttl=3600, show_spinner=False)
def yf_download_candidates(candidates, start, end, retries=1, pause=0.8):
    """
    Try multiple ticker candidates in order. Return first successful DataFrame or empty df.
    """
    for t in candidates:
        for attempt in range(retries+1):
            try:
                df = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
                if df is not None and not df.empty:
                    df = df.reset_index()
                    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                    for c in ['Open','High','Low','Close','Adj Close','Volume']:
                        if c not in df.columns:
                            df[c] = np.nan
                    df['Return_Pct'] = df['Close'].pct_change()*100
                    df['__SOURCE_TICKER'] = t
                    return df
                # small pause before retrying same ticker
                time.sleep(pause)
            except Exception:
                time.sleep(pause)
                continue
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
# Safe comparison helpers (FIX FOR THE ERROR)
# ---------------------------
def safe_count_positive(series):
    """Safely count positive values, handling NaN"""
    try:
        return int(series.dropna().gt(0).sum())
    except Exception:
        return 0

def safe_count_negative(series):
    """Safely count negative values, handling NaN"""
    try:
        return int(series.dropna().lt(0).sum())
    except Exception:
        return 0

def safe_mean(series):
    """Safely calculate mean, handling NaN"""
    try:
        clean = series.dropna()
        if len(clean) > 0:
            return clean.mean()
        return np.nan
    except Exception:
        return np.nan

# ---------------------------
# Sidebar controls (recommended tickers)
# ---------------------------
with st.sidebar:
    st.header("Settings — FINAL")
    years = st.slider("GANN years (start,end)", 2023, 2035, (2023, 2025))
    years_list = list(range(years[0], years[1]+1))

    st.markdown("### Markets (recommended)")
    # Recommended primary tickers and fallbacks
    recommended_map = {
        "Nifty 50": ["^NSEI", "^NIFTY50", "^NSEI.NS", "NIFTY50.NS"],
        "Dow Jones": ["DJI", "^DJI"],
        "Nasdaq": ["IXIC", "^IXIC"]
    }
    markets = st.multiselect("Markets to include", options=list(recommended_map.keys()), default=list(recommended_map.keys()))
    # build candidates dict to pass to downloader
    market_candidates = {}
    for m in markets:
        # allow user override of primary ticker (optional)
        user_val = st.text_input(f"Primary ticker for {m} (optional)", value=recommended_map[m][0])
        # build candidates list with user choice first, then default fallbacks
        cand = [user_val] + [c for c in recommended_map[m] if c != user_val]
        # unique
        seen=[]
        cand_clean=[x for x in cand if not (x in seen or seen.append(x))]
        market_candidates[m] = cand_clean

    st.markdown("### Date range")
    end_date = st.date_input("End date", value=date.today())
    start_date = st.date_input("Start date", value=end_date - relativedelta(years=3))

    st.markdown("### GANN Angles")
    all_angles = [30,45,60,72,90,120,135,150,180,210,225,240,252,270,288,300,315,330]
    angles_sel = st.multiselect("Angles", all_angles, default=all_angles)

    st.markdown("### Pressure date methods")
    pressure_methods = st.multiselect("Methods", ['simple','advanced','astro'], default=['simple','advanced','astro'])

    st.markdown("### Detection & Export")
    move_thresholds = st.multiselect("Mark moves > (%)", [1,2,3,5], default=[1,2])
    vol_multiplier = st.slider("Volume spike multiplier", 1.5, 5.0, 2.0, 0.1)
    enable_excel = st.checkbox("Enable Excel export", value=True)
    enable_pdf = st.checkbox("Enable PDF export", value=True)

    if st.button("Clear cache and refresh"):
        st.cache_data.clear()
        st.success("Cache cleared.")
        st.rerun()

# ---------------------------
# Fetch data for each selected market (robust)
# ---------------------------
st.markdown("---")
st.subheader("Fetching market data (robust mode)")

market_data = {}
fetch_errors = {}

for m, candidates in market_candidates.items():
    st.info(f"Downloading {m} (candidates: {', '.join(candidates)})")
    df = yf_download_candidates(candidates, start=start_date.strftime("%Y-%m-%d"), end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"))
    if df is None or df.empty:
        fetch_errors[m] = f"No data for candidates: {candidates}"
        market_data[m] = pd.DataFrame()
    else:
        market_data[m] = df

if fetch_errors:
    st.warning("Some markets had no data:")
    for k,v in fetch_errors.items():
        st.write(f" - {k}: {v}")

ok_markets = [k for k,v in market_data.items() if (v is not None and not v.empty)]
st.success(f"Markets with data: {', '.join(ok_markets) if ok_markets else 'None'}")

# ---------------------------
# Build GANN master
# ---------------------------
gann_master = build_gann_master(years_list, angles_sel, pressure_methods)
st.info(f"GANN master generated — {len(gann_master)} entries (deduped).")

# ---------------------------
# Align GANN dates to primary market trading dates
# ---------------------------
primary_market = ok_markets[0] if ok_markets else None
if primary_market:
    primary_sel = st.selectbox("Primary market to align signals", ok_markets, index=0)
    primary_market = primary_sel
else:
    primary_market = None

signals_df = pd.DataFrame()
if primary_market:
    dfp = market_data[primary_market]
    trading_dates = dfp['Date'].dt.date.tolist()
    rows=[]
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

# Safe classify move function
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
else:
    st.warning("No aligned GANN signals found for the chosen primary market and date range.")

# ---------------------------
# Tabs: Overview / Signals / Charts / Tools / Exports
# ---------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview","GANN Signals","Charts","Gann Tools","Exports"])

# Overview
with tab1:
    st.subheader("Overview")
    if primary_market and not market_data[primary_market].empty:
        dfp = market_data[primary_market]
        last = dfp.iloc[-1]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Latest Close", safe_fmt(last.get('Close', np.nan), "{:.2f}"), delta=safe_fmt(last.get('Return_Pct', np.nan), "{:.2f}%"))
        c2.metric("30d Avg Vol", safe_fmt(dfp['Volume'].tail(30).mean(), "{:.0f}"))
        c3.metric("52w High", safe_fmt(dfp['Close'].rolling(252,min_periods=1).max().iloc[-1], "{:.2f}"))
        c4.metric("52w Low", safe_fmt(dfp['Close'].rolling(252,min_periods=1).min().iloc[-1], "{:.2f}"))
    else:
        st.warning("No primary market data for overview.")

    st.markdown("### Market snapshot")
    snapshot_rows=[]
    for m,df in market_data.items():
        if df is None or df.empty:
            snapshot_rows.append({'Market':m, 'Latest Close':'N/A', '1d %':'N/A', 'Status':'No Data'})
        else:
            last = df.iloc[-1]
            snapshot_rows.append({'Market':m, 'Latest Close': safe_fmt(last.get('Close', np.nan)), '1d %': safe_fmt(last.get('Return_Pct', np.nan), "{:.2f}%"), 'Status':'OK'})
    st.dataframe(pd.DataFrame(snapshot_rows), use_container_width=True)

# GANN Signals
with tab2:
    st.subheader("GANN Signals")
    if signals_df.empty:
        st.info("No GANN signals aligned. Check date range or primary market.")
    else:
        f_from = st.date_input("From", value=signals_df['GANN_Date'].min())
        f_to = st.date_input("To", value=signals_df['GANN_Date'].max())
        only_sig = st.checkbox("Show only significant moves", value=False)
        df_view = signals_df[(signals_df['GANN_Date'] >= f_from) & (signals_df['GANN_Date'] <= f_to)]
        if only_sig:
            df_view = df_view[df_view['MoveTag'] != ""]
        st.dataframe(df_view, use_container_width=True, height=420)
        if not df_view.empty:
            # FIXED: Using safe comparison functions
            avg_change = safe_mean(df_view['Change_Pct'])
            wins = safe_count_positive(df_view['Change_Pct'])
            losses = safe_count_negative(df_view['Change_Pct'])
            
            st.write("Average move:", safe_fmt(avg_change, "{:.2f}%"))
            st.write("Wins:", wins, "Losses:", losses)

# Charts
with tab3:
    st.subheader("Candlestick Charts & GANN Markers")
    if not ok_markets:
        st.warning("No market data available for charts.")
    else:
        chart_market = st.selectbox("Chart market", ok_markets, index=0)
        dfc = market_data.get(chart_market, pd.DataFrame())
        if dfc.empty:
            st.warning("Selected market has no data.")
        else:
            ch_from = st.date_input("Chart from", value=end_date - relativedelta(years=1), key="chart_from")
            ch_to = st.date_input("Chart to", value=end_date, key="chart_to")
            show20 = st.checkbox("SMA20", value=True)
            show50 = st.checkbox("SMA50", value=True)
            show200 = st.checkbox("SMA200", value=False)
            show_gann = st.checkbox("Show GANN markers", value=True)
            show_fan = st.checkbox("Show Gann fan (approx)", value=False)

            plot_df = dfc[(dfc['Date'].dt.date >= ch_from) & (dfc['Date'].dt.date <= ch_to)].copy()
            if plot_df.empty:
                st.warning("No data in this chart range.")
            else:
                if show20: plot_df['SMA20'] = plot_df['Close'].rolling(20, min_periods=1).mean()
                if show50: plot_df['SMA50'] = plot_df['Close'].rolling(50, min_periods=1).mean()
                if show200: plot_df['SMA200'] = plot_df['Close'].rolling(200, min_periods=1).mean()

                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=plot_df['Date'], open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='Candles'))
                if show20: fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['SMA20'], name='SMA20', line=dict(width=1.5, dash='dash')))
                if show50: fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['SMA50'], name='SMA50', line=dict(width=1.5, dash='dot')))
                if show200: fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['SMA200'], name='SMA200', line=dict(width=1.5)))

                if show_gann:
                    tset = set(plot_df['Date'].dt.date.tolist())
                    for gd in gann_master['GANN_Date']:
                        if gd >= ch_from and gd <= ch_to:
                            nd = find_nearest_trading_date(gd, tset, lookback=5)
                            if nd:
                                yv = plot_df.loc[plot_df['Date'].dt.date == nd, 'Close']
                                if not yv.empty:
                                    fig.add_trace(go.Scatter(x=[pd.to_datetime(nd)], y=[float(yv.iloc[0])], mode='markers+text', marker=dict(size=9, symbol='diamond'), text=[f"GANN {gd}"], textposition='top center', showlegend=False))

                if show_fan:
                    origin_date = plot_df['Date'].iloc[-1]
                    origin_price = plot_df['Close'].iloc[-1]
                    xvals = plot_df['Date']
                    slopes = [1,0.5,0.25,2]
                    for s in slopes:
                        yvals = [origin_price + (((x - origin_date).days) * s * 0.1) for x in xvals]
                        fig.add_trace(go.Scatter(x=xvals, y=yvals, mode='lines', line=dict(width=1, dash='dash'), showlegend=False))

                fig.update_layout(template='plotly_dark', height=650)
                st.plotly_chart(fig, use_container_width=True)

                vol_fig = px.bar(plot_df, x='Date', y='Volume', title='Volume')
                vol_fig.update_layout(template='plotly_dark', height=220)
                st.plotly_chart(vol_fig, use_container_width=True)

# GANN Tools
with tab4:
    st.subheader("Gann Tools")
    if not ok_markets:
        st.warning("No data for Gann tools.")
    else:
        tool_market = st.selectbox("Market for tools", ok_markets)
        dfm = market_data.get(tool_market, pd.DataFrame())
        if dfm.empty:
            st.warning("No data.")
        else:
            last_price = dfm['Close'].iloc[-1]
            st.markdown(f"**Last Close ({tool_market})**: {safe_fmt(last_price,'{:.2f}')}")
            levels = square_of_9(last_price, steps=12)
            st.markdown("**Square-of-9 levels (approx)**")
            st.dataframe(pd.DataFrame({'Level':levels}))
            S,R = support_resistance(dfm)
            st.write("Support (20d):", safe_fmt(S))
            st.write("Resistance (20d):", safe_fmt(R))
            vdf = detect_vol_spikes(dfm, vol_multiplier)
            spikes = vdf[vdf['VolSpike']].tail(10)
            st.markdown(f"Recent volume spikes (last 10): {len(spikes)}")
            if not spikes.empty:
                st.dataframe(spikes[['Date','Close','Volume']].tail(10))

# Exports
with tab5:
    st.subheader("Exports & Report")
    if not ok_markets:
        st.warning("No market data for exports.")
    else:
        export_market = st.selectbox("Export market", ok_markets)
        edf = market_data.get(export_market, pd.DataFrame())
        if edf.empty:
            st.warning("Selected market has no data.")
        else:
            # Excel
            if enable_excel:
                if st.button("Prepare Excel (data + signals)"):
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                        edf.to_excel(writer, sheet_name='market_data', index=False)
                        if not signals_df.empty:
                            signals_df.to_excel(writer, sheet_name='gann_signals', index=False)
                    buf.seek(0)
                    b64 = base64.b64encode(buf.read()).decode()
                    st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="gann_export_{export_market}.xlsx">📥 Download Excel</a>', unsafe_allow_html=True)

            # PDF
            if enable_pdf:
                if st.button("Generate PDF Summary"):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=14)
                    pdf.cell(0,8,f"GANN Report - {export_market}", ln=True, align='C')
                    pdf.ln(4)
                    pdf.set_font("Arial", size=10)
                    last = edf.iloc[-1]
                    pdf.cell(0,6,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
                    pdf.cell(0,6,f"Latest Close: {safe_fmt(last.get('Close',np.nan),'%0.2f')}", ln=True)
                    pdf.cell(0,6,f"1d %: {safe_fmt(last.get('Return_Pct',np.nan),'{:.2f}%')}", ln=True)
                    pdf.ln(6)
                    if not signals_df.empty:
                        pdf.set_font("Arial", size=9)
                        pdf.cell(0,6,"Recent GANN Signals (sample):", ln=True)
                        sample = signals_df.tail(10)
                        for _, r in sample.iterrows():
                            pdf.cell(0,6, f"{r['GANN_Date']} | {r['GANN_Type'][:12]:12} | Close {safe_fmt(r.get('Close',np.nan))} | {safe_fmt(r.get('Change_Pct',np.nan),'{:.2f}%')}", ln=True)
                    else:
                        pdf.cell(0,6, "No GANN signals to include.", ln=True)
                    pdf_bytes = pdf.output(dest='S').encode('latin-1')
                    b64 = base64.b64encode(pdf_bytes).decode()
                    st.markdown(f'<a href="data:application/pdf;base64,{b64}" download="gann_report_{export_market}.pdf">📄 Download PDF</a>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div class='small'>© GANN Pro — Final Reliable • Use recommended tickers; verify ticker variations for your region if needed.</div>", unsafe_allow_html=True)
