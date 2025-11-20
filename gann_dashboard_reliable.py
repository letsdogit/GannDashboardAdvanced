# gann_dashboard_reliable_final.py
"""
GANN PRO — FINAL FULL RELIABLE VERSION
With recommended tickers + deep reliability layer + safe numeric handling
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
# Page config + CSS
# ---------------------------
st.set_page_config(page_title="GANN Pro — Final Reliable Dashboard",
                   layout="wide",
                   page_icon="📈")

st.markdown("""
<style>
:root{
    --bg:#07101d;
    --card:#0e1a2b;
    --muted:#9bb0cc;
    --accent:#7dd3fc;
    --accent2:#a78bfa;
}
body { background: linear-gradient(180deg,var(--bg),#030b18); color: #e8f0ff; }
.block-container { padding-top: 1rem; }
.stButton>button {
    background: linear-gradient(90deg,var(--accent),var(--accent2));
    border: none; color: #042236;
    font-weight: 700; border-radius: 8px;
    padding: 8px 16px;
}
.card {
    background: rgba(255,255,255,0.04);
    padding: 14px; border-radius: 12px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.5);
}
.small { color: var(--muted); font-size: 12px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h2>📈 GANN PRO — Final Reliable Dashboard</h2>", unsafe_allow_html=True)
st.markdown("<div class='small'>Recommended tickers: ^NSEI, DJI, IXIC with full fallback safety.</div>", unsafe_allow_html=True)
st.write("")

# ---------------------------
# Helpers
# ---------------------------
def safe_fmt(value, fmt="{:.2f}", na="N/A"):
    """Very safe number formatting."""
    try:
        if value is None or (isinstance(value,float) and (math.isnan(value) or math.isinf(value))):
            return na
        return fmt.format(value)
    except:
        return na

def try_run(fn, fallback=None, *a, **k):
    try:
        return fn(*a, **k)
    except:
        return fallback

# ---------------------------
# Safe Yahoo data fetch (multiple tickers)
# ---------------------------
@st.cache_data(ttl=3600)
def fetch_with_fallback(tickers, start, end):
    """Try multiple tickers until data is found."""
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False)
            if df is not None and not df.empty:
                df = df.reset_index()
                df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

                # ensure all OHLC fields exist
                for col in ["Open","High","Low","Close","Adj Close","Volume"]:
                    if col not in df.columns:
                        df[col] = np.nan

                df["Return_Pct"] = df["Close"].pct_change()*100
                df["__SOURCE__"] = t
                return df
        except:
            pass

    return pd.DataFrame()  # if all failed

# ---------------------------
# GANN Date Generators
# ---------------------------
SPRING_EQ = (3,21)

def gann_angles(years, angles):
    rows = []
    for y in years:
        base = date(y, SPRING_EQ[0], SPRING_EQ[1])
        for a in angles:
            offset = int(round((a/360)*365.25))
            rows.append({
                "GANN_Date": base + timedelta(days=offset),
                "Type": f"{a}° from Equinox",
                "Source": "Angle"
            })
    return pd.DataFrame(rows)

def gann_es(years):
    eq = {
        "Spring Equinox": (3,21),
        "Summer Solstice": (6,21),
        "Fall Equinox":   (9,23),
        "Winter Solstice":(12,21)
    }
    rows=[]
    for y in years:
        for name,(m,d) in eq.items():
            rows.append({
                "GANN_Date": date(y,m,d),
                "Type": name,
                "Source": "EquinoxSolstice"
            })
    return pd.DataFrame(rows)

def pressure_dates(years, methods):
    rows=[]
    for y in years:
        base = date(y, SPRING_EQ[0], SPRING_EQ[1])
        quarters = [base + relativedelta(months=+i) for i in (3,6,9,12)]

        if "simple" in methods:
            for anchor in [base]+quarters:
                for c in [7,14,28]:
                    for n in range(1,13):
                        rows.append({"GANN_Date": anchor + timedelta(days=c*n),
                                     "Type": f"Pressure_{c}d",
                                     "Source": "Simple"})

        if "advanced" in methods:
            for anchor in [base]+quarters:
                for c in [45,60,90,120]:
                    for n in range(1,10):
                        rows.append({"GANN_Date": anchor + timedelta(days=c*n),
                                     "Type": f"Pressure_{c}d",
                                     "Source": "Advanced"})

        if "astro" in methods:
            for anchor in [base]+quarters:
                for c in [19,33,51,72]:
                    for n in range(1,10):
                        rows.append({"GANN_Date": anchor + timedelta(days=c*n),
                                     "Type": f"Astro_{c}d",
                                     "Source": "Astro"})
    df=pd.DataFrame(rows)
    if not df.empty:
        df["GANN_Date"] = pd.to_datetime(df["GANN_Date"]).dt.date
        df=df.drop_duplicates(subset=["GANN_Date","Type"])
    return df

def build_gann(years, angles, methods):
    df = pd.concat([
        gann_angles(years, angles),
        gann_es(years),
        pressure_dates(years, methods)
    ], ignore_index=True)
    df["GANN_Date"]=pd.to_datetime(df["GANN_Date"]).dt.date
    df=df.drop_duplicates(subset=["GANN_Date","Type"])
    df=df.sort_values("GANN_Date").reset_index(drop=True)
    return df

# ---------------------------
# Utility functions (signals, tools)
# ---------------------------
def nearest_trading(gdate, dates, lookback=7):
    for i in range(lookback+1):
        d = gdate - timedelta(days=i)
        if d in dates:
            return d
    return None

def square9(price, steps=12):
    out=[]
    for i in range(1, steps+1):
        out.append(price*(1+0.01*i))
        out.append(price*(1-0.01*i))
    return sorted(out)

def sup_res(df):
    if df.empty:
        return np.nan, np.nan
    S = df["Low"].rolling(20,min_periods=1).min().iloc[-1]
    R = df["High"].rolling(20,min_periods=1).max().iloc[-1]
    return S,R

def vol_spikes(df, mult):
    df=df.copy()
    df["Vol20"]=df["Volume"].rolling(20,min_periods=1).mean()
    df["Spike"]=df["Volume"] > df["Vol20"]*mult
    return df

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Settings")

    yrs = st.slider("Years", 2023, 2035, (2023,2025))
    years_list = list(range(yrs[0], yrs[1]+1))

    st.markdown("### Markets (Recommended)")
    recommended = {
        "Nifty 50": ["^NSEI","^NIFTY50","NSEI.NS","NIFTY50.NS"],
        "Dow Jones": ["DJI","^DJI"],
        "Nasdaq": ["IXIC","^IXIC"]
    }

    selected_markets = st.multiselect("Choose markets", list(recommended.keys()),
                                      default=list(recommended.keys()))

    market_fallbacks = {}
    for m in selected_markets:
        user_primary = st.text_input(f"Primary ticker for {m}", value=recommended[m][0])
        candidates = [user_primary] + [x for x in recommended[m] if x != user_primary]
        # Make unique
        seen=set()
        candidates=[x for x in candidates if not (x in seen or seen.add(x))]
        market_fallbacks[m] = candidates

    st.markdown("### Date Range")
    end_date = st.date_input("End date", value=date.today())
    start_date = st.date_input("Start date", value=end_date - relativedelta(years=3))

    st.markdown("### GANN Settings")
    angle_options = [30,45,60,72,90,120,135,150,180,210,225,240,252,270,288,300,315,330]
    angles = st.multiselect("Angles", angle_options, default=angle_options)
    methods = st.multiselect("Pressure Methods", ["simple","advanced","astro"],
                             default=["simple","advanced","astro"])

    st.markdown("### Move Detection")
    move_thresh = st.multiselect("Significant move thresholds (%)",
                                 [1,2,3,5], default=[1,2])
    vol_mult = st.slider("Volume spike multiplier", 1.5,5.0,2.0,0.1)

    st.markdown("### Export Options")
    enable_excel = st.checkbox("Enable Excel export", True)
    enable_pdf = st.checkbox("Enable PDF export", True)

    if st.button("Clear Cache + Refresh"):
        st.cache_data.clear()
        st.rerun()

# ---------------------------
# Fetch market data
# ---------------------------
st.markdown("---")
st.subheader("Downloading Market Data")

market_data = {}
failures = {}

for m,candidates in market_fallbacks.items():
    st.info(f"Fetching {m}: trying {', '.join(candidates)}")
    df = fetch_with_fallback(
        candidates,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date+timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    if df.empty:
        failures[m]=f"No data for: {candidates}"
        market_data[m]=pd.DataFrame()
    else:
        market_data[m]=df

if failures:
    st.warning("Some markets failed:")
    for k,v in failures.items():
        st.write(f" - {k}: {v}")

ok_markets = [m for m,df in market_data.items() if not df.empty]
st.success("Working markets: " + (", ".join(ok_markets) if ok_markets else "None"))

# ---------------------------
# Build GANN master
# ---------------------------
gann_df = build_gann(years_list, angles, methods)
st.info(f"GANN dates generated: {len(gann_df)} unique entries")

# ---------------------------
# Align GANN signals (Primary Market)
# ---------------------------
primary_market = ok_markets[0] if ok_markets else None
if primary_market:
    primary_market = st.selectbox("Primary Market", ok_markets, index=0)

signals_df = pd.DataFrame()

if primary_market:
    base_df = market_data[primary_market]
    tdates = base_df["Date"].dt.date.tolist()

    rows = []
    for _,row in gann_df.iterrows():
        gd=row["GANN_Date"]
        nd=nearest_trading(gd,tdates,lookback=5)
        if not nd:
            continue
        mr=base_df[base_df["Date"].dt.date==nd]
        if mr.empty:
            continue
        rec=mr.iloc[-1]
        rows.append({
            "GANN_Date": gd,
            "GANN_Type": row["Type"],
            "Source": row["Source"],
            "Market_Date": nd,
            "Close": rec.get("Close",np.nan),
            "Change_Pct": rec.get("Return_Pct",np.nan),
        })

    signals_df=pd.DataFrame(rows)
    signals_df=signals_df.sort_values("GANN_Date").reset_index(drop=True)

# Safe MoveTag
def tag_move(value, thresholds):
    try:
        if value is None or pd.isna(value):
            return ""
        val=float(value)
        for t in sorted(thresholds, reverse=True):
            if abs(val)>=t:
                return f">{t}%"
        return ""
    except:
        return ""

if not signals_df.empty:
    signals_df["MoveTag"]=signals_df["Change_Pct"].apply(lambda x: tag_move(x, move_thresh))

# ---------------------------
# Tabs
# ---------------------------
tab1,tab2,tab3,tab4,tab5 = st.tabs(["Overview","GANN Signals","Charts","GANN Tools","Exports"])

# ---------------------------
# Overview
# ---------------------------
with tab1:
    st.subheader("Overview")

    if primary_market and not market_data[primary_market].empty:
        dfp = market_data[primary_market]
        last=dfp.iloc[-1]

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Last Close", safe_fmt(last.get("Close",np.nan)),
                  delta=safe_fmt(last.get("Return_Pct",np.nan),"{:.2f}%"))
        c2.metric("30d Avg Vol", safe_fmt(dfp["Volume"].tail(30).mean(),"{:.0f}"))
        c3.metric("52w High", safe_fmt(dfp["Close"].rolling(252,min_periods=1).max().iloc[-1]))
        c4.metric("52w Low", safe_fmt(dfp["Close"].rolling(252,min_periods=1).min().iloc[-1]))
    else:
        st.warning("No primary market selected.")

    st.markdown("### Market Snapshots")
    snap=[]
    for m,df in market_data.items():
        if df.empty:
            snap.append({"Market":m,"Close":"N/A","1d%":"N/A","Status":"No Data"})
        else:
            last=df.iloc[-1]
            snap.append({
                "Market":m,
                "Close": safe_fmt(last.get("Close",np.nan)),
                "1d%": safe_fmt(last.get("Return_Pct",np.nan),"{:.2f}%"),
                "Status": "OK"
            })
    st.dataframe(pd.DataFrame(snap), use_container_width=True)

# ---------------------------
# GANN Signals
# ---------------------------
with tab2:
    st.subheader("GANN Signals")
    if signals_df.empty:
        st.info("No signals available.")
    else:
        ffrom = st.date_input("From", value=signals_df["GANN_Date"].min())
        fto   = st.date_input("To",   value=signals_df["GANN_Date"].max())

        filtered = signals_df[(signals_df["GANN_Date"]>=ffrom)&(signals_df["GANN_Date"]<=fto)]

        only_sig = st.checkbox("Show only significant moves", value=False)
        if only_sig:
            filtered = filtered[filtered["MoveTag"]!=""]

        st.dataframe(filtered, use_container_width=True, height=420)

        # Safe numeric conversion for win/loss
        cp = pd.to_numeric(filtered["Change_Pct"], errors="coerce")
        wins = (cp>0).sum()
        losses = (cp<0).sum()

        st.write("Wins:", int(wins), "Losses:", int(losses))
        st.write("Average Move:", safe_fmt(cp.mean(), "{:.2f}%"))

# ---------------------------
# Charts
# ---------------------------
with tab3:
    st.subheader("Charts")
    if not ok_markets:
        st.warning("No market data.")
    else:
        chart_mkt = st.selectbox("Chart Market", ok_markets)
        dfc = market_data[chart_mkt]
        if dfc.empty:
            st.warning("No data for selected market.")
        else:
            ch_from = st.date_input("Chart From", value=end_date - relativedelta(years=1))
            ch_to = st.date_input("Chart To", value=end_date)

            plot = dfc[(dfc["Date"].dt.date>=ch_from)&(dfc["Date"].dt.date<=ch_to)]
            if plot.empty:
                st.warning("Nothing to plot.")
            else:
                show20 = st.checkbox("SMA20", True)
                show50 = st.checkbox("SMA50", True)
                show200= st.checkbox("SMA200", False)
                show_gann = st.checkbox("Show GANN Markers", True)
                show_fan = st.checkbox("Show GANN Fan Approx", False)

                if show20: plot["SMA20"]=plot["Close"].rolling(20,min_periods=1).mean()
                if show50: plot["SMA50"]=plot["Close"].rolling(50,min_periods=1).mean()
                if show200: plot["SMA200"]=plot["Close"].rolling(200,min_periods=1).mean()

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=plot["Date"], open=plot["Open"], high=plot["High"],
                    low=plot["Low"], close=plot["Close"], name="Candles"
                ))
                if show20: fig.add_trace(go.Scatter(x=plot["Date"],y=plot["SMA20"],name="SMA20"))
                if show50: fig.add_trace(go.Scatter(x=plot["Date"],y=plot["SMA50"],name="SMA50"))
                if show200: fig.add_trace(go.Scatter(x=plot["Date"],y=plot["SMA200"],name="SMA200"))

                if show_gann:
                    tset=set(plot["Date"].dt.date.tolist())
                    for gd in gann_df["GANN_Date"]:
                        if gd>=ch_from and gd<=ch_to:
                            nd=nearest_trading(gd,tset)
                            if nd:
                                close_val=plot.loc[plot["Date"].dt.date==nd,"Close"]
                                if not close_val.empty:
                                    fig.add_trace(go.Scatter(
                                        x=[pd.to_datetime(nd)],
                                        y=[float(close_val.iloc[0])],
                                        mode="markers+text",
                                        marker=dict(size=8,symbol="diamond"),
                                        text=[f"GANN {gd}"],
                                        textposition="top center",
                                        showlegend=False
                                    ))

                if show_fan:
                    origin_date=plot["Date"].iloc[-1]
                    origin_price=plot["Close"].iloc[-1]
                    slopes=[1,0.5,0.25,2]
                    for s in slopes:
                        yvals=[]
                        for d in plot["Date"]:
                            days=(d-origin_date).days
                            yvals.append(origin_price + days*s*0.1)
                        fig.add_trace(go.Scatter(x=plot["Date"], y=yvals, mode="lines", showlegend=False,
                                                 line=dict(width=1, dash="dash")))

                fig.update_layout(template="plotly_dark", height=650)
                st.plotly_chart(fig, use_container_width=True)

                vol_fig = px.bar(plot, x="Date", y="Volume", title="Volume")
                vol_fig.update_layout(template="plotly_dark", height=250)
                st.plotly_chart(vol_fig, use_container_width=True)

# ---------------------------
# GANN Tools
# ---------------------------
with tab4:
    st.subheader("GANN Tools")
    if not ok_markets:
        st.warning("No markets.")
    else:
        tool_m = st.selectbox("Market", ok_markets)
        dfm = market_data[tool_m]
        if dfm.empty:
            st.warning("No data.")
        else:
            last_price = dfm["Close"].iloc[-1]
            st.markdown(f"**Last Close:** {safe_fmt(last_price)}")

            st.markdown("**Square-of-9 Levels**")
            levels = square9(last_price, steps=12)
            st.dataframe(pd.DataFrame({"Levels":levels}), height=300)

            S,R = sup_res(dfm)
            st.write("Support (20d):", safe_fmt(S))
            st.write("Resistance (20d):", safe_fmt(R))

            st.markdown("**Volume Spikes**")
            vols = vol_spikes(dfm, vol_mult)
            spikes = vols[vols["Spike"]].tail(10)
            if spikes.empty:
                st.write("No recent spikes.")
            else:
                st.dataframe(spikes[["Date","Close","Volume"]])

# ---------------------------
# Exports
# ---------------------------
with tab5:
    st.subheader("Exports")

    if not ok_markets:
        st.warning("No markets available.")
    else:
        export_mkt = st.selectbox("Export Market", ok_markets)
        df_export = market_data[export_mkt]

        # Excel export
        if enable_excel:
            if st.button("Generate Excel"):
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    df_export.to_excel(writer, sheet_name="market_data", index=False)
                    if not signals_df.empty:
                        signals_df.to_excel(writer, sheet_name="gann_signals", index=False)
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode()
                st.markdown(
                    f'<a download="gann_export_{export_mkt}.xlsx" '
                    f'href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}">📥 Download Excel</a>',
                    unsafe_allow_html=True
                )

        # PDF export
        if enable_pdf:
            if st.button("Generate PDF Summary"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=14)
                pdf.cell(0,10,f"GANN Report - {export_mkt}",ln=True,align='C')
                pdf.ln(3)
                pdf.set_font("Arial", size=10)
                last=df_export.iloc[-1]
                pdf.cell(0,6,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",ln=True)
                pdf.cell(0,6,f"Last Close: {safe_fmt(last.get('Close',np.nan))}",ln=True)
                pdf.cell(0,6,f"1d %: {safe_fmt(last.get('Return_Pct',np.nan),'{:.2f}%')}",ln=True)
                pdf.ln(5)

                pdf.set_font("Arial", size=9)
                if not signals_df.empty:
                    pdf.cell(0,6,"Recent GANN Signals (sample):",ln=True)
                    for _,r in signals_df.tail(10).iterrows():
                        pdf.cell(0,6,
                                 f"{r['GANN_Date']} | {r['GANN_Type'][:14]:14} | Close {safe_fmt(r['Close'])} | "
                                 f"{safe_fmt(r['Change_Pct'],'{:.2f}%')}",
                                 ln=True)
                pdf_bytes = pdf.output(dest="S").encode("latin-1")
                b64 = base64.b64encode(pdf_bytes).decode()
                st.markdown(
                    f'<a download="gann_report_{export_mkt}.pdf" '
                    f'href="data:application/pdf;base64,{b64}">📄 Download PDF</a>',
                    unsafe_allow_html=True
                )

st.markdown("---")
st.markdown("<div class='small'>© 2025 — GANN PRO Final Reliable Version</div>", unsafe_allow_html=True)
