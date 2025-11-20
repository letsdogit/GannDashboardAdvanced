# gann_dashboard_reliable.py
"""
GANN Dashboard — Full PRO Version (Updated)
Includes complete reliability layer + st.rerun support
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
import io
import base64
import time
import traceback
import math

# ---------------------------------------------------------
#                🔥 STREAMLIT CONFIG + PREMIUM CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="GANN Pro — Reliable Dashboard",
    layout="wide",
    page_icon="📈",
)

st.markdown("""
<style>
:root{
    --bg:#060b17;
    --card:#0a1627;
    --muted:#8da2c6;
    --accent:#7dd3fc;
    --accent2:#a78bfa;
}
body{ background: linear-gradient(180deg,var(--bg),#020813); color:#e9eef8; }
.block-container{ padding-top: 1rem; }
h1,h2,h3{ color:white; }

.stButton>button{
    background: linear-gradient(90deg,var(--accent),var(--accent2));
    border:none; border-radius:8px;
    color:#0b2136; font-weight:700;
}

.card{
    background: rgba(255,255,255,0.05);
    padding:14px; border-radius:12px;
    box-shadow:0 6px 20px rgba(0,0,0,0.5);
}

.small { color:var(--muted); font-size:13px; }
</style>
""", unsafe_allow_html=True)


st.markdown("<h2>📈 GANN Pro — Full Advanced Dashboard</h2>", unsafe_allow_html=True)
st.markdown("<div class='small'>Advanced GANN date analytics, candlesticks, signals, exports, and a full reliability layer.</div>", unsafe_allow_html=True)
st.write("")

# ---------------------------------------------------------
#                      🔧 RELIABILITY HELPERS
# ---------------------------------------------------------
def safe_fmt(val, fmt="{:.2f}", na="N/A"):
    """Safe numeric formatting — no crashes."""
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
        traceback.format_exc()
        return fallback

@st.cache_data(ttl=3600, show_spinner=False)
def safe_yahoo_download(ticker, start, end):
    """Robust Yahoo download with fallback strategy."""
    attempts = [
        ticker,
        ticker + ".NS",
        ticker.replace("^",""),
        ticker.replace("^","") + ".NS"
    ]

    for t in attempts:
        try:
            df = yf.download(t, start=start, end=end, progress=False)
            if df is not None and not df.empty:
                df = df.reset_index()
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                for c in ['Open','High','Low','Close','Adj Close','Volume']:
                    if c not in df.columns:
                        df[c] = np.nan
                df['Return_Pct'] = df['Close'].pct_change()*100
                return df
        except:
            pass

    return pd.DataFrame()  # failed


# ---------------------------------------------------------
#                     🜁 GANN DATE GENERATORS
# ---------------------------------------------------------
SPRING_EQ = (3,21)

def generate_angle_dates(years, angles):
    rows=[]
    for y in years:
        base = date(y, SPRING_EQ[0], SPRING_EQ[1])
        for a in angles:
            offset = int((a/360)*365.25)
            rows.append({
                "GANN_Date": (base + timedelta(days=offset)),
                "Type": f"{a}° from Equinox",
                "Source": "Angle"
            })
    return pd.DataFrame(rows)

def generate_equinox_solstice(years):
    map_dates = {
        "Spring Equinox": (3,21),
        "Summer Solstice": (6,21),
        "Fall Equinox": (9,23),
        "Winter Solstice": (12,21),
    }
    rows=[]
    for y in years:
        for k,(m,d) in map_dates.items():
            rows.append({
                "GANN_Date": date(y,m,d),
                "Type": k,
                "Source": "EquinoxSolstice"
            })
    return pd.DataFrame(rows)

def generate_pressure(years, methods):
    rows=[]
    for y in years:
        base = date(y, SPRING_EQ[0], SPRING_EQ[1])
        quarter_points = [base + relativedelta(months=+q) for q in (3,6,9,12)]

        # simple: cycles 7,14,28
        if "simple" in methods:
            for cp in [base]+quarter_points:
                for c in [7,14,28]:
                    for n in range(1,13):
                        rows.append({"GANN_Date": cp + timedelta(days=c*n),
                                     "Type": f"Pressure_{c}d",
                                     "Source": "Simple"})

        # advanced: 45,60,90,120
        if "advanced" in methods:
            for cp in [base]+quarter_points:
                for c in [45,60,90,120]:
                    for n in range(1,8):
                        rows.append({"GANN_Date": cp + timedelta(days=c*n),
                                     "Type": f"Pressure_{c}d",
                                     "Source": "Advanced"})

        # astro cycles: 19,33,51,72
        if "astro" in methods:
            for cp in [base]+quarter_points:
                for c in [19,33,51,72]:
                    for n in range(1,8):
                        rows.append({"GANN_Date": cp + timedelta(days=c*n),
                                     "Type": f"Astro_{c}d",
                                     "Source": "Astro"})
    df=pd.DataFrame(rows)
    if not df.empty:
        df["GANN_Date"]=pd.to_datetime(df["GANN_Date"]).dt.date
        df=df.drop_duplicates(subset=["GANN_Date","Type"])
    return df

def build_gann_master(years, angles, pressure_methods):
    df = pd.concat([
        generate_angle_dates(years, angles),
        generate_equinox_solstice(years),
        generate_pressure(years, pressure_methods)
    ], ignore_index=True)

    df["GANN_Date"]=pd.to_datetime(df["GANN_Date"]).dt.date
    df=df.drop_duplicates(subset=["GANN_Date","Type"])
    df=df.sort_values("GANN_Date").reset_index(drop=True)
    return df


# ---------------------------------------------------------
#                 📊 GANN TOOLS & SIGNAL HELPERS
# ---------------------------------------------------------
def find_prev_trading_day(gdate, trading_dates):
    """Return nearest trading day <= gdate (lookback 7 days)."""
    for i in range(7):
        c = gdate - timedelta(days=i)
        if c in trading_dates:
            return c
    return None

def square_of_9(price, steps=12):
    lev=[]
    for i in range(1, steps+1):
        lev.append(price*(1+0.01*i))
        lev.append(price*(1-0.01*i))
    return sorted(lev)

def support_resistance(df):
    if df.empty: return (np.nan, np.nan)
    S = df["Low"].rolling(20,min_periods=1).min().iloc[-1]
    R = df["High"].rolling(20,min_periods=1).max().iloc[-1]
    return (S,R)

def detect_volume_spikes(df, mult):
    if df.empty: return df
    df=df.copy()
    df["VolMean20"]=df["Volume"].rolling(20,min_periods=1).mean()
    df["Spike"]=df["Volume"]> df["VolMean20"]*mult
    return df


# ---------------------------------------------------------
#                 🧭 SIDEBAR CONFIGURATION
# ---------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    years_range = st.slider("GANN years", 2023, 2035, (2023,2025))
    years_list = list(range(years_range[0], years_range[1]+1))

    st.markdown("### Markets")
    default_tickers = {
        "Nifty 50":"^NSEI",
        "Dow Jones":"^DJI",
        "Nasdaq":"^IXIC"
    }
    selected_markets = st.multiselect("Select markets", default_tickers.keys(),
                                      default=list(default_tickers.keys()))
    market_tickers={}
    for m in selected_markets:
        market_tickers[m] = st.text_input(f"{m} ticker", value=default_tickers[m])

    st.markdown("### Data Range")
    end_date = st.date_input("End date", value=date.today())
    start_date = st.date_input("Start date", value=end_date - relativedelta(years=3))

    st.markdown("### GANN Angles")
    angle_list = [30,45,60,72,90,120,135,150,180,210,225,240,252,270,288,300,315,330]
    angle_sel = st.multiselect("Angles", angle_list, default=angle_list)

    st.markdown("### Pressure cycles")
    pressure_sel = st.multiselect("Methods", ["simple","advanced","astro"],
                                  default=["simple","advanced","astro"])

    st.markdown("### Move Detection")
    move_thresh = st.multiselect("Move threshold (%)", [1,2,3,5], default=[1,2])
    vol_mult = st.slider("Volume spike multiplier", 1.5,5.0,2.0,0.1)

    st.markdown("### Export")
    enable_pdf = st.checkbox("PDF export", value=True)
    enable_excel = st.checkbox("Excel export", value=True)

    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared. Click Manual Refresh.")
        st.rerun()


# ---------------------------------------------------------
#                   📥 DATA FETCHING (Reliable)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("Data Download Status")

market_data={}
failures={}

for name,tkr in market_tickers.items():
    df = safe_yahoo_download(
        tkr,
        start_date.strftime("%Y-%m-%d"),
        (end_date+timedelta(days=1)).strftime("%Y-%m-%d")
    )
    if df.empty:
        failures[name]=f"No data for {tkr}"
        market_data[name]=pd.DataFrame()
    else:
        market_data[name]=df

if failures:
    st.warning("Some tickers failed:")
    for k,v in failures.items():
        st.write(f"❌ {k}: {v}")

ok_list=[k for k,v in market_data.items() if not v.empty]
st.success("Fetched data for: " + (", ".join(ok_list) if ok_list else "None"))

# ---------------------------------------------------------
#                   🜁 GANN MASTER DATES
# ---------------------------------------------------------
gann_master = build_gann_master(years_list, angle_sel, pressure_sel)
st.info(f"Generated {len(gann_master)} GANN date entries.")


# ---------------------------------------------------------
#                    🧩 ALIGN GANN SIGNALS
# ---------------------------------------------------------
primary_market = st.selectbox("Primary Market", ok_list) if ok_list else None

signals_df=pd.DataFrame()
if primary_market:
    dfp = market_data[primary_market]
    tdates = dfp["Date"].dt.date.tolist()

    rows=[]
    for _,row in gann_master.iterrows():
        gd = row["GANN_Date"]
        mdate = find_prev_trading_day(gd, tdates)
        if not mdate:
            continue
        mr = dfp[dfp["Date"].dt.date == mdate]
        if not mr.empty:
            last=mr.iloc[-1]
            rows.append({
                "GANN_Date": gd,
                "GANN_Type": row["Type"],
                "Market_Date": mdate,
                "Close": last["Close"],
                "Change_Pct": last["Return_Pct"],
                "Source": row["Source"]
            })
    signals_df = pd.DataFrame(rows)
    signals_df=signals_df.sort_values("GANN_Date").reset_index(drop=True)
    if not signals_df.empty:
        signals_df["MoveTag"] = signals_df["Change_Pct"].apply(
            lambda x: next((f">{t}%" for t in sorted(move_thresh,reverse=True) if abs(x)>=t), "")
        )


# ---------------------------------------------------------
#                       MAIN TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "GANN Signals", "Charts", "Gann Tools", "Exports"
])

# ---------------------------- OVERVIEW -------------------------------
with tab1:
    st.subheader("Market Overview")
    if primary_market:
        dfp=market_data[primary_market]
        if dfp.empty:
            st.warning("Primary market data not available.")
        else:
            last=dfp.iloc[-1]

            col1,col2,col3,col4=st.columns(4)
            col1.metric("Last Close", safe_fmt(last["Close"],"{:.2f}"),
                        delta=safe_fmt(last["Return_Pct"],"{:.2f}%"))
            col2.metric("Avg Vol (30d)", safe_fmt(dfp["Volume"].tail(30).mean(), "{:.0f}"))
            col3.metric("52w High", safe_fmt(dfp["Close"].rolling(252,min_periods=1).max().iloc[-1]))
            col4.metric("52w Low", safe_fmt(dfp["Close"].rolling(252,min_periods=1).min().iloc[-1]))

    st.markdown("### Comparison Table")
    rows=[]
    for m,df in market_data.items():
        if df.empty:
            rows.append({"Market":m,"Close":"N/A","1d%":"N/A","Status":"No Data"})
        else:
            last=df.iloc[-1]
            rows.append({"Market":m,"Close":safe_fmt(last["Close"]), "1d%":safe_fmt(last["Return_Pct"],"{:.2f}%"),"Status":"OK"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---------------------------- SIGNALS -------------------------------
with tab2:
    st.subheader("GANN Signals")
    if signals_df.empty:
        st.warning("No signals found.")
    else:
        min_d = st.date_input("From", value=signals_df["GANN_Date"].min())
        max_d = st.date_input("To", value=signals_df["GANN_Date"].max())

        df_filtered = signals_df[(signals_df["GANN_Date"]>=min_d) & (signals_df["GANN_Date"]<=max_d)]
        only_sig = st.checkbox("Significant Only", value=False)
        if only_sig:
            df_filtered=df_filtered[df_filtered["MoveTag"]!=""]

        st.dataframe(df_filtered, use_container_width=True, height=420)

        if not df_filtered.empty:
            st.write("Average Move:", safe_fmt(df_filtered["Change_Pct"].mean(), "{:.2f}%"))
            st.write("Count:", len(df_filtered))


# ---------------------------- CHARTS -------------------------------
with tab3:
    st.subheader("Candlestick & GANN markers")

    chart_market = st.selectbox("Chart Market", ok_list)

    dfc = market_data[chart_market]
    if dfc.empty:
        st.warning("No chart data.")
    else:
        c_from = st.date_input("Chart from:", value=end_date - relativedelta(years=1))
        c_to = st.date_input("Chart to:", value=end_date)
        show20 = st.checkbox("SMA20", True)
        show50 = st.checkbox("SMA50", True)
        show200 = st.checkbox("SMA200", False)
        show_gann = st.checkbox("GANN markers", True)
        show_fan = st.checkbox("Gann Fan Approx", False)

        pdf = dfc[(dfc["Date"].dt.date>=c_from)&(dfc["Date"].dt.date<=c_to)]
        if pdf.empty:
            st.warning("Range empty.")
        else:
            if show20: pdf["SMA20"]=pdf["Close"].rolling(20,min_periods=1).mean()
            if show50: pdf["SMA50"]=pdf["Close"].rolling(50,min_periods=1).mean()
            if show200: pdf["SMA200"]=pdf["Close"].rolling(200,min_periods=1).mean()

            fig=go.Figure()
            fig.add_trace(go.Candlestick(
                x=pdf["Date"], open=pdf["Open"],high=pdf["High"],
                low=pdf["Low"],close=pdf["Close"], name="Candle"
            ))
            if show20: fig.add_trace(go.Scatter(x=pdf["Date"], y=pdf["SMA20"], name="SMA20"))
            if show50: fig.add_trace(go.Scatter(x=pdf["Date"], y=pdf["SMA50"], name="SMA50"))
            if show200: fig.add_trace(go.Scatter(x=pdf["Date"], y=pdf["SMA200"], name="SMA200"))

            # GANN markers
            if show_gann:
                tset=set(pdf["Date"].dt.date)
                for gd in gann_master["GANN_Date"]:
                    if gd>=c_from and gd<=c_to:
                        # nearest trading day
                        nd=find_prev_trading_day(gd, list(tset))
                        if nd:
                            yv = pdf.loc[pdf["Date"].dt.date==nd,"Close"]
                            if not yv.empty:
                                fig.add_trace(go.Scatter(
                                    x=[pd.to_datetime(nd)],
                                    y=[float(yv.iloc[0])],
                                    mode="markers+text",
                                    marker=dict(size=10, symbol="diamond"),
                                    text=[f"GANN {gd}"],
                                    textposition="top center",
                                    showlegend=False
                                ))

            # Gann fan approx
            if show_fan:
                origin_date=pdf["Date"].iloc[-1]
                origin_price=pdf["Close"].iloc[-1]
                xvals=pdf["Date"]
                slopes=[1,0.5,0.25,2]
                for s in slopes:
                    y=[origin_price + ( (x-origin_date).days ) * s * 0.1 for x in xvals]
                    fig.add_trace(go.Scatter(x=xvals,y=y,mode="lines",line=dict(width=1, dash="dash"), showlegend=False))

            fig.update_layout(template="plotly_dark", height=650)
            st.plotly_chart(fig, use_container_width=True)

            # volume
            vol_fig = px.bar(pdf, x="Date", y="Volume", title="Volume")
            vol_fig.update_layout(template="plotly_dark", height=220)
            st.plotly_chart(vol_fig, use_container_width=True)


# ---------------------------- GANN TOOLS -------------------------------
with tab4:
    st.subheader("GANN Tools")

    tool_m = st.selectbox("Select market", ok_list)
    dfm = market_data[tool_m]

    if dfm.empty:
        st.warning("No data.")
    else:
        last_price = dfm["Close"].iloc[-1]
        st.markdown(f"**Last Close:** {safe_fmt(last_price,'{:.2f}')}")
        levels = square_of_9(last_price, steps=12)
        st.markdown("**Square of 9 Levels (approx)**")
        st.dataframe(pd.DataFrame({"Levels":levels}))

        S,R = support_resistance(dfm)
        st.write("Support:", safe_fmt(S))
        st.write("Resistance:", safe_fmt(R))

        vdf = detect_volume_spikes(dfm, vol_mult)
        spikes = vdf[vdf["Spike"]].tail(10)
        st.markdown("**Recent Volume Spikes**")
        if spikes.empty:
            st.write("None")
        else:
            st.dataframe(spikes[["Date","Close","Volume"]])


# ---------------------------- EXPORTS -------------------------------
with tab5:
    st.subheader("Exports & Reports")

    export_m = st.selectbox("Export market", ok_list)
    dfe = market_data[export_m]

    if dfe.empty:
        st.warning("No data for export.")
    else:
        # Excel
        if enable_excel:
            buf = io.BytesIO()
            if st.button("Generate Excel"):
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    dfe.to_excel(writer, sheet_name="market_data", index=False)
                    if not signals_df.empty:
                        signals_df.to_excel(writer, sheet_name="signals", index=False)
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode()
                st.markdown(f'<a download="gann_export.xlsx" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}">📥 Download Excel</a>', unsafe_allow_html=True)

        # PDF
        if enable_pdf:
            if st.button("Generate PDF Summary"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=16)
                pdf.cell(0,10,f"GANN Report - {export_m}", ln=True, align='C')
                pdf.ln(5)
                pdf.set_font("Arial", size=10)
                last=dfe.iloc[-1]
                pdf.cell(0,6,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",ln=True)
                pdf.cell(0,6,f"Last Close: {safe_fmt(last['Close'])}",ln=True)
                pdf.cell(0,6,f"1d %: {safe_fmt(last['Return_Pct'],'{:.2f}%')}",ln=True)
                pdf.ln(6)

                if not signals_df.empty:
                    pdf.set_font("Arial", size=9)
                    pdf.cell(0,6,"Recent GANN Signals:",ln=True)
                    sample=signals_df.tail(10)
                    for _,r in sample.iterrows():
                        pdf.cell(0,6,f"{r['GANN_Date']} | {r['GANN_Type']} | Close {safe_fmt(r['Close'])} | {safe_fmt(r['Change_Pct'],'{:.2f}%')}",ln=True)
                pdf_bytes=pdf.output(dest="S").encode("latin-1")
                b64=base64.b64encode(pdf_bytes).decode()
                st.markdown(f'<a download="gann_report.pdf" href="data:application/pdf;base64,{b64}">📄 Download PDF</a>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div class='small'>© 2025 GANN Pro Dashboard • Advanced version with reliability layer.</div>", unsafe_allow_html=True)
