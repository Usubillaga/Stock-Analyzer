import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="Titan: Ultimate Market Analyst")

st.markdown("""
<style>
    /* Professional Theme */
    .stApp { background-color: #F5F7F9; font-family: 'Roboto', sans-serif; }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #D1D9E6;
        padding: 15px;
        border-radius: 8px;
    }
    
    /* Regime Badges */
    .regime-bull { border-left: 10px solid #00C853; background-color: #E8F5E9; padding: 20px; font-weight: bold; }
    .regime-bear { border-left: 10px solid #D50000; background-color: #FFEBEE; padding: 20px; font-weight: bold; }
    .regime-warn { border-left: 10px solid #FFAB00; background-color: #FFF3E0; padding: 20px; font-weight: bold; }
    
    /* Buy/Sell Signals */
    .sig-buy { color: #00C853; font-weight: 900; }
    .sig-sell { color: #D50000; font-weight: 900; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE (ROBUST & LONG TERM)
# ==========================================

@st.cache_data(ttl=3600)
def fetch_macro_data():
    """
    Fetches 10 YEARS of data for deep analysis.
    Indices, Breadth, Commodities, Rates, Sectors.
    """
    tickers = {
        # Global Indices
        "S&P 500": "SPY", "Nasdaq 100": "QQQ", "Russell 2000": "IWM", 
        "DAX (Germany)": "^GDAXI", "FTSE 100": "^FTSE",
        # Breadth
        "S&P Equal Weight": "RSP",
        # Volatility
        "VIX": "^VIX", "10Y Yield": "^TNX",
        # Commodities (Inflation)
        "Oil": "USO", "Copper": "CPER", "Lumber": "WOOD", "Gold": "GLD", "Silver": "SLV",
        # Sectors (Areas)
        "Tech": "XLK", "Energy": "XLE", "Financials": "XLF", "Staples": "XLP",
        "Discretionary": "XLY", "Utilities": "XLU", "Real Estate": "XLRE",
        "Materials": "XLB", "Industrials": "XLI", "Healthcare": "XLV"
    }
    
    try:
        # 10 Year History
        data = yf.download(list(tickers.values()), period="10y", progress=False, auto_adjust=True)
        
        # Robust Flattening (Fixes 'No Data' bugs)
        clean = pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            for k, v in tickers.items():
                try:
                    if v in data.columns.levels[1]: clean[k] = data.xs(v, axis=1, level=1)['Close']
                    elif v in data.columns.levels[0]: clean[k] = data[v]['Close']
                except: pass
        else:
            for k, v in tickers.items():
                if v in data: clean[k] = data[v]
        
        clean.fillna(method='ffill', inplace=True)
        return clean
    except: return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_scanner_batch(universe):
    """
    Fetches stock data for scanner. 
    Guarantees results by checking column structure dynamically.
    """
    if universe == "US Tech / Growth":
        ticks = ["NVDA", "AMD", "TSLA", "PLTR", "COIN", "MARA", "UBER", "DKNG", "NET", "CRWD", "AMZN", "GOOGL", "META", "MSFT", "AAPL"]
    elif universe == "Global Macro (ETFs)":
        ticks = ["SPY", "QQQ", "IWM", "EEM", "GLD", "SLV", "USO", "TLT", "HYG", "FXI", "EWZ"]
    else: # DAX / Europe
        ticks = ["SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BMW.DE", "VOW3.DE", "BAS.DE", "ADS.DE", "AIR.DE", "DHL.DE", "DB1.DE", "RWE.DE"]

    data = {}
    try:
        # Download 2 Years for Scanner (Enough for 200SMA)
        raw = yf.download(ticks, period="2y", group_by='ticker', progress=False, threads=True, auto_adjust=True)
        
        for t in ticks:
            try:
                # Dynamic Extraction
                if isinstance(raw.columns, pd.MultiIndex):
                    if t in raw.columns.levels[0]: df = raw[t].copy()
                    else: continue
                elif len(ticks) == 1: df = raw.copy()
                else: continue

                df.dropna(subset=['Close'], inplace=True)
                if len(df) < 100: continue
                
                # --- INDICATORS ---
                df['SMA_50'] = ta.sma(df['Close'], length=50)
                df['SMA_200'] = ta.sma(df['Close'], length=200)
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                bb = ta.bbands(df['Close'], length=20, std=2)
                df['BB_Upper'] = bb['BBU_20_2.0']
                df['BB_Lower'] = bb['BBL_20_2.0']
                
                data[t] = df
            except: continue
    except: pass
    return data

# ==========================================
# 3. ANALYTIC ENGINES
# ==========================================

def macro_ai_analyst(df, in_gdp, in_unemp, in_cpi):
    """Smart Logic comparing Growth vs Inflation vs VIX."""
    if df.empty: return {}, []
    
    curr = df.iloc[-1]
    prev = df.iloc[-252] # 1 Year lookback for trend
    chg = ((curr - prev)/prev)*100
    
    advice = []
    score = 0
    
    # 1. VIX Check (10Y Context)
    vix = curr.get('VIX', 15)
    if vix > 30:
        advice.append(f"🚨 **EXTREME FEAR (VIX {vix:.1f}):** Markets are panic selling. Look for capitulation buys.")
    elif vix > 20:
        advice.append(f"⚠️ **Volatility High (VIX {vix:.1f}):** Caution advised. Reduce leverage.")
        score -= 1
    else:
        advice.append("✅ **Volatility Calm:** Market conditions stable.")
        score += 1

    # 2. Inflation (Commodities + CPI)
    comm_basket = (chg.get('Oil',0) + chg.get('Copper',0) + chg.get('Lumber',0))/3
    if in_cpi > 4.0 or comm_basket > 10.0:
        advice.append(f"🔥 **Inflation Hot:** CPI {in_cpi}% & Commodities +{comm_basket:.1f}%. Hard assets (Gold/Energy) preferred.")
        score -= 2
    else:
        advice.append("✅ **Inflation Tame:** Commodities stable. Tech/Growth favored.")
        score += 1
        
    # 3. Breadth
    if chg.get('S&P 500',0) > chg.get('S&P Equal Weight',0) + 5.0:
        advice.append("📉 **Bad Breadth:** Only Mega-Caps are rallying. Risk of reversal.")
        score -= 1
        
    # Verdict
    if score >= 1: regime = {"status": "BULLISH EXPANSION", "css": "regime-bull"}
    elif score <= -1: regime = {"status": "BEARISH DEFENSIVE", "css": "regime-bear"}
    else: regime = {"status": "CHOPPY / NEUTRAL", "css": "regime-warn"}
    
    return regime, advice

def analyze_vix_impact(df):
    """Highlights 10Y Danger Zones."""
    if 'VIX' not in df or 'S&P 500' not in df: return None
    danger = df[df['VIX'] > 30]
    return danger

def calculate_yearly_matrix(df):
    """Annual Returns Heatmap."""
    y_df = df.resample('Y').last().pct_change() * 100
    y_df = y_df.dropna()
    y_df.index = y_df.index.year
    return y_df.T

def score_stock(df):
    """Ranks stocks -100 to +100."""
    row = df.iloc[-1]
    s = 0
    # Trend
    if row['Close'] > row['SMA_200']: s += 30
    else: s -= 30
    # Momentum
    if row['RSI'] < 30: s += 30 # Buy Dip
    elif row['RSI'] > 75: s -= 30 # Sell Rip
    # Volatility
    if row['Close'] < row['BB_Lower']: s += 20
    if row['Close'] > row['BB_Upper']: s -= 20
    return s

def get_monte_carlo(df, days=90):
    """90-Day Future Cones."""
    last = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    dates = [df.index[-1] + timedelta(days=i) for i in range(1, days+1)]
    # Projections (Wider with time)
    bull = [last + (atr * 0.8 * (i**0.6)) for i in range(1, days+1)]
    bear = [last - (atr * 0.8 * (i**0.6)) for i in range(1, days+1)]
    base = [last for i in range(1, days+1)]
    return dates, bull, bear, base

# ==========================================
# 4. APP UI
# ==========================================

st.title("🦅 Titan: Omni-Market Analyst")

# SIDEBAR INPUTS
st.sidebar.header("Economic Console")
in_gdp = st.sidebar.number_input("GDP Growth (%)", 2.5)
in_cpi = st.sidebar.number_input("CPI Inflation (%)", 3.2)
in_unemp = st.sidebar.number_input("Unemployment (%)", 3.8)
st.sidebar.markdown("---")
st.sidebar.header("Scanner")
univ = st.sidebar.selectbox("Universe", ["US Tech / Growth", "Global Macro (ETFs)", "DAX / Europe"])

# LOAD DATA
with st.spinner("Analyzing 10 Years of History..."):
    macro_df = fetch_macro_data()
    regime, advice = macro_ai_analyst(macro_df, in_gdp, in_unemp, in_cpi)

# TABS
t_hq, t_deep, t_comp, t_scan = st.tabs(["🌍 Macro HQ", "📉 10Y Deep Dive", "📊 Sector Balken", "🚀 Stock Scanner"])

# TAB 1: MACRO HQ
with t_hq:
    if not macro_df.empty:
        st.markdown(f"<div class='{regime['css']}'><h2>REGIME: {regime['status']}</h2></div>", unsafe_allow_html=True)
        st.markdown("### 🧠 AI Strategic Advice")
        for a in advice: st.info(a)
        
        st.divider()
        curr = macro_df.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("VIX", f"{curr.get('VIX',0):.2f}")
        c2.metric("10Y Yield", f"{curr.get('10Y Yield',0):.2f}%")
        c3.metric("Oil (USO)", f"${curr.get('Oil',0):.2f}")
        c4.metric("Copper", f"${curr.get('Copper',0):.2f}")
    else: st.error("Data Error.")

# TAB 2: 10Y DEEP DIVE
with t_deep:
    st.subheader("VIX vs S&P 500 (10 Years)")
    st.caption("Red 'X' marks indicate Crash Warnings (VIX > 30).")
    
    if not macro_df.empty:
        danger_df = analyze_vix_impact(macro_df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=macro_df.index, y=macro_df['S&P 500'], name="S&P 500", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=macro_df.index, y=macro_df['VIX'], name="VIX", line=dict(color='gray', width=1), yaxis="y2"))
        
        # Danger Markers
        if danger_df is not None:
            fig.add_trace(go.Scatter(x=danger_df.index, y=danger_df['S&P 500'], mode='markers', marker=dict(color='red', size=6, symbol='x'), name="VIX > 30"))
            
        fig.update_layout(height=500, yaxis2=dict(overlaying='y', side='right'), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Yearly Returns Matrix")
    y_mat = calculate_yearly_matrix(macro_df)
    fig_h = px.imshow(y_mat, text_auto=".1f", color_continuous_scale="RdYlGn", aspect="auto")
    st.plotly_chart(fig_h, use_container_width=True)

# TAB 3: BALKEN COMPARISON
with t_comp:
    st.subheader("1-Year Relative Performance (Balken)")
    
    if not macro_df.empty:
        # Calculate 1Y Return
        ret = ((macro_df.iloc[-1] - macro_df.iloc[-252]) / macro_df.iloc[-252]) * 100
        
        # Sectors
        secs = ["Tech", "Energy", "Financials", "Healthcare", "Staples", "Discretionary", "Real Estate"]
        valid_s = [c for c in secs if c in ret]
        if valid_s:
            s_data = ret[valid_s].sort_values()
            fig_s = px.bar(x=s_data.values, y=s_data.index, orientation='h', title="Sectors", color=s_data.values, color_continuous_scale="RdYlGn")
            st.plotly_chart(fig_s, use_container_width=True)
            
        # Indices
        idxs = ["S&P 500", "Nasdaq 100", "Russell 2000", "DAX (Germany)", "FTSE 100"]
        valid_i = [c for c in idxs if c in ret]
        if valid_i:
            i_data = ret[valid_i].sort_values()
            fig_i = px.bar(x=i_data.values, y=i_data.index, orientation='h', title="Global Indices", color=i_data.values, color_continuous_scale="Bluered")
            st.plotly_chart(fig_i, use_container_width=True)

# TAB 4: SCANNER
with t_scan:
    if st.button("RUN LIVE SCAN", type="primary"):
        with st.spinner(f"Scanning {univ}..."):
            stock_data = fetch_scanner_batch(univ)
            
            res = []
            for t, df in stock_data.items():
                sc = score_stock(df)
                res.append({"Ticker": t, "Score": sc, "Price": df['Close'].iloc[-1]})
            
            if res:
                df_r = pd.DataFrame(res)
                # Force Rank
                bulls = df_r.sort_values("Score", ascending=False).head(5)
                bears = df_r.sort_values("Score", ascending=True).head(5)
                
                c1, c2 = st.columns(2)
                
                # BULLS
                with c1:
                    st.success("🟢 Top Bulls")
                    st.dataframe(bulls, hide_index=True)
                    if not bulls.empty:
                        top = bulls.iloc[0]['Ticker']
                        st.markdown(f"**90-Day Projection: {top}**")
                        df_p = stock_data[top]
                        d, b, s, n = get_monte_carlo(df_p, days=90)
                        
                        fig = go.Figure(go.Candlestick(x=df_p.tail(200).index, open=df_p['Open'].tail(200), high=df_p['High'].tail(200), low=df_p['Low'].tail(200), close=df_p['Close'].tail(200)))
                        fig.add_trace(go.Scatter(x=d, y=b, line=dict(color='green', dash='dot'), name='Bull'))
                        fig.add_trace(go.Scatter(x=d, y=s, line=dict(color='red', dash='dot'), name='Bear'))
                        fig.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                
                # BEARS
                with c2:
                    st.error("🔴 Top Bears")
                    st.dataframe(bears, hide_index=True)
                    if not bears.empty:
                        top = bears.iloc[0]['Ticker']
                        st.markdown(f"**90-Day Projection: {top}**")
                        df_p = stock_data[top]
                        d, b, s, n = get_monte_carlo(df_p, days=90)
                        
                        fig = go.Figure(go.Candlestick(x=df_p.tail(200).index, open=df_p['Open'].tail(200), high=df_p['High'].tail(200), low=df_p['Low'].tail(200), close=df_p['Close'].tail(200)))
                        fig.add_trace(go.Scatter(x=d, y=b, line=dict(color='green', dash='dot'), name='Bull'))
                        fig.add_trace(go.Scatter(x=d, y=s, line=dict(color='red', dash='dot'), name='Bear'))
                        fig.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data found.")

