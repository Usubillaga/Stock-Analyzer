import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# 1. UI CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="Titan: Omni-Scanner")

st.markdown("""
<style>
    /* Global Styles */
    .stApp { background-color: #F0F2F6; color: #111; font-family: 'Segoe UI', sans-serif; }
    
    /* Metrics & Cards */
    div[data-testid="stMetric"] { background-color: #FFF; border: 1px solid #ddd; padding: 10px; border-radius: 5px; }
    .card { background-color: #fff; padding: 20px; border-radius: 10px; border: 1px solid #ddd; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    
    /* Regime Badges */
    .regime-bull { border-left: 10px solid #2ecc71; background-color: #e8f8f5; padding: 15px; }
    .regime-bear { border-left: 10px solid #e74c3c; background-color: #fdedec; padding: 15px; }
    .regime-warn { border-left: 10px solid #f1c40f; background-color: #fef9e7; padding: 15px; }
    
    /* Signal Text */
    .buy-sig { color: #27ae60; font-weight: 900; }
    .sell-sig { color: #c0392b; font-weight: 900; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE (ROBUST)
# ==========================================

@st.cache_data(ttl=3600)
def fetch_macro_data():
    """Fetches Macro, Indices, and Commodities."""
    tickers = {
        # Indices
        "S&P 500": "SPY", "Nasdaq 100": "QQQ", "Russell 2000": "IWM", "DAX": "DAX",
        # Breadth
        "S&P Equal Wgt": "RSP",
        # Commodities
        "Oil": "USO", "Gold": "GLD", "Copper": "CPER", "Lumber": "WOOD", "Silver": "SLV",
        # Rates/VIX
        "10Y Yield": "^TNX", "VIX": "^VIX",
        # Sectors
        "Tech": "XLK", "Energy": "XLE", "Staples": "XLP", "Discretionary": "XLY"
    }
    
    data = yf.download(list(tickers.values()), period="1y", progress=False, auto_adjust=True)
    
    # Flatten MultiIndex
    clean = pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        for k, v in tickers.items():
            try:
                if v in data.columns.levels[0]: clean[k] = data[v]['Close']
                elif v in data.columns.levels[1]: clean[k] = data.xs(v, axis=1, level=1)['Close']
            except: pass
    else:
        for k, v in tickers.items():
            if v in data: clean[k] = data[v]
            
    clean.fillna(method='ffill', inplace=True)
    return clean

@st.cache_data(ttl=600)
def fetch_scanner_batch(universe_name):
    """
    Fetches a hardcoded list of liquid stocks to ensure results.
    """
    if universe_name == "High Growth / Tech":
        # Ensure we have enough stocks so the scanner always finds something
        ticks = ["NVDA", "AMD", "TSLA", "PLTR", "COIN", "MARA", "MSTR", "HOOD", "SOFI", "UBER", "DKNG", "ROKU", "NET", "CRWD", "SNOW", "PANW", "ZS", "CVNA", "UPST", "AI", "IONQ", "RIVN", "LCID", "NIO", "BABA", "PDD", "JD", "BIDU", "TCEHY", "SE"]
    elif universe_name == "Mega Cap Stability":
        ticks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "BRK-B", "JPM", "V", "MA", "PG", "JNJ", "XOM", "CVX", "COST", "PEP", "KO", "MCD", "WMT", "HD", "LOW", "LIN", "LLY", "UNH", "ABBV", "MRK", "PFE", "TMO", "DHR", "ABT"]
    else: # DAX / Europe
        ticks = ["SAP", "SIE.DE", "ALV.DE", "DTE.DE", "BMW.DE", "VOW3.DE", "BAS.DE", "ADS.DE", "AIR.DE", "DHL.DE", "DB1.DE", "MUV2.DE", "IFX.DE", "HEN3.DE", "RWE.DE", "EOAN.DE", "BAYN.DE", "CON.DE", "HNR1.DE", "HEI.DE", "FRE.DE", "FME.DE", "BEI.DE", "MTX.DE", "SY1.DE", "DBK.DE", "CBK.DE", "VNA.DE", "ZAL.DE", "HFG.DE"]

    data = {}
    try:
        raw = yf.download(ticks, period="1y", group_by='ticker', progress=False, threads=True, auto_adjust=True)
        for t in ticks:
            try:
                if isinstance(raw.columns, pd.MultiIndex): df = raw[t].copy()
                else: continue
                
                df.dropna(subset=['Close'], inplace=True)
                if len(df) < 50: continue
                
                # --- CALCULATE SIGNALS ---
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
# 3. ANALYSIS LOGIC
# ==========================================

def macro_analyst(df, in_gdp, in_unemp, in_cpi):
    """Combines inputs with market data."""
    if df.empty: return {}, []
    
    curr = df.iloc[-1]
    prev = df.iloc[-22]
    chg = ((curr - prev)/prev)*100
    
    # Logic
    score = 0
    advice = []
    
    # 1. Inflation (CPI + Commodities)
    comm_basket = (chg.get('Oil',0) + chg.get('Copper',0) + chg.get('Lumber',0))/3
    if in_cpi > 4.0 or comm_basket > 3.0:
        advice.append(f"🔥 **Inflationary Pressure:** CPI {in_cpi}% & Commodities rising. Bad for PE ratios.")
        score -= 2
    else:
        advice.append("✅ **Inflation Stable:** Commodities are behaving. Supportive for stocks.")
        score += 1
        
    # 2. Growth (GDP + Copper)
    if in_gdp < 1.0 or in_unemp > 5.0:
        advice.append("⚠️ **Recession Risk:** GDP/Employment data is weak.")
        score -= 2
    else:
        advice.append("🏗️ **Growth Intact:** Economy appears stable.")
        score += 1
        
    # 3. Rates
    if curr.get('10Y Yield',0) > 4.5:
        advice.append("💸 **High Rates:** 10Y Yield > 4.5% is a headwind.")
        score -= 1
        
    # Verdict
    if score > 0: regime = {"status": "BULLISH (RISK ON)", "css": "regime-bull"}
    elif score < 0: regime = {"status": "BEARISH (DEFENSIVE)", "css": "regime-bear"}
    else: regime = {"status": "NEUTRAL / CHOPPY", "css": "regime-warn"}
    
    return regime, advice

def score_stock(df):
    """
    Returns a score (-100 to 100).
    """
    row = df.iloc[-1]
    s = 0
    
    # Trend (40%)
    if row['Close'] > row['SMA_200']: s += 20
    else: s -= 20
    if row['Close'] > row['SMA_50']: s += 20
    else: s -= 20
    
    # Momentum (30%)
    if row['RSI'] < 30: s += 30      # Oversold bounce
    elif row['RSI'] > 75: s -= 30    # Overbought dump
    elif 50 < row['RSI'] < 70: s += 10 # Strong trend
    
    # Volatility (30%)
    # Price touching lower band?
    if row['Close'] <= row['BB_Lower'] * 1.01: s += 30
    # Price touching upper band?
    if row['Close'] >= row['BB_Upper'] * 0.99: s -= 30
    
    return s

def get_signal_label(score):
    if score >= 50: return "STRONG BUY"
    elif score >= 20: return "BUY"
    elif score <= -50: return "STRONG SELL"
    elif score <= -20: return "SELL"
    else: return "NEUTRAL"

def get_monte_carlo(df, days=30):
    last = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    
    dates = [df.index[-1] + timedelta(days=i) for i in range(1, days+1)]
    bull = [last + (atr * 0.5 * i) for i in range(1, days+1)]
    bear = [last - (atr * 0.5 * i) for i in range(1, days+1)]
    
    return dates, bull, bear

# ==========================================
# 4. APP LAYOUT
# ==========================================

st.title("🦅 Titan: Omni-Scanner & Macro Analyst")

# SIDEBAR
st.sidebar.header("1. Economic Inputs")
in_gdp = st.sidebar.number_input("GDP Growth (%)", 2.5)
in_unemp = st.sidebar.number_input("Unemployment (%)", 3.8)
in_cpi = st.sidebar.number_input("CPI Inflation (%)", 3.2)
st.sidebar.markdown("---")
st.sidebar.header("2. Scanner Config")
univ = st.sidebar.selectbox("Market Universe", ["High Growth / Tech", "Mega Cap Stability", "DAX / Europe"])

# LOAD DATA
with st.spinner("Analyzing Global Markets..."):
    macro_df = fetch_macro_data()
    regime, advice = macro_analyst(macro_df, in_gdp, in_unemp, in_cpi)

# TABS
t1, t2, t3 = st.tabs(["🌍 Macro Analyst", "📊 Breadth & Sectors", "🚀 Stock Scanner"])

# --- TAB 1: MACRO ---
with t1:
    st.markdown(f"<div class='{regime['css']}'><h2>MARKET REGIME: {regime['status']}</h2></div>", unsafe_allow_html=True)
    st.markdown("### 🧠 AI Analyst Advice")
    for a in advice: st.info(a)
    
    st.divider()
    curr = macro_df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VIX", f"{curr.get('VIX',0):.2f}")
    c2.metric("10Y Yield", f"{curr.get('10Y Yield',0):.2f}%")
    c3.metric("Oil (USO)", f"${curr.get('Oil',0):.2f}")
    c4.metric("Copper", f"${curr.get('Copper',0):.2f}")

# --- TAB 2: BREADTH ---
with t2:
    st.subheader("Market Breadth: Equal Weight (RSP) vs Cap Weight (SPY)")
    if 'S&P 500' in macro_df and 'S&P Equal Wgt' in macro_df:
        df_b = macro_df[['S&P 500', 'S&P Equal Wgt']].copy()
        df_b = (df_b / df_b.iloc[0]) * 100
        st.line_chart(df_b)
        
    st.subheader("Inflation Basket (Gold, Oil, Copper, Lumber)")
    coms = ['Gold', 'Oil', 'Copper', 'Lumber']
    valid = [c for c in coms if c in macro_df]
    if valid:
        df_c = (macro_df[valid] / macro_df[valid].iloc[0]) * 100
        st.line_chart(df_c)

# --- TAB 3: SCANNER ---
with t3:
    col_k, col_v = st.columns([1, 4])
    with col_k:
        run = st.button("RUN LIVE SCAN", type="primary", use_container_width=True)
    
    if run:
        with st.spinner(f"Scanning {univ} for opportunities..."):
            stock_data = fetch_scanner_batch(univ)
            
            results = []
            for t, df in stock_data.items():
                sc = score_stock(df)
                sig = get_signal_label(sc)
                last_p = df['Close'].iloc[-1]
                rsi = df['RSI'].iloc[-1]
                
                results.append({
                    "Ticker": t, 
                    "Signal": sig,
                    "Score": sc, 
                    "Price": last_p,
                    "RSI": rsi
                })
            
            if not results:
                st.error("No data returned. Market might be closed.")
            else:
                df_res = pd.DataFrame(results)
                
                # Filter Bulls & Bears
                bulls = df_res[df_res['Score'] > 10].sort_values("Score", ascending=False).head(10)
                bears = df_res[df_res['Score'] < -10].sort_values("Score", ascending=True).head(10)
                
                # --- UI OUTPUT ---
                c_bull, c_bear = st.columns(2)
                
                with c_bull:
                    st.success("🟢 TOP BULLISH (LONG)")
                    st.dataframe(bulls[['Ticker', 'Signal', 'Price', 'Score']], hide_index=True)
                    
                    if not bulls.empty:
                        top = bulls.iloc[0]['Ticker']
                        st.markdown(f"**Bull Case Projection: {top}**")
                        df_p = stock_data[top]
                        d, b, s = get_monte_carlo(df_p)
                        
                        fig = go.Figure(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name='Price'))
                        fig.add_trace(go.Scatter(x=d, y=b, line=dict(color='green', dash='dot'), name='Bull Path'))
                        fig.update_layout(height=350, template="plotly_white", margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                with c_bear:
                    st.error("🔴 TOP BEARISH (SHORT)")
                    st.dataframe(bears[['Ticker', 'Signal', 'Price', 'Score']], hide_index=True)
                    
                    if not bears.empty:
                        top = bears.iloc[0]['Ticker']
                        st.markdown(f"**Bear Case Projection: {top}**")
                        df_p = stock_data[top]
                        d, b, s = get_monte_carlo(df_p)
                        
                        fig = go.Figure(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name='Price'))
                        fig.add_trace(go.Scatter(x=d, y=s, line=dict(color='red', dash='dot'), name='Bear Path'))
                        fig.update_layout(height=350, template="plotly_white", margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

