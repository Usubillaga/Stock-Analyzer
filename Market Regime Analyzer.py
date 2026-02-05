import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import requests
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="Titan: Macro-Technical Scanner")

# Custom CSS for a professional look
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .bull-text { color: #00FF7F; font-weight: bold; }
    .bear-text { color: #FF5252; font-weight: bold; }
    .highlight { background-color: #333; padding: 2px 5px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE (TICKERS & MACRO)
# ==========================================

@st.cache_data(ttl=86400)
def get_market_tickers(index_name):
    """Scrapes Wikipedia for S&P 500 or Nasdaq 100 tickers."""
    tickers = []
    try:
        if index_name == "S&P 500":
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            tickers = tables[0]['Symbol'].tolist()
        elif index_name == "Nasdaq 100":
            url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            tables = pd.read_html(url)
            # Find table with Ticker/Symbol
            for t in tables:
                if 'Ticker' in t.columns:
                    tickers = t['Ticker'].tolist(); break
                if 'Symbol' in t.columns:
                    tickers = t['Symbol'].tolist(); break
        
        # Clean tickers for yfinance (e.g. BRK.B -> BRK-B)
        tickers = [str(t).replace('.', '-') for t in tickers]
        return tickers
    except:
        return ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "TSLA", "META", "AMD", "NFLX"]

@st.cache_data(ttl=3600)
def get_macro_data():
    """Fetches Commodities, Yields, and Sector ETFs."""
    # CL=F (Oil), HG=F (Copper), GC=F (Gold), ^TNX (10Y Yield), LBS=F (Lumber)
    # XLY (Discretionary), XLP (Staples)
    tickers = ["CL=F", "HG=F", "GC=F", "^TNX", "LBS=F", "XLY", "XLP", "SPY"]
    try:
        df = yf.download(tickers, period="6mo", progress=False)['Close']
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def process_batch_technicals(ticker_list):
    """
    Downloads and calculates indicators for a list of stocks.
    Optimized for speed using batch downloading.
    """
    if not ticker_list: return {}
    
    # Batch download
    df_bulk = yf.download(ticker_list, period="1y", group_by='ticker', progress=False, threads=True)
    
    results = {}
    
    for t in ticker_list:
        try:
            # Extract single ticker dataframe
            if len(ticker_list) > 1:
                df = df_bulk[t].copy()
            else:
                df = df_bulk.copy()
            
            # Clean data
            if df.empty: continue
            df = df.dropna(subset=['Close'])
            if len(df) < 200: continue
            
            # --- CALCULATE INDICATORS ---
            
            # 1. Trend & Structure
            df.ta.sma(length=50, append=True)
            df.ta.sma(length=200, append=True)
            
            # 2. Momentum
            df.ta.rsi(length=14, append=True)
            
            # 3. Volatility / Amplitude
            df.ta.bbands(length=20, std=2, append=True) # Adds BBL_20, BBU_20
            df.ta.adx(length=14, append=True)            # Adds ADX_14
            df.ta.atr(length=14, append=True)            # Adds ATRr_14
            
            # 4. Volume
            df['Vol_SMA'] = df['Volume'].rolling(20).mean()
            df['RVol'] = df['Volume'] / df['Vol_SMA'] # Relative Volume
            
            # Rename for easy access
            df['BB_Upper'] = df['BBU_20_2.0']
            df['BB_Lower'] = df['BBL_20_2.0']
            df['RSI'] = df['RSI_14']
            df['ADX'] = df['ADX_14']
            df['ATR'] = df['ATRr_14']

            results[t] = df
            
        except Exception:
            continue
            
    return results

# ==========================================
# 3. ANALYSIS LOGIC
# ==========================================

def get_macro_regime(macro_df, sim_mode=False, sim_inflation=0, sim_yield=0):
    """
    Determines if we are in Bull, Bear, or Stagflation mode.
    Can be overridden by simulation sliders.
    """
    regime = {}
    
    if sim_mode:
        # --- SIMULATION LOGIC ---
        if sim_inflation > 6.0 or sim_yield > 5.0:
            regime = {"status": "EXTREME BEARISH", "color": "red", "msg": "Hyperinflation/Rate Shock Simulated."}
        elif sim_inflation > 4.0:
            regime = {"status": "STAGFLATION", "color": "orange", "msg": "Simulated High Inflation."}
        elif sim_inflation < 2.5 and sim_yield < 3.5:
            regime = {"status": "GOLDILOCKS BULL", "color": "green", "msg": "Simulated Low Rate/Inflation Environment."}
        else:
            regime = {"status": "NEUTRAL", "color": "gray", "msg": "Standard Simulation."}
        return regime, {}
    
    # --- LIVE DATA LOGIC ---
    curr = macro_df.iloc[-1]
    prev = macro_df.iloc[-20] # 1 month ago
    pct = ((curr - prev) / prev) * 100
    
    # Inflation Signals (Oil & Copper)
    inflation_score = 0
    if pct.get('CL=F', 0) > 5: inflation_score += 1
    if pct.get('HG=F', 0) > 5: inflation_score += 1
    
    # Risk Signals (Discretionary vs Staples)
    # If XLY outperforms XLP, market is Risk-On
    risk_on = pct.get('XLY', 0) > pct.get('XLP', 0)
    
    # Yield Pressure
    high_rates = curr.get('^TNX', 0) > 4.5
    
    if high_rates and inflation_score >= 1:
        regime = {"status": "BEARISH (Rate Pressure)", "color": "red", "msg": "Rising Yields & Commodity Costs. Caution."}
    elif risk_on and not high_rates:
        regime = {"status": "BULLISH (Growth)", "color": "green", "msg": "Risk-On flows active. Rates stable."}
    elif not risk_on and inflation_score == 0:
        regime = {"status": "DEFENSIVE", "color": "orange", "msg": "Market rotating to safety (Staples/Gold)."}
    else:
        regime = {"status": "CHOPPY / NEUTRAL", "color": "gray", "msg": "Mixed macro signals."}
        
    return regime, pct

def calculate_score(df, trend_bias="bull"):
    """
    Scoring Algorithm (0-100).
    Combines Structure, Momentum, and Amplitude.
    """
    row = df.iloc[-1]
    score = 0
    reasons = []
    
    # 1. MARKET STRUCTURE (40 pts)
    # Bull: Price > 200 SMA & 50 SMA
    if trend_bias == "bull":
        if row['Close'] > row['SMA_200']: score += 25
        if row['Close'] > row['SMA_50']: score += 15
    # Bear: Price < 200 SMA & 50 SMA
    else:
        if row['Close'] < row['SMA_200']: score += 25
        if row['Close'] < row['SMA_50']: score += 15
        
    # 2. MOMENTUM RSI (20 pts)
    # Bull: RSI not yet overbought (room to grow)
    if trend_bias == "bull" and 40 < row['RSI'] < 70: score += 20
    # Bear: RSI not yet oversold (room to fall)
    if trend_bias == "bear" and 30 < row['RSI'] < 60: score += 20
    
    # 3. AMPLITUDE & VOLATILITY (20 pts)
    # ADX > 20 means strong trend
    if row['ADX'] > 20: score += 10
    
    # Bollinger Band Interaction
    if trend_bias == "bull":
        # Squeeze Breakout or Riding Upper Band
        if row['Close'] > row['BB_Upper'] * 0.98: 
            score += 10
            reasons.append("Testing Upper Band")
    else:
        # Breakdown or Riding Lower Band
        if row['Close'] < row['BB_Lower'] * 1.02:
            score += 10
            reasons.append("Testing Lower Band")

    # 4. VOLUME (20 pts)
    if row['RVol'] > 1.2:
        score += 20
        reasons.append("High Volume")
        
    return score, reasons

# ==========================================
# 4. FRONTEND SIDEBAR
# ==========================================

st.sidebar.title("⚙️ Titan Control Panel")

# A. Macro Settings
st.sidebar.header("1. Macro Environment")
macro_mode = st.sidebar.radio("Macro Data Source:", ["Auto-Live Data", "Manual Simulation"])

sim_inf = 3.0
sim_yld = 4.0

if macro_mode == "Manual Simulation":
    st.sidebar.warning("🛠️ Manual Mode Active")
    sim_inf = st.sidebar.slider("Simulate Inflation (%)", 0.0, 20.0, 3.0)
    sim_yld = st.sidebar.slider("Simulate 10Y Yield (%)", 0.0, 10.0, 4.0)

# B. Scanner Settings
st.sidebar.header("2. Asset Universe")
universe = st.sidebar.selectbox("Select Market:", ["Nasdaq 100", "S&P 500", "Custom Watchlist"])
custom_list = ""
if universe == "Custom Watchlist":
    custom_list = st.sidebar.text_area("Tickers (comma sep)", "NVDA, TSLA, AAPL, AMD, COIN, MSTR")

st.sidebar.header("3. Filters")
min_conf = st.sidebar.slider("Min Confidence Score", 0, 100, 70)
limit_num = st.sidebar.number_input("Max Stocks to Scan", 10, 500, 50)

if st.sidebar.button("🚀 RUN ANALYSIS"):
    active_scan = True
else:
    active_scan = False

# ==========================================
# 5. MAIN DASHBOARD
# ==========================================

st.title("🦅 Titan: Macro & Amplitude Scanner")

# --- MACRO SECTION ---
macro_data = get_macro_data()
if not macro_data.empty:
    regime, changes = get_macro_regime(macro_data, 
                                       sim_mode=(macro_mode=="Manual Simulation"),
                                       sim_inflation=sim_inf, 
                                       sim_yield=sim_yld)

    # Display Macro Header
    st.markdown(f"""
    <div style='background-color:#111; padding:20px; border-radius:10px; border-left: 10px solid {regime['color']}'>
        <h2 style='margin:0; color:{regime['color']}'>{regime['status']}</h2>
        <p style='margin:0; font-size:18px;'>{regime['msg']}</p>
    </div>
    """, unsafe_allow_html=True)

    if macro_mode == "Auto-Live Data":
        # Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Oil (Inflation)", f"{macro_data['CL=F'].iloc[-1]:.2f}", f"{changes.get('CL=F',0):.2f}%")
        c2.metric("Copper (Economy)", f"{macro_data['HG=F'].iloc[-1]:.2f}", f"{changes.get('HG=F',0):.2f}%")
        c3.metric("10Y Yield", f"{macro_data['^TNX'].iloc[-1]:.2f}%", f"{changes.get('^TNX',0):.2f}%")
        c4.metric("Risk Sentiment", "Risk On" if changes.get('XLY',0) > changes.get('XLP',0) else "Risk Off")

# --- STOCK SCANNER SECTION ---
if active_scan:
    st.divider()
    
    # 1. Define List
    if universe == "Custom Watchlist":
        tickers = [x.strip().upper() for x in custom_list.split(',')]
    else:
        with st.spinner(f"Fetching {universe} components..."):
            tickers = get_market_tickers(universe)
            tickers = tickers[:limit_num] # Apply limit

    # 2. Process Data
    status_text = st.empty()
    status_text.text(f"Scanning {len(tickers)} assets... Please wait.")
    progress_bar = st.progress(0)
    
    stock_data = process_batch_technicals(tickers)
    
    bullish_results = []
    bearish_results = []

    # 3. Analyze Each Stock
    for i, (t, df) in enumerate(stock_data.items()):
        # Bull Score
        b_score, b_reasons = calculate_score(df, "bull")
        if b_score >= min_conf:
            # Macro Veto: If macro is Extreme Bearish, penalize bulls
            if regime['status'] == "EXTREME BEARISH": b_score -= 30
            
            if b_score >= min_conf:
                bullish_results.append({
                    "Ticker": t, "Score": b_score, "Price": df['Close'].iloc[-1],
                    "RSI": df['RSI'].iloc[-1], "ADX": df['ADX'].iloc[-1],
                    "Reason": ", ".join(b_reasons)
                })

        # Bear Score
        s_score, s_reasons = calculate_score(df, "bear")
        if s_score >= min_conf:
             bearish_results.append({
                "Ticker": t, "Score": s_score, "Price": df['Close'].iloc[-1],
                "RSI": df['RSI'].iloc[-1], "ADX": df['ADX'].iloc[-1],
                "Reason": ", ".join(s_reasons)
            })
            
        progress_bar.progress((i + 1) / len(stock_data))
    
    status_text.empty()
    progress_bar.empty()

    # 4. Display Results
    
    col_bull, col_bear = st.columns(2)
    
    # --- BULLISH OUTPUT ---
    with col_bull:
        st.subheader("🟢 Bullish / Buy Setups")
        if bullish_results:
            df_bull = pd.DataFrame(bullish_results).sort_values("Score", ascending=False)
            st.dataframe(df_bull.style.background_gradient(subset=["Score"], cmap="Greens"), hide_index=True, use_container_width=True)
            
            # Deep Dive Chart
            top_bull = df_bull.iloc[0]['Ticker']
            st.markdown(f"**Top Long Pick: {top_bull}**")
            
            # Interactive Plotly Chart
            df_p = stock_data[top_bull].tail(100)
            fig = go.Figure()
            # Candles
            fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name='Price'))
            # Bollinger Bands
            fig.add_trace(go.Scatter(x=df_p.index, y=df_p['BB_Upper'], line=dict(color='rgba(0, 255, 127, 0.4)'), name='Upper BB'))
            fig.add_trace(go.Scatter(x=df_p.index, y=df_p['BB_Lower'], line=dict(color='rgba(0, 255, 127, 0.4)'), name='Lower BB'))
            # 200 SMA
            fig.add_trace(go.Scatter(x=df_p.index, y=df_p['SMA_200'], line=dict(color='blue', width=2), name='200 SMA'))
            
            fig.update_layout(height=400, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No strong bullish setups found based on current criteria.")

    # --- BEARISH OUTPUT ---
    with col_bear:
        st.subheader("🔴 Bearish / Sell Setups")
        if bearish_results:
            df_bear = pd.DataFrame(bearish_results).sort_values("Score", ascending=False)
            st.dataframe(df_bear.style.background_gradient(subset=["Score"], cmap="Reds"), hide_index=True, use_container_width=True)
            
            # Deep Dive Chart
            top_bear = df_bear.iloc[0]['Ticker']
            st.markdown(f"**Top Short Pick: {top_bear}**")
            
            # Interactive Plotly Chart
            df_p = stock_data[top_bear].tail(100)
            fig = go.Figure()
            # Candles
            fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name='Price'))
            # Bollinger Bands
            fig.add_trace(go.Scatter(x=df_p.index, y=df_p['BB_Upper'], line=dict(color='rgba(255, 82, 82, 0.4)'), name='Upper BB'))
            fig.add_trace(go.Scatter(x=df_p.index, y=df_p['BB_Lower'], line=dict(color='rgba(255, 82, 82, 0.4)'), name='Lower BB'))
            # 200 SMA
            fig.add_trace(go.Scatter(x=df_p.index, y=df_p['SMA_200'], line=dict(color='blue', width=2), name='200 SMA'))
            
            fig.update_layout(height=400, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No strong bearish setups found based on current criteria.")

elif not active_scan:
    st.info("👈 Select settings in the sidebar and click 'RUN ANALYSIS' to start.")
