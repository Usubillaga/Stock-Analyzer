import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==========================================
# 1. PROFESSIONAL UI CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="Titan: Global Macro & Technical Engine")

# Custom Dark Mode CSS for a "Bloomberg Terminal" feel
st.markdown("""
<style>
    /* Global Background & Font */
    .stApp { background-color: #0e1117; color: #FAFAFA; font-family: 'Roboto', sans-serif; }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] { font-size: 24px; color: #FAFAFA; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #888; }
    
    /* Custom Cards */
    .card { background-color: #1c1f26; padding: 20px; border-radius: 10px; border: 1px solid #333; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .bull-card { border-left: 5px solid #00C853; }
    .bear-card { border-left: 5px solid #FF5252; }
    .neutral-card { border-left: 5px solid #FFAB00; }
    
    /* Headers */
    h1, h2, h3 { font-weight: 300; letter-spacing: 1px; }
    .highlight { color: #4FC3F7; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ADVANCED DATA ENGINE
# ==========================================

@st.cache_data(ttl=3600)
def get_macro_data():
    """
    Fetches comprehensive macro assets: Commodities, Yields, and 11 Sector ETFs.
    """
    tickers = {
        # Commodities & Yields
        "Oil": "CL=F", "Gold": "GC=F", "Copper": "HG=F", "10Y Yield": "^TNX", "VIX": "^VIX",
        
        # Benchmark
        "S&P 500": "SPY",
        
        # Sectors
        "Tech": "XLK", "Financials": "XLF", "Energy": "XLE", "Healthcare": "XLV",
        "Discretionary": "XLY", "Staples": "XLP", "Materials": "XLB", "Utilities": "XLU",
        "Industrials": "XLI", "Real Estate": "XLRE", "Comms": "XLC"
    }
    
    # Download 6 months of data
    df = yf.download(list(tickers.values()), period="6mo", progress=False)['Close']
    
    # Rename columns
    rev_map = {v: k for k, v in tickers.items()}
    df.columns = [rev_map.get(c, c) for c in df.columns]
    
    return df

@st.cache_data(ttl=86400)
def get_market_universe(choice):
    """Fetches tickers for S&P 500 or Nasdaq 100."""
    try:
        if choice == "S&P 500":
            payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            return [x.replace('.', '-') for x in payload['Symbol'].tolist()]
        elif choice == "Nasdaq 100":
            payload = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
            for t in payload:
                if 'Ticker' in t.columns: return [x.replace('.', '-') for x in t['Ticker'].tolist()]
                if 'Symbol' in t.columns: return [x.replace('.', '-') for x in t['Symbol'].tolist()]
    except:
        return ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD"] # Fallback

@st.cache_data(ttl=3600)
def batch_technical_analysis(ticker_list):
    """
    Massive batch processor for technicals.
    """
    if not ticker_list: return {}
    df_bulk = yf.download(ticker_list, period="1y", group_by='ticker', progress=False, threads=True)
    results = {}
    
    for t in ticker_list:
        try:
            # Handle MultiIndex
            df = df_bulk[t].copy() if len(ticker_list) > 1 else df_bulk.copy()
            
            if df.empty or df['Close'].isnull().all(): continue
            df.dropna(inplace=True)
            if len(df) < 100: continue
            
            # --- CALCULATIONS ---
            # Structure
            df.ta.sma(length=50, append=True)
            df.ta.sma(length=200, append=True)
            # Momentum
            df.ta.rsi(length=14, append=True)
            # Volatility
            df.ta.bbands(length=20, std=2, append=True)
            df.ta.adx(length=14, append=True)
            # Volume
            df['Vol_SMA'] = df['Volume'].rolling(20).mean()
            df['RVol'] = df['Volume'] / df['Vol_SMA']
            
            # Friendly Names
            df['BB_Upper'] = df['BBU_20_2.0']
            df['BB_Lower'] = df['BBL_20_2.0']
            
            results[t] = df
        except: continue
    return results

# ==========================================
# 3. LOGIC & INTELLIGENCE
# ==========================================

def analyze_sectors(macro_df):
    """
    Compares Sector Performance relative to SPY (Alpha).
    Determines Cyclical vs Defensive rotation.
    """
    curr = macro_df.iloc[-1]
    start = macro_df.iloc[-22] # 1 month ago
    
    # Calculate % Return over last month
    returns = ((curr - start) / start) * 100
    
    # Calculate Relative Strength (Alpha) vs SPY
    spy_ret = returns['S&P 500']
    alpha = returns - spy_ret
    
    # Logic: Risk On/Off
    # Compare Cyclical (XLY) vs Defensive (XLP)
    risk_ratio = returns['Discretionary'] - returns['Staples']
    inflation_pressure = returns['Energy'] + returns['Materials']
    
    analysis = {
        "returns": returns,
        "alpha": alpha,
        "risk_on": risk_ratio > 0,
        "inflationary": inflation_pressure > 5.0,
        "leader": returns.idxmax(),
        "laggard": returns.idxmin()
    }
    return analysis

def get_market_narrative(analysis, macro_df):
    """Generates the AI textual analysis."""
    narrative = []
    
    # 1. Risk Sentiment
    if analysis['risk_on']:
        narrative.append("🟢 **Risk-On Environment:** Investors are favoring Cyclicals (Discretionary) over Staples. This suggests confidence in economic growth.")
        narrative.append("👉 **Approach:** Look for pullbacks in Tech and Consumer Discretionary to go Long.")
    else:
        narrative.append("🔴 **Risk-Off / Defensive:** Investors are hiding in Staples/Utilities. Fear is present.")
        narrative.append("👉 **Approach:** Avoid High Beta tech. Look for shorts in Consumer Discretionary or longs in Utilities/Gold.")

    # 2. Inflation/Rates
    yield_val = macro_df['10Y Yield'].iloc[-1]
    if yield_val > 4.5:
        narrative.append(f"⚠️ **High Rate Alert ({yield_val:.2f}%):** Yields are crushing valuations. Real Estate (XLRE) and Tech (XLK) face headwinds.")
    
    # 3. Materials/Inflation
    if analysis['inflationary']:
        narrative.append("🔥 **Inflation Trade Active:** Energy and Materials are outperforming. Look at stocks like XOM, CVX, FCX.")
        
    return narrative

def score_stock(df, regime_bias="neutral"):
    """
    0-100 Score.
    regime_bias: 'bull' (favor growth), 'bear' (favor defense), 'neutral'
    """
    row = df.iloc[-1]
    score_bull = 0
    score_bear = 0
    
    # --- BULLISH SCORING ---
    # Structure
    if row['Close'] > row['SMA_200']: score_bull += 30
    if row['Close'] > row['SMA_50']: score_bull += 10
    # Momentum (Dip Buying)
    if 40 < row['RSI_14'] < 60: score_bull += 20 # Sweet spot
    if row['RSI_14'] < 30: score_bull += 10 # Oversold bounce
    # Volatility
    if row['ADX_14'] > 25: score_bull += 10
    # Breakout
    if row['Close'] > row['BB_Upper'] * 0.99: score_bull += 20
    
    # --- BEARISH SCORING ---
    # Structure
    if row['Close'] < row['SMA_200']: score_bear += 30
    if row['Close'] < row['SMA_50']: score_bear += 10
    # Momentum (Overextended)
    if 40 < row['RSI_14'] < 60: score_bear += 20
    if row['RSI_14'] > 70: score_bear += 10
    # Breakdown
    if row['Close'] < row['BB_Lower'] * 1.01: score_bear += 20
    
    return score_bull, score_bear

# ==========================================
# 4. MAIN APP LAYOUT
# ==========================================

st.title("🦅 TITAN | Macro-Quant Analytics")

# Load Data
with st.spinner("Establishing Satellite Uplink... (Fetching Macro Data)"):
    macro_data = get_macro_data()
    sector_analysis = analyze_sectors(macro_data)
    narrative = get_market_narrative(sector_analysis, macro_data)

# TABS for Clean UI
tab_macro, tab_scanner = st.tabs(["🌍 Macro Headquarters", "🔭 Stock Sniper Scope"])

# ==========================================
# TAB 1: MACRO HEADQUARTERS
# ==========================================
with tab_macro:
    # 1. Top Level Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Calculate daily changes
    curr = macro_data.iloc[-1]
    prev = macro_data.iloc[-2]
    
    col1.metric("S&P 500", f"{curr['S&P 500']:.2f}", f"{(curr['S&P 500']/prev['S&P 500']-1)*100:.2f}%")
    col2.metric("10Y Yield", f"{curr['10Y Yield']:.2f}%", f"{(curr['10Y Yield']-prev['10Y Yield']):.2f}")
    col3.metric("Oil (WTI)", f"${curr['Oil']:.2f}", f"{(curr['Oil']/prev['Oil']-1)*100:.2f}%")
    col4.metric("VIX (Fear)", f"{curr['VIX']:.2f}", f"{(curr['VIX']-prev['VIX']):.2f}", delta_color="inverse")
    col5.metric("Gold", f"${curr['Gold']:.2f}", f"{(curr['Gold']/prev['Gold']-1)*100:.2f}%")

    st.divider()
    
    # 2. Sector Rotation Visualization
    c_chart, c_text = st.columns([2, 1])
    
    with c_chart:
        st.subheader("Sector Relative Performance (1 Month Alpha)")
        # Filter only sector columns
        sector_cols = ["Tech", "Financials", "Energy", "Healthcare", "Discretionary", 
                       "Staples", "Materials", "Utilities", "Industrials", "Real Estate", "Comms"]
        
        alpha_data = sector_analysis['alpha'][sector_cols].sort_values(ascending=True)
        
        # Color logic
        colors = ['#FF5252' if x < 0 else '#00E676' for x in alpha_data.values]
        
        fig = go.Figure(go.Bar(
            x=alpha_data.values,
            y=alpha_data.index,
            orientation='h',
            marker_color=colors
        ))
        fig.update_layout(
            template="plotly_dark", 
            title="Performance vs S&P 500",
            xaxis_title="Relative Strength (%)",
            height=400,
            margin=dict(l=0,r=0,t=40,b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with c_text:
        st.markdown("<div class='card neutral-card'>", unsafe_allow_html=True)
        st.markdown("### 🧠 AI Market Analyst")
        for line in narrative:
            st.markdown(line)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.info(f"**Top Performing Sector:** {sector_analysis['leader']} (+{sector_analysis['returns'][sector_analysis['leader']]:.1f}%)")
        st.info(f"**Weakest Link:** {sector_analysis['laggard']} ({sector_analysis['returns'][sector_analysis['laggard']]:.1f}%)")

# ==========================================
# TAB 2: STOCK SNIPER SCOPE
# ==========================================
with tab_scanner:
    
    # Controls
    with st.expander("⚙️ Scanner Settings", expanded=True):
        c1, c2, c3 = st.columns(3)
        universe = c1.selectbox("Market Universe", ["Nasdaq 100", "S&P 500", "Custom List"])
        if universe == "Custom List":
            custom_txt = st.text_area("Tickers", "NVDA, TSLA, AMD, AAPL, MSFT")
        min_score = c2.slider("Min Confidence Score", 0, 100, 75)
        max_stocks = c3.number_input("Speed Limit (Max Stocks)", 10, 500, 50)
        
        run_btn = st.button("🚀 INITIATE SCAN", type="primary", use_container_width=True)

    if run_btn:
        # Fetch Logic
        status = st.empty()
        status.info("📡 Scanning Satellite Imagery...")
        
        if universe == "Custom List":
            tickers = [x.strip().upper() for x in custom_txt.split(',')]
        else:
            tickers = get_market_universe(universe)[:max_stocks]
            
        data_dict = batch_technical_analysis(tickers)
        
        bull_list = []
        bear_list = []
        
        # Scoring Loop
        progress = st.progress(0)
        for i, (t, df) in enumerate(data_dict.items()):
            b_score, s_score = score_stock(df)
            
            last_price = df['Close'].iloc[-1]
            rsi = df['RSI_14'].iloc[-1]
            
            if b_score >= min_score:
                bull_list.append({"Ticker": t, "Score": b_score, "Price": last_price, "RSI": rsi})
            if s_score >= min_score:
                bear_list.append({"Ticker": t, "Score": s_score, "Price": last_price, "RSI": rsi})
            
            progress.progress((i+1)/len(data_dict))
            
        progress.empty()
        status.empty()
        
        # Display Results
        col_bull, col_bear = st.columns(2)
        
        # BULLISH RESULTS
        with col_bull:
            st.markdown("<h3 style='color:#00E676; border-bottom: 2px solid #00E676'>🐂 Bullish Opportunities</h3>", unsafe_allow_html=True)
            if bull_list:
                df_bull = pd.DataFrame(bull_list).sort_values("Score", ascending=False)
                st.dataframe(df_bull.style.background_gradient(subset=["Score"], cmap="Greens"), use_container_width=True, hide_index=True)
                
                # Charting Best Bull
                top_bull = df_bull.iloc[0]['Ticker']
                df_p = data_dict[top_bull].tail(100)
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name="Price"))
                fig.add_trace(go.Scatter(x=df_p.index, y=df_p['BB_Upper'], line=dict(color='green', width=1, dash='dot'), name="Upper Band"))
                fig.add_trace(go.Scatter(x=df_p.index, y=df_p['SMA_50'], line=dict(color='yellow', width=2), name="50 SMA"))
                fig.update_layout(title=f"Top Bull: {top_bull}", height=350, template="plotly_dark", margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No stocks met the Bullish criteria.")

        # BEARISH RESULTS
        with col_bear:
            st.markdown("<h3 style='color:#FF5252; border-bottom: 2px solid #FF5252'>🐻 Bearish Opportunities</h3>", unsafe_allow_html=True)
            if bear_list:
                df_bear = pd.DataFrame(bear_list).sort_values("Score", ascending=False)
                st.dataframe(df_bear.style.background_gradient(subset=["Score"], cmap="Reds"), use_container_width=True, hide_index=True)
                
                # Charting Best Bear
                top_bear = df_bear.iloc[0]['Ticker']
                df_p = data_dict[top_bear].tail(100)
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name="Price"))
                fig.add_trace(go.Scatter(x=df_p.index, y=df_p['BB_Lower'], line=dict(color='red', width=1, dash='dot'), name="Lower Band"))
                fig.add_trace(go.Scatter(x=df_p.index, y=df_p['SMA_50'], line=dict(color='yellow', width=2), name="50 SMA"))
                fig.update_layout(title=f"Top Bear: {top_bear}", height=350, template="plotly_dark", margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No stocks met the Bearish criteria.")
