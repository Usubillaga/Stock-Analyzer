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
st.set_page_config(layout="wide", page_title="Titan: Global Market Command")

st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #1e1e1e; font-family: 'Segoe UI', sans-serif; }
    div[data-testid="stMetric"] {
        background-color: #F8F9FA;
        border: 1px solid #dee2e6;
        padding: 10px;
        border-radius: 5px;
    }
    .card {
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 15px;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .bull-txt { color: #00C853; font-weight: bold; }
    .bear-txt { color: #D50000; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE (SPECIFIC & ROBUST)
# ==========================================

@st.cache_data(ttl=3600)
def get_market_data():
    """
    Fetches specific Indices and Sector ETFs.
    """
    # 1. Global Indices
    indices = {
        "S&P 500": "^GSPC",
        "Nasdaq 100": "^NDX",
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT",
        "DAX (Germany)": "^GDAXI",
        "FTSE 100 (UK)": "^FTSE",
        "Nikkei 225 (Japan)": "^N225",
        "Euro Stoxx 50": "^STOXX50E"
    }
    
    # 2. Sector ETFs (Specific Names)
    sectors = {
        "Technology (XLK)": "XLK",
        "Financials (XLF)": "XLF",
        "Energy (XLE)": "XLE",
        "Healthcare (XLV)": "XLV",
        "Consumer Disc (XLY)": "XLY",
        "Consumer Staples (XLP)": "XLP",
        "Materials (XLB)": "XLB",
        "Utilities (XLU)": "XLU",
        "Real Estate (XLRE)": "XLRE",
        "Industrials (XLI)": "XLI",
        "Comms (XLC)": "XLC"
    }
    
    # 3. Macro Commodities
    macro = {
        "Oil (WTI)": "CL=F",
        "Gold": "GC=F",
        "Copper": "HG=F",
        "10Y Yield": "^TNX",
        "VIX": "^VIX"
    }

    # Combine all tickers
    all_tickers = {**indices, **sectors, **macro}
    
    # Download
    data = yf.download(list(all_tickers.values()), period="1y", progress=False)['Close']
    
    # Rename columns to friendly names
    rev_map = {v: k for k, v in all_tickers.items()}
    data.columns = [rev_map.get(c, c) for c in data.columns]
    
    # Fill gaps
    data.fillna(method='ffill', inplace=True)
    
    return data, indices, sectors, macro

@st.cache_data(ttl=3600)
def run_scanner(market_choice):
    """
    Scans specifically selected markets.
    """
    if market_choice == "US Tech (Nasdaq)":
        tickers = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "INTC", "CSCO", "ADBE", "QCOM", "TXN", "AMGN"]
    elif market_choice == "Germany (DAX)":
        tickers = ["SIE.DE", "SAP.DE", "ALV.DE", "DTE.DE", "BMW.DE", "VOW3.DE", "BAS.DE", "ADS.DE", "AIR.DE", "DHL.DE"]
    else: # Crypto / High Beta
        tickers = ["COIN", "MSTR", "MARA", "RIOT", "HOOD", "SOFI", "PLTR", "UBER", "DKNG", "ROKU"]

    results = {}
    try:
        raw = yf.download(tickers, period="1y", group_by='ticker', progress=False, threads=True)
        for t in tickers:
            try:
                df = raw[t].copy() if len(tickers) > 1 else raw.copy()
                df.dropna(subset=['Close'], inplace=True)
                if len(df) < 50: continue
                
                # Tech Analysis
                df['SMA_50'] = ta.sma(df['Close'], length=50)
                df['SMA_200'] = ta.sma(df['Close'], length=200)
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                
                # Bollinger
                bb = ta.bbands(df['Close'], length=20, std=2)
                df['BB_Upper'] = bb['BBU_20_2.0']
                df['BB_Lower'] = bb['BBL_20_2.0']
                
                results[t] = df
            except: continue
    except: pass
    return results

# ==========================================
# 3. HELPER FUNCTIONS (PROJECTIONS & SCORING)
# ==========================================

def get_projections(df, days=30):
    """Draws future cones based on ATR."""
    last_p = df['Close'].iloc[-1]
    last_atr = df['ATR'].iloc[-1]
    
    dates = [df.index[-1] + timedelta(days=i) for i in range(1, days+1)]
    
    # Bull = Up trend + Volatility
    bull = [last_p + (last_atr * 0.5 * i) for i in range(1, days+1)]
    # Bear = Down trend - Volatility
    bear = [last_p - (last_atr * 0.5 * i) for i in range(1, days+1)]
    # Neutral = Flat
    base = [last_p for i in range(1, days+1)]
    
    return dates, bull, bear, base

def score_asset(df):
    """Weighted Score (-100 to +100)."""
    row = df.iloc[-1]
    score = 0
    
    # Trend
    if row['Close'] > row['SMA_200']: score += 25
    else: score -= 25
    
    # Momentum
    if row['RSI'] < 30: score += 25 # Oversold bounce
    elif row['RSI'] > 70: score -= 25 # Overbought drop
    
    # Mean Reversion
    if row['Close'] < row['BB_Lower']: score += 25
    if row['Close'] > row['BB_Upper']: score -= 25
    
    return score

# ==========================================
# 4. DASHBOARD LAYOUT
# ==========================================

st.title("🦅 Titan: Global Market Command")

with st.spinner("Connecting to Global Exchanges..."):
    market_df, idx_map, sec_map, mac_map = get_market_data()

# TABS
tab_macro, tab_indices, tab_compare, tab_scanner = st.tabs([
    "🌍 Macro & Regimes", 
    "📈 Global Indices", 
    "⚖️ Comparative Lab", 
    "🚀 Stock Scanner"
])

# --- TAB 1: MACRO ---
with tab_macro:
    st.subheader("Market Environment Status")
    
    # Regime Logic
    curr = market_df.iloc[-1]
    chg = ((curr - market_df.iloc[-20])/market_df.iloc[-20])*100
    
    inflationary = (chg['Oil (WTI)'] > 5) or (chg['Copper'] > 5)
    rates_high = curr['10Y Yield'] > 4.5
    
    if rates_high and inflationary:
        st.error("🛑 REGIME: STAGFLATION (High Rates + Rising Costs). Cash is safe.")
    elif not rates_high and not inflationary:
        st.success("✅ REGIME: GOLDILOCKS (Stable Rates + Low Inflation). Buy Growth.")
    else:
        st.warning("⚠️ REGIME: NEUTRAL / CHOPPY. Be selective.")

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("S&P 500", f"{curr['S&P 500']:.0f}")
    c2.metric("10Y Yield", f"{curr['10Y Yield']:.2f}%")
    c3.metric("VIX (Fear)", f"{curr['VIX']:.2f}")
    c4.metric("Oil (WTI)", f"${curr['Oil (WTI)']:.2f}")
    c5.metric("Gold", f"${curr['Gold']:.2f}")

# --- TAB 2: GLOBAL INDICES ---
with tab_indices:
    st.subheader("Major Global Indices")
    
    # Create normalized chart
    idx_cols = list(idx_map.keys())
    valid = [c for c in idx_cols if c in market_df.columns]
    
    norm_df = (market_df[valid] / market_df[valid].iloc[0]) * 100
    
    fig = go.Figure()
    for col in valid:
        # Highlight DAX and SPY
        width = 3 if "DAX" in col or "S&P" in col else 1
        fig.add_trace(go.Scatter(x=norm_df.index, y=norm_df[col], name=col, line=dict(width=width)))
    
    fig.update_layout(template="plotly_white", height=500, title="Relative Performance (Base=100)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: COMPARATIVE LAB ---
with tab_compare:
    c_sel, c_view = st.columns([1, 3])
    
    with c_sel:
        st.markdown("### Asset Selector")
        # Combine Sector and Macro lists
        options = list(sec_map.keys()) + list(mac_map.keys())
        selected = st.multiselect("Compare Assets:", options, default=["Technology (XLK)", "Energy (XLE)"])
        
    with c_view:
        if selected:
            fig2 = go.Figure()
            for s in selected:
                if s in market_df.columns:
                    # History
                    series = market_df[s]
                    fig2.add_trace(go.Scatter(x=series.index, y=series, name=s))
                    
                    # Simple Projection Line
                    x = np.arange(len(series))
                    z = np.polyfit(x, series.values, 1)
                    p = np.poly1d(z)
                    
                    fut_dates = [series.index[-1] + timedelta(days=i) for i in range(30)]
                    fut_y = p(np.arange(len(series), len(series)+30))
                    
                    fig2.add_trace(go.Scatter(x=fut_dates, y=fut_y, line=dict(dash='dot', width=1), showlegend=False))
            
            fig2.update_layout(template="plotly_white", height=500, title="Price Trends & Projections")
            st.plotly_chart(fig2, use_container_width=True)

# --- TAB 4: SCANNER ---
with tab_scanner:
    c_ctrl, c_out = st.columns([1, 3])
    
    with c_ctrl:
        st.markdown("### Scanner Settings")
        mkt = st.selectbox("Market Universe", ["US Tech (Nasdaq)", "Germany (DAX)", "High Beta / Crypto"])
        scan = st.button("RUN ANALYSIS", type="primary")
        
    if scan:
        with st.spinner("Analyzing Market Structure..."):
            res = run_scanner(mkt)
            
            scores = []
            for t, df in res.items():
                sc = score_asset(df)
                scores.append({"Ticker": t, "Score": sc, "Price": df['Close'].iloc[-1]})
            
            res_df = pd.DataFrame(scores)
            bulls = res_df[res_df['Score'] > 10].sort_values("Score", ascending=False)
            bears = res_df[res_df['Score'] < -10].sort_values("Score", ascending=True)
            
    with c_out:
        if scan and res:
            col1, col2 = st.columns(2)
            
            # BULLS
            with col1:
                st.markdown("### 🟢 Bullish Setups")
                if not bulls.empty:
                    st.dataframe(bulls.style.background_gradient(cmap="Greens"), hide_index=True)
                    
                    # Chart Top Bull
                    top = bulls.iloc[0]['Ticker']
                    st.caption(f"Projection: {top}")
                    df_p = res[top]
                    dates, b_line, s_line, n_line = get_projections(df_p)
                    
                    fig = go.Figure(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name="Price"))
                    fig.add_trace(go.Scatter(x=dates, y=b_line, line=dict(color='green', dash='dot'), name="Bull Path"))
                    fig.add_trace(go.Scatter(x=dates, y=s_line, line=dict(color='red', dash='dot'), name="Bear Path"))
                    fig.update_layout(template="plotly_white", height=350, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            # BEARS
            with col2:
                st.markdown("### 🔴 Bearish Setups")
                if not bears.empty:
                    st.dataframe(bears.style.background_gradient(cmap="Reds_r"), hide_index=True)
                    
                    # Chart Top Bear
                    top = bears.iloc[0]['Ticker']
                    st.caption(f"Projection: {top}")
                    df_p = res[top]
                    dates, b_line, s_line, n_line = get_projections(df_p)
                    
                    fig = go.Figure(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name="Price"))
                    fig.add_trace(go.Scatter(x=dates, y=b_line, line=dict(color='green', dash='dot'), name="Bull Path"))
                    fig.add_trace(go.Scatter(x=dates, y=s_line, line=dict(color='red', dash='dot'), name="Bear Path"))
                    fig.update_layout(template="plotly_white", height=350, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

