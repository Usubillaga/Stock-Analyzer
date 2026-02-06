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
    .regime-box { padding: 15px; border-radius: 5px; margin-bottom: 20px; color: white; font-weight: bold;}
    .regime-red { background-color: #D32F2F; }
    .regime-green { background-color: #388E3C; }
    .regime-orange { background-color: #F57C00; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE (BULLETPROOF)
# ==========================================

@st.cache_data(ttl=3600)
def get_market_data():
    """
    Fetches specific Indices, Sector ETFs, and Macro Assets.
    Includes explicit flattening logic to prevent data errors.
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
    
    # 2. Sector ETFs
    sectors = {
        "Tech (XLK)": "XLK",
        "Financials (XLF)": "XLF",
        "Energy (XLE)": "XLE",
        "Healthcare (XLV)": "XLV",
        "Discretionary (XLY)": "XLY",
        "Staples (XLP)": "XLP",
        "Materials (XLB)": "XLB",
        "Utilities (XLU)": "XLU",
        "Real Estate (XLRE)": "XLRE",
        "Industrials (XLI)": "XLI",
        "Comms (XLC)": "XLC"
    }
    
    # 3. Macro Commodities
    macro = {
        "Oil (USO)": "USO",    # ETF is more reliable than Future
        "Gold (GLD)": "GLD",
        "Copper (CPER)": "CPER",
        "Silver (SLV)": "SLV",
        "10Y Yield": "^TNX",
        "VIX": "^VIX"
    }

    # Combine all tickers
    all_tickers = {**indices, **sectors, **macro}
    symbol_list = list(all_tickers.values())
    
    try:
        # Download with auto_adjust to get proper prices
        data = yf.download(symbol_list, period="1y", progress=False, auto_adjust=True)
        
        # --- CRITICAL FIX: FLATTEN MULTI-INDEX ---
        # YFinance returns (Price, Ticker) structure. We want just Ticker -> Price.
        clean_data = pd.DataFrame()
        
        # Handle 'Close' column extraction
        if isinstance(data.columns, pd.MultiIndex):
            # Check if 'Close' is level 0 or level 1
            try:
                # Common case: Level 0 is 'Close', Level 1 is Ticker
                if 'Close' in data.columns.levels[0]:
                    clean_data = data['Close']
                # Alternate case: Level 0 is Ticker, Level 1 is 'Close'
                else:
                    for t in symbol_list:
                        if t in data.columns.levels[0]:
                             clean_data[t] = data[t]['Close']
            except:
                # Last resort loop
                for t in symbol_list:
                    try: clean_data[t] = data.xs(t, axis=1, level=1)['Close']
                    except: pass
        else:
            # Single ticker download structure
            clean_data = data['Close'] if 'Close' in data else data

        # Rename columns from Symbols (XLK) to Friendly Names (Tech (XLK))
        rev_map = {v: k for k, v in all_tickers.items()}
        clean_data.rename(columns=rev_map, inplace=True)
        
        # Fill gaps
        clean_data.fillna(method='ffill', inplace=True)
        
        return clean_data, indices, sectors, macro
        
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return pd.DataFrame(), indices, sectors, macro

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
        raw = yf.download(tickers, period="1y", group_by='ticker', progress=False, threads=True, auto_adjust=True)
        for t in tickers:
            try:
                # Handle MultiIndex extraction
                if isinstance(raw.columns, pd.MultiIndex):
                     df = raw[t].copy()
                elif len(tickers) == 1: 
                     df = raw.copy()
                else: continue # Skip if structure is weird

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
# 3. INTELLIGENCE ENGINES
# ==========================================

def macro_ai_analyst(df):
    """
    Restored AI Analyst: Evaluates Inflation, Rates, and Risk.
    """
    if df.empty: return {}, []
    
    curr = df.iloc[-1]
    prev = df.iloc[-22] # Monthly lookback
    chg = ((curr - prev) / prev) * 100
    
    narrative = []
    
    # 1. INFLATION CHECK
    inflation_assets = ['Oil (USO)', 'Copper (CPER)', 'Gold (GLD)']
    inf_count = sum([1 for a in inflation_assets if chg.get(a, 0) > 4.0])
    inflationary = inf_count >= 2
    
    # 2. RATE CHECK
    yield_val = curr.get('10Y Yield', 0)
    high_rates = yield_val > 4.5
    
    # 3. RISK APPETITE CHECK
    # Discretionary (Offense) vs Staples (Defense)
    risk_on = chg.get('Discretionary (XLY)', 0) > chg.get('Staples (XLP)', 0)
    
    # DETERMINE REGIME
    if high_rates and inflationary:
        status = "STAGFLATION (Red Alert)"
        color = "regime-red"
        narrative.append("🚨 **CRITICAL:** Inflation is rising (Commodities Up) while Rates are high.")
        narrative.append("👉 **Strategy:** Cash is King. Short Tech/Real Estate. Long Energy.")
    elif risk_on and not inflationary:
        status = "GOLDILOCKS (Bull Market)"
        color = "regime-green"
        narrative.append("🟢 **OPTIMAL:** Growth is leading (Discretionary > Staples) and inflation is tame.")
        narrative.append("👉 **Strategy:** Aggressive Longs in Tech (XLK) and Indices.")
    else:
        status = "NEUTRAL / CHOPPY"
        color = "regime-orange"
        narrative.append("⚠️ **CAUTION:** Mixed signals. Market is rotating defensively.")
        narrative.append("👉 **Strategy:** Stock picking only. Tight stops.")
        
    return {"status": status, "color": color, "yield": yield_val}, narrative

def get_projections(df, days=30):
    """Monte Carlo-style Cone Projections."""
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
    """Detailed Weighted Scoring."""
    row = df.iloc[-1]
    score = 0
    
    # Trend (40%)
    if row['Close'] > row['SMA_200']: score += 20
    else: score -= 20
    if row['Close'] > row['SMA_50']: score += 20
    else: score -= 20
    
    # Momentum (30%)
    if row['RSI'] < 30: score += 30 # Deep Oversold
    elif row['RSI'] > 70: score -= 30 # Overbought
    elif 50 < row['RSI'] < 70: score += 10 # Strong Momentum
    
    # Structure (30%)
    if row['Close'] < row['BB_Lower']: score += 30 # Mean Reversion Buy
    if row['Close'] > row['BB_Upper']: score -= 30 # Mean Reversion Sell
    
    return score

# ==========================================
# 4. DASHBOARD LAYOUT
# ==========================================

st.title("🦅 Titan: Global Market Command")

with st.spinner("Establishing Satellite Link to Global Exchanges..."):
    market_df, idx_map, sec_map, mac_map = get_market_data()

if market_df.empty:
    st.error("⚠️ CRITICAL ERROR: Data feed disconnected. Please refresh.")
    st.stop()

# Run AI Analyst
regime_data, ai_text = macro_ai_analyst(market_df)

# TABS
tab_macro, tab_indices, tab_compare, tab_scanner = st.tabs([
    "🌍 Macro Headquarters", 
    "📈 Global Indices", 
    "⚖️ Comparative Lab", 
    "🚀 Stock Scanner"
])

# --- TAB 1: MACRO HQ ---
with tab_macro:
    # 1. Regime Banner
    st.markdown(f"""
    <div class='regime-box {regime_data['color']}'>
        <h2>MARKET REGIME: {regime_data['status']}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. AI Narrative
    st.markdown("### 🧠 Macro AI Analyst")
    for line in ai_text:
        st.markdown(line)
    
    st.divider()

    # 3. Key Metrics
    curr = market_df.iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    
    def safe_m(col, key, is_cur=False):
        if key in curr:
            fmt = f"${curr[key]:.2f}" if is_cur else f"{curr[key]:.2f}"
            col.metric(key, fmt)
            
    safe_m(c1, "S&P 500")
    safe_m(c2, "10Y Yield", False)
    safe_m(c3, "VIX", False)
    safe_m(c4, "Oil (USO)", True)
    safe_m(c5, "Gold (GLD)", True)

# --- TAB 2: GLOBAL INDICES ---
with tab_indices:
    st.subheader("Global Equity Performance (Normalized)")
    
    # Filter only Indices
    idx_cols = list(idx_map.keys())
    valid_idx = [c for c in idx_cols if c in market_df.columns]
    
    if valid_idx:
        # Normalize to 100
        norm_df = (market_df[valid_idx] / market_df[valid_idx].iloc[0]) * 100
        
        fig = go.Figure()
        for col in valid_idx:
            # Highlight DAX and SPY
            width = 4 if "S&P" in col else 3 if "DAX" in col else 1
            opacity = 1.0 if "S&P" in col else 0.7
            fig.add_trace(go.Scatter(x=norm_df.index, y=norm_df[col], name=col, line=dict(width=width), opacity=opacity))
        
        fig.update_layout(template="plotly_white", height=500, title="Relative Strength (Base=100)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: COMPARATIVE LAB ---
with tab_compare:
    st.markdown("### 🔬 Asset X-Ray")
    
    c_sel, c_chart = st.columns([1, 3])
    with c_sel:
        st.info("Overlay Sectors, Commodities, and Indices to spot correlations.")
        # Combine lists
        all_opts = list(sec_map.keys()) + list(mac_map.keys()) + list(idx_map.keys())
        # Default choices
        picks = st.multiselect("Select Assets:", all_opts, default=["Tech (XLK)", "Energy (XLE)"])
        
    with c_chart:
        if picks:
            fig2 = go.Figure()
            for p in picks:
                if p in market_df.columns:
                    # Plot History
                    series = market_df[p]
                    fig2.add_trace(go.Scatter(x=series.index, y=series, name=p))
                    
                    # Simple Trend Projection
                    x = np.arange(len(series))
                    z = np.polyfit(x, series.values, 1) # Linear fit
                    poly = np.poly1d(z)
                    
                    # Future Dates
                    fut_x = np.arange(len(series), len(series)+30)
                    fut_dates = [series.index[-1] + timedelta(days=i) for i in range(30)]
                    
                    fig2.add_trace(go.Scatter(x=fut_dates, y=poly(fut_x), line=dict(dash='dot', width=2), showlegend=False))
            
            fig2.update_layout(template="plotly_white", height=500, title="Price History + Trend Projection")
            st.plotly_chart(fig2, use_container_width=True)

# --- TAB 4: STOCK SCANNER ---
with tab_scanner:
    c_ctrl, c_view = st.columns([1, 3])
    
    with c_ctrl:
        st.markdown("### 📡 Scanner Controls")
        univ = st.selectbox("Universe", ["US Tech (Nasdaq)", "Germany (DAX)", "High Beta / Crypto"])
        do_scan = st.button("🚀 INITIATE SCAN", type="primary")
        st.caption("Scans for Trend, Momentum & Structure.")
        
    if do_scan:
        with st.spinner("Analyzing Market Structure..."):
            res = run_scanner(univ)
            
            scores = []
            if res:
                for t, df in res.items():
                    sc = score_asset(df)
                    last_p = df['Close'].iloc[-1]
                    scores.append({"Ticker": t, "Score": sc, "Price": last_p})
            
            # Create DF safely
            if scores:
                res_df = pd.DataFrame(scores)
                bulls = res_df[res_df['Score'] > 10].sort_values("Score", ascending=False)
                bears = res_df[res_df['Score'] < -10].sort_values("Score", ascending=True)
            else:
                bulls = pd.DataFrame()
                bears = pd.DataFrame()
                
    with c_view:
        if do_scan and scores:
            col_b, col_s = st.columns(2)
            
            # --- BULLS ---
            with col_b:
                st.success("🟢 Bullish Candidates (Buy)")
                if not bulls.empty:
                    st.dataframe(bulls.style.background_gradient(cmap="Greens"), hide_index=True)
                    
                    # Chart Top Bull
                    top = bulls.iloc[0]['Ticker']
                    st.caption(f"Projection: {top}")
                    df_p = res[top]
                    d, b_l, s_l, n_l = get_projections(df_p)
                    
                    fig = go.Figure(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name='Price'))
                    fig.add_trace(go.Scatter(x=d, y=b_l, line=dict(color='green', dash='dot'), name='Bull Case'))
                    fig.add_trace(go.Scatter(x=d, y=s_l, line=dict(color='red', dash='dot'), name='Bear Case'))
                    fig.update_layout(template="plotly_white", height=350, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            # --- BEARS ---
            with col_s:
                st.error("🔴 Bearish Candidates (Sell)")
                if not bears.empty:
                    st.dataframe(bears.style.background_gradient(cmap="Reds_r"), hide_index=True)
                    
                    # Chart Top Bear
                    top = bears.iloc[0]['Ticker']
                    st.caption(f"Projection: {top}")
                    df_p = res[top]
                    d, b_l, s_l, n_l = get_projections(df_p)
                    
                    fig = go.Figure(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name='Price'))
                    fig.add_trace(go.Scatter(x=d, y=b_l, line=dict(color='green', dash='dot'), name='Bull Case'))
                    fig.add_trace(go.Scatter(x=d, y=s_l, line=dict(color='red', dash='dot'), name='Bear Case'))
                    fig.update_layout(template="plotly_white", height=350, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
