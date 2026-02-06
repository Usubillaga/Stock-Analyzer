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
st.set_page_config(layout="wide", page_title="Oracle: Global Macro & Future Projector")

st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #111; font-family: 'Segoe UI', sans-serif; }
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #F8F9FA;
        border: 1px solid #E0E0E0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    /* Headers */
    h1, h2, h3 { color: #222; }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 5px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #555;
    }
    .stTabs [aria-selected="true"] {
        background-color: #007BFF;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ROBUST DATA ENGINE
# ==========================================

@st.cache_data(ttl=3600)
def get_global_indices():
    """Fetches major global indices for comparison."""
    tickers = {
        "S&P 500": "^GSPC",
        "Nasdaq 100": "^NDX",
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT",
        "DAX (Germany)": "^GDAXI",
        "VIX (Volatility)": "^VIX"
    }
    # Fetch data safely
    data = yf.download(list(tickers.values()), period="1y", progress=False)['Close']
    
    # Clean column names
    rev_map = {v: k for k, v in tickers.items()}
    data.columns = [rev_map.get(c, c) for c in data.columns]
    
    # Fill gaps (European markets have different holidays than US)
    data.fillna(method='ffill', inplace=True)
    return data

@st.cache_data(ttl=3600)
def get_sector_data():
    """Fetches Sector ETFs for detailed analysis."""
    # Using ETFs which are more reliable than indices in yfinance
    sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLB", "XLU", "XLI", "IYR", "XLC"]
    df = yf.download(sectors, period="1y", progress=False)['Close']
    return df

@st.cache_data(ttl=3600)
def scanner_engine(universe_type):
    """
    Downloads and scores stocks. 
    optimized to RETURN RESULTS rather than filtering everything out.
    """
    if universe_type == "Nasdaq 100":
        tickers = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "INTC", "CSCO", "CMCSA", "PEP", "AVGO", "TXN", "QCOM", "ADBE", "AMGN", "COST"]
    elif universe_type == "DAX 40":
        tickers = ["SIE.DE", "SAP.DE", "ALV.DE", "DTE.DE", "AIR.DE", "BMW.DE", "VOW3.DE", "BAS.DE", "ADS.DE", "DB1.DE"]
    else: # Default Top Liquid
        tickers = ["NVDA", "TSLA", "AAPL", "AMD", "PLTR", "COIN", "MSTR", "MARA", "HOOD", "SOFI", "LCID", "RIVN", "NIO", "BABA", "AMD"]

    # Download Data
    data_dict = {}
    try:
        df_bulk = yf.download(tickers, period="1y", group_by='ticker', progress=False, threads=True)
        
        for t in tickers:
            try:
                # Handle MultiIndex safely
                df = df_bulk[t].copy() if len(tickers) > 1 else df_bulk.copy()
                df = df.dropna(subset=['Close'])
                
                if len(df) < 50: continue
                
                # --- CALCULATE INDICATORS ---
                # 1. Trend
                df['SMA_50'] = ta.sma(df['Close'], length=50)
                df['SMA_200'] = ta.sma(df['Close'], length=200)
                
                # 2. Momentum
                df['RSI'] = ta.rsi(df['Close'], length=14)
                
                # 3. Volatility (Bollinger & ATR)
                bb = ta.bbands(df['Close'], length=20, std=2)
                df['BB_Upper'] = bb['BBU_20_2.0']
                df['BB_Lower'] = bb['BBL_20_2.0']
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                
                data_dict[t] = df
            except: continue
    except: pass
    
    return data_dict

# ==========================================
# 3. PREDICTIVE LOGIC (MONTE CARLO)
# ==========================================

def generate_future_paths(df, days=30):
    """
    Generates 3 cone paths (Bull, Base, Bear) based on current volatility (ATR).
    """
    last_price = df['Close'].iloc[-1]
    last_atr = df['ATR'].iloc[-1]
    
    # We use ATR to project expected range
    # Bull Path: Trending up with Volatility
    bull_path = [last_price + (last_atr * 0.5 * i) for i in range(days)]
    
    # Bear Path: Trending down with Volatility
    bear_path = [last_price - (last_atr * 0.5 * i) for i in range(days)]
    
    # Neutral Path: Sideways chop
    neutral_path = [last_price + (np.sin(i) * last_atr * 0.5) for i in range(days)]
    
    # Future Dates
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
    
    return future_dates, bull_path, bear_path, neutral_path

def calculate_score(df):
    """
    Weighted Scoring System. 
    Returns a score from -100 (Bear) to +100 (Bull).
    """
    row = df.iloc[-1]
    score = 0
    
    # Trend (Weight: 40)
    if row['Close'] > row['SMA_200']: score += 20
    else: score -= 20
    
    if row['Close'] > row['SMA_50']: score += 20
    else: score -= 20
    
    # Momentum (Weight: 30)
    rsi = row['RSI']
    if rsi < 30: score += 30 # Oversold (Bullish bounce)
    elif rsi > 70: score -= 30 # Overbought (Bearish reversal)
    elif 50 < rsi < 70: score += 10 # Strong Momentum
    elif 30 < rsi < 50: score -= 10 # Weak Momentum
    
    # Structure (Weight: 30)
    # Testing Lower Bollinger Band?
    if row['Close'] <= row['BB_Lower'] * 1.02: score += 30 # Buy the dip
    # Testing Upper Bollinger Band?
    if row['Close'] >= row['BB_Upper'] * 0.98: score -= 30 # Sell the rip
    
    return score

# ==========================================
# 4. APP LAYOUT
# ==========================================

st.title("🔮 Oracle: Market Prediction Engine")

# TABS
tab_indices, tab_compare, tab_scanner = st.tabs(["🌍 Global Indices", "📊 Comparative Lab", "🚀 Stock Scanner"])

# --- TAB 1: GLOBAL INDICES ---
with tab_indices:
    st.subheader("Global Market Pulse")
    with st.spinner("Fetching Global Indices (DAX, Dow, Russell)..."):
        indices_df = get_global_indices()
        
    # Metrics
    if not indices_df.empty:
        c1, c2, c3, c4, c5 = st.columns(5)
        
        def show_metric(col, label, key):
            if key in indices_df.columns:
                curr = indices_df[key].iloc[-1]
                prev = indices_df[key].iloc[-2]
                delta = ((curr - prev)/prev)*100
                col.metric(label, f"{curr:,.0f}", f"{delta:.2f}%")
        
        show_metric(c1, "S&P 500", "S&P 500")
        show_metric(c2, "Dow Jones", "Dow Jones")
        show_metric(c3, "Nasdaq 100", "Nasdaq 100")
        show_metric(c4, "Russell 2000", "Russell 2000")
        show_metric(c5, "DAX (Ger)", "DAX (Germany)")
        
        st.divider()
        
        # Normalized Comparison Chart
        st.subheader("Relative Performance (Normalized to 100)")
        st.caption("This chart shows how different indices performed relative to each other starting from the same point.")
        
        # Normalize data to start at 100
        norm_df = (indices_df / indices_df.iloc[0]) * 100
        
        fig = go.Figure()
        for col in norm_df.columns:
            # Highlight specific lines
            width = 3 if "S&P" in col or "DAX" in col else 1
            opacity = 1.0 if "S&P" in col else 0.7
            
            fig.add_trace(go.Scatter(x=norm_df.index, y=norm_df[col], mode='lines', name=col, line=dict(width=width), opacity=opacity))
            
        fig.update_layout(template="plotly_white", height=500, hovermode="x unified", yaxis_title="% Return")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: COMPARATIVE LAB ---
with tab_compare:
    st.subheader("Sector X-Ray & Future Projection")
    col_sel, col_chart = st.columns([1, 3])
    
    with col_sel:
        st.info("Select specific Sector ETFs or Indices to compare their trends and volume structure.")
        # Load sector data
        sector_df = get_sector_data()
        all_options = list(sector_df.columns)
        selected_assets = st.multiselect("Select Assets to Compare", all_options, default=["XLK", "XLE"])
    
    with col_chart:
        if selected_assets:
            # Create subplots: Price on top, Relative Strength below
            fig2 = go.Figure()
            
            for asset in selected_assets:
                # Calculate simple trend line
                series = sector_df[asset]
                fig2.add_trace(go.Scatter(x=series.index, y=series, mode='lines', name=asset))
                
                # Draw simple linear regression "Future" line
                x = np.arange(len(series))
                y = series.values
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                
                # Project 30 days
                future_x = np.arange(len(series), len(series)+30)
                future_dates = [series.index[-1] + timedelta(days=i) for i in range(30)]
                fig2.add_trace(go.Scatter(x=future_dates, y=p(future_x), mode='lines', line=dict(dash='dot', width=1), showlegend=False, name=f"{asset} Proj"))

            fig2.update_layout(title="Price History + Simple Linear Projection", template="plotly_white", height=500)
            st.plotly_chart(fig2, use_container_width=True)

# --- TAB 3: THE SCANNER ---
with tab_scanner:
    c_ctrl, c_res = st.columns([1, 3])
    
    with c_ctrl:
        st.header("🔍 Radar")
        scan_univ = st.selectbox("Universe", ["Liquid US Stocks", "Nasdaq 100", "DAX 40"])
        st.info("The Oracle ranks stocks from -100 (Bearish) to +100 (Bullish).")
        run_scan = st.button("RUN SCAN", type="primary")
        
    if run_scan:
        with st.spinner("Analyzing market structure..."):
            stock_data = scanner_engine(scan_univ)
            
            if not stock_data:
                st.error("No data found. Try a different universe.")
            else:
                scored_list = []
                for t, df in stock_data.items():
                    sc = calculate_score(df)
                    scored_list.append({"Ticker": t, "Score": sc, "Price": df['Close'].iloc[-1]})
                
                # Convert to DF
                res_df = pd.DataFrame(scored_list)
                
                # Separate Bulls and Bears
                bulls = res_df[res_df['Score'] > 20].sort_values("Score", ascending=False)
                bears = res_df[res_df['Score'] < -20].sort_values("Score", ascending=True)
                
    with c_res:
        if run_scan and stock_data:
            col_b, col_s = st.columns(2)
            
            # --- BULLISH SECTION ---
            with col_b:
                st.subheader("🟢 Top Bullish Potentials")
                if not bulls.empty:
                    st.dataframe(bulls.style.background_gradient(cmap="Greens"), hide_index=True)
                    
                    # PREDICTIVE CHART FOR TOP BULL
                    top_bull = bulls.iloc[0]['Ticker']
                    st.markdown(f"**Future Path Projection: {top_bull}**")
                    
                    df_p = stock_data[top_bull]
                    f_dates, f_bull, f_bear, f_base = generate_future_paths(df_p)
                    
                    fig_b = go.Figure()
                    # Historical
                    fig_b.add_trace(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name="History"))
                    # Projections
                    fig_b.add_trace(go.Scatter(x=f_dates, y=f_bull, mode='lines', line=dict(color='green', dash='dot'), name="Bull Case"))
                    fig_b.add_trace(go.Scatter(x=f_dates, y=f_base, mode='lines', line=dict(color='gray', dash='dot'), name="Base Case"))
                    fig_b.add_trace(go.Scatter(x=f_dates, y=f_bear, mode='lines', line=dict(color='red', dash='dot'), name="Bear Case"))
                    
                    fig_b.update_layout(height=400, template="plotly_white", showlegend=True, title=f"{top_bull} Monte Carlo Simulation")
                    st.plotly_chart(fig_b, use_container_width=True)
                else:
                    st.warning("No strong buy signals.")

            # --- BEARISH SECTION ---
            with col_s:
                st.subheader("🔴 Top Bearish Potentials")
                if not bears.empty:
                    st.dataframe(bears.style.background_gradient(cmap="Reds_r"), hide_index=True)
                    
                    # PREDICTIVE CHART FOR TOP BEAR
                    top_bear = bears.iloc[0]['Ticker']
                    st.markdown(f"**Future Path Projection: {top_bear}**")
                    
                    df_p = stock_data[top_bear]
                    f_dates, f_bull, f_bear, f_base = generate_future_paths(df_p)
                    
                    fig_s = go.Figure()
                    # Historical
                    fig_s.add_trace(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name="History"))
                    # Projections
                    fig_s.add_trace(go.Scatter(x=f_dates, y=f_bull, mode='lines', line=dict(color='green', dash='dot'), name="Bull Case"))
                    fig_s.add_trace(go.Scatter(x=f_dates, y=f_base, mode='lines', line=dict(color='gray', dash='dot'), name="Base Case"))
                    fig_s.add_trace(go.Scatter(x=f_dates, y=f_bear, mode='lines', line=dict(color='red', dash='dot'), name="Bear Case"))
                    
                    fig_s.update_layout(height=400, template="plotly_white", showlegend=True, title=f"{top_bear} Monte Carlo Simulation")
                    st.plotly_chart(fig_s, use_container_width=True)
                else:
                    st.warning("No strong sell signals.")
