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
st.set_page_config(layout="wide", page_title="Titan: Platinum Market Analyst")

st.markdown("""
<style>
    /* Professional Theme */
    .stApp { background-color: #F8F9FA; font-family: 'Segoe UI', sans-serif; color: #212529; }
    
    /* Cards & Metrics */
    div[data-testid="stMetric"] { background-color: #FFFFFF; border: 1px solid #DEE2E6; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .titan-card { background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #ddd; margin-bottom: 20px; }
    
    /* Regime Badges */
    .regime-bull { border-left: 10px solid #2E7D32; background-color: #E8F5E9; padding: 20px; font-weight: bold; border-radius: 5px; color: #1B5E20; }
    .regime-bear { border-left: 10px solid #C62828; background-color: #FFEBEE; padding: 20px; font-weight: bold; border-radius: 5px; color: #B71C1C; }
    .regime-warn { border-left: 10px solid #EF6C00; background-color: #FFF3E0; padding: 20px; font-weight: bold; border-radius: 5px; color: #E65100; }
    
    /* AI Analyst Box */
    .ai-box { background-color: #E3F2FD; padding: 15px; border-radius: 8px; border-left: 5px solid #1976D2; margin-bottom: 20px; color: #0D47A1; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE (BULLETPROOF)
# ==========================================

def flatten_yfinance(df):
    """
    CRITICAL FIX: Flattens MultiIndex columns from yfinance (Price, Ticker) -> (Ticker).
    """
    if df.empty: return df
    
    # If MultiIndex (e.g., levels: Price, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # Check if 'Close' is in the top level (new yfinance)
            if 'Close' in df.columns.levels[0]:
                return df['Close']
            # Check if 'Close' is in the second level (old yfinance)
            elif 'Close' in df.columns.levels[1]:
                return df.xs('Close', axis=1, level=1)
        except:
            pass
    return df

@st.cache_data(ttl=3600)
def fetch_long_term_data():
    """Fetches 10 YEARS of Macro Data."""
    tickers = {
        "S&P 500": "SPY", "Nasdaq 100": "QQQ", "Russell 2000": "IWM", 
        "DAX (Germany)": "^GDAXI", "FTSE 100": "^FTSE", "Nikkei 225": "^N225",
        "S&P Equal Weight": "RSP",
        "VIX": "^VIX", "10Y Yield": "^TNX", "USD Index": "DX-Y.NYB",
        "Oil": "USO", "Copper": "CPER", "Lumber": "WOOD", "Gold": "GLD", "Silver": "SLV",
        "Tech": "XLK", "Energy": "XLE", "Financials": "XLF", "Staples": "XLP",
        "Discretionary": "XLY", "Utilities": "XLU", "Real Estate": "XLRE",
        "Materials": "XLB", "Industrials": "XLI", "Healthcare": "XLV"
    }
    
    try:
        raw = yf.download(list(tickers.values()), period="10y", progress=False, auto_adjust=True)
        # Flatten
        df = flatten_yfinance(raw)
        # Rename to friendly names
        rev_map = {v: k for k, v in tickers.items()}
        df.rename(columns=rev_map, inplace=True)
        df.fillna(method='ffill', inplace=True)
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_scanner_batch(universe):
    """
    Fetches stock data for the scanner. 
    INCLUDES FUNDAMENTALS (P/E, Market Cap).
    """
    if universe == "US Tech / Growth":
        ticks = ["NVDA", "AMD", "TSLA", "PLTR", "COIN", "UBER", "AMZN", "GOOGL", "META", "MSFT", "AAPL", "NET", "CRWD", "RIVN"]
    elif universe == "Global Macro (ETFs)":
        ticks = ["SPY", "QQQ", "IWM", "EEM", "GLD", "SLV", "USO", "TLT", "HYG", "FXI", "EWZ"]
    else: # DAX / Europe
        ticks = ["SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BMW.DE", "ADS.DE", "AIR.DE", "DHL.DE", "DB1.DE", "RWE.DE"]

    # 1. Fetch Price History
    data_dict = {}
    try:
        raw = yf.download(ticks, period="2y", group_by='ticker', progress=False, threads=True, auto_adjust=True)
        
        # 2. Fetch Fundamentals (Snapshot)
        fundamentals = {}
        for t in ticks:
            try:
                info = yf.Ticker(t).info
                fundamentals[t] = {
                    "PE": info.get('trailingPE', 0),
                    "Fwd PE": info.get('forwardPE', 0),
                    "Mkt Cap": info.get('marketCap', 0)
                }
            except: 
                fundamentals[t] = {"PE": 0, "Fwd PE": 0, "Mkt Cap": 0}

        for t in ticks:
            try:
                # Robust Extraction logic
                if isinstance(raw.columns, pd.MultiIndex):
                    if t in raw.columns.levels[0]: df = raw[t].copy()
                    else: continue
                elif len(ticks) == 1: df = raw.copy()
                else: continue

                # Clean
                df.dropna(subset=['Close'], inplace=True)
                if len(df) < 100: continue
                
                # Indicators
                df['SMA_50'] = ta.sma(df['Close'], length=50)
                df['SMA_200'] = ta.sma(df['Close'], length=200)
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                bb = ta.bbands(df['Close'], length=20, std=2)
                if bb is not None:
                    df['BB_Upper'] = bb['BBU_20_2.0']
                    df['BB_Lower'] = bb['BBL_20_2.0']
                
                # Attach Fundamentals to DataFrame as single values (for later use)
                df.attrs['fundamentals'] = fundamentals[t]
                
                data_dict[t] = df
            except: continue
            
        return data_dict
    except: return {}

@st.cache_data(ttl=3600)
def fetch_index_composition(index_name):
    """Fetches Top Holdings live weights."""
    if index_name == "S&P 500":
        tickers = ["MSFT", "AAPL", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "TSLA", "LLY", "AVGO"]
    elif index_name == "Nasdaq 100":
        tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "AVGO", "META", "TSLA", "GOOGL", "COST", "AMD"]
    elif index_name == "DAX (Germany)":
        tickers = ["SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BMW.DE", "AIR.DE", "ADS.DE", "DHL.DE", "BAS.DE", "MU.DE"]
    else: return pd.DataFrame()

    data_list = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            mc = stock.info.get('marketCap', 0)
            if mc > 0: data_list.append({"Ticker": t, "Market Cap": mc})
        except: continue
    
    df = pd.DataFrame(data_list)
    if not df.empty:
        df['Weight (%)'] = (df['Market Cap'] / df['Market Cap'].sum()) * 100
        df = df.sort_values("Weight (%)", ascending=False)
    return df

# ==========================================
# 3. ANALYTIC ENGINES (SMART LOGIC)
# ==========================================

def macro_ai_analyst(df, gdp, unemp, cpi):
    if df.empty: return {}, [], {}
    
    curr = df.iloc[-1]
    # 1 Month Momentum
    idx = -22 if len(df) > 22 else 0
    
    # Calculate returns for Bar Chart
    perf = ((curr - df.iloc[idx]) / df.iloc[idx]) * 100
    
    advice = []
    score = 0
    
    # 1. VIX SENSITIVITY
    vix = curr.get('VIX', 15)
    if vix > 28:
        advice.append(f"🚨 **CRASH MODE (VIX {vix:.1f}):** Extreme Fear. Hedging required.")
        score -= 3
    elif vix > 20:
        advice.append(f"⚠️ **High Stress (VIX {vix:.1f}):** Volatility is elevated. Caution.")
        score -= 1
    else:
        advice.append("✅ **Calm Seas:** VIX is low (<20). Supports bullish trend.")
        score += 1

    # 2. ECONOMIC INPUTS
    if gdp < 1.0 or unemp > 5.0:
        advice.append(f"⚠️ **Recession Risk:** GDP {gdp}% / Unemployment {unemp}%.")
        score -= 2
    
    # 3. INFLATION & GOLD
    gold_perf = perf.get('Gold', 0)
    oil_perf = perf.get('Oil', 0)
    
    if cpi > 4.0 or gold_perf > 5.0:
        advice.append(f"🔥 **Inflation Hedge:** Gold is rallying (+{gold_perf:.1f}%). Inflation is sticky.")
        score -= 1
    else:
        advice.append("✅ **Inflation Tame:** Commodities stable.")
        score += 1

    # VERDICT
    if score >= 1: regime = {"status": "BULLISH (RISK ON)", "css": "regime-bull"}
    elif score <= -2: regime = {"status": "BEARISH (DEFENSIVE)", "css": "regime-bear"}
    else: regime = {"status": "NEUTRAL / CHOPPY", "css": "regime-warn"}
    
    return regime, advice, perf

def sector_ai_analyst(ret_series):
    """Analyzes Sector Rotation (Risk On/Off)."""
    if ret_series.empty: return "No data."
    
    tech = ret_series.get("Tech", 0)
    staples = ret_series.get("Staples", 0)
    
    if tech > staples: return "🐂 **Risk-On:** Technology is leading Defensives."
    else: return "🐻 **Risk-Off:** Investors hiding in Staples."

def analyze_10y_cycles(df):
    """Dedicated Analyst for the 10Y Chart."""
    if 'VIX' not in df: return []
    avg_vix = df['VIX'].mean()
    curr_vix = df['VIX'].iloc[-1]
    
    insights = []
    if curr_vix > 30:
        insights.append("• **Cycle Status:** CRASH MODE. Historically, VIX > 30 marks panic bottoms.")
    elif curr_vix < 12:
        insights.append("• **Cycle Status:** COMPLACENCY. Markets are overly calm.")
    else:
        insights.append(f"• **Cycle Status:** Normal Volatility (Current: {curr_vix:.1f}).")
    return insights

def calculate_yearly_matrix(df):
    """Annual Returns Heatmap."""
    y_df = df.resample('Y').last().pct_change() * 100
    y_df = y_df.dropna()
    y_df.index = y_df.index.year
    return y_df.T

def score_stock(df):
    row = df.iloc[-1]
    s = 0
    # Trend
    if row['Close'] > row['SMA_200']: s += 30
    else: s -= 30
    # Momentum
    if row['RSI'] < 30: s += 30 
    elif row['RSI'] > 75: s -= 30
    # Volatility Squeeze
    if row['Close'] < row['BB_Lower']: s += 20
    if row['Close'] > row['BB_Upper']: s -= 20
    return s

def get_monte_carlo(df, days=90):
    last = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    dates = [df.index[-1] + timedelta(days=i) for i in range(1, days+1)]
    bull = [last + (atr * 0.8 * (i**0.6)) for i in range(1, days+1)]
    bear = [last - (atr * 0.8 * (i**0.6)) for i in range(1, days+1)]
    return dates, bull, bear

# ==========================================
# 4. APP UI LAYOUT
# ==========================================

st.title("🦅 Titan: Omni-Market Analyst")

# SIDEBAR (Complete)
st.sidebar.header("1. Economic Inputs")
in_gdp = st.sidebar.number_input("GDP Growth (%)", 2.5)
in_unemp = st.sidebar.number_input("Unemployment (%)", 3.8)
in_cpi = st.sidebar.number_input("CPI Inflation (%)", 3.2)
st.sidebar.markdown("---")
st.sidebar.header("2. Scanner Settings")
univ = st.sidebar.selectbox("Universe", ["US Tech / Growth", "Global Macro (ETFs)", "DAX / Europe"])

# LOAD DATA
with st.spinner("Analyzing 10 Years of History..."):
    macro_df = fetch_long_term_data()
    regime, advice, macro_perf = macro_ai_analyst(macro_df, in_gdp, in_unemp, in_cpi)

# TABS
t_hq, t_deep, t_sec, t_idx, t_scan = st.tabs(["🌍 Macro HQ", "📉 10Y Deep Dive", "📊 Sector Balken", "🍰 Composition", "🚀 Stock Scanner"])

# TAB 1: MACRO HQ
with t_hq:
    if not macro_df.empty:
        st.markdown(f"<div class='{regime['css']}'><h2>REGIME: {regime['status']}</h2></div>", unsafe_allow_html=True)
        st.markdown("### 🧠 Sensitive AI Analyst")
        for a in advice: st.info(a)
        
        st.divider()
        
        # MACRO BAR CHART (Balken)
        st.subheader("Macro Asset Performance (1 Month)")
        macro_assets = ["Gold", "Silver", "Oil", "Copper", "10Y Yield", "USD Index", "S&P 500"]
        valid_m = [c for c in macro_assets if c in macro_perf]
        if valid_m:
            m_data = macro_perf[valid_m].sort_values()
            fig_m = px.bar(
                x=m_data.values, y=m_data.index, orientation='h',
                title="Commodities vs Yields vs Indices",
                color=m_data.values, color_continuous_scale="RdYlGn",
                labels={'x': 'Return (%)', 'y': 'Asset'}
            )
            st.plotly_chart(fig_m, use_container_width=True)

        curr = macro_df.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("VIX", f"{curr.get('VIX',0):.2f}")
        c2.metric("10Y Yield", f"{curr.get('10Y Yield',0):.2f}%")
        c3.metric("Gold", f"${curr.get('Gold',0):.2f}")
        c4.metric("Oil", f"${curr.get('Oil',0):.2f}")
    else: st.error("Data Error.")

# TAB 2: 10Y DEEP DIVE
with t_deep:
    st.subheader("VIX vs S&P 500 (10 Years)")
    if not macro_df.empty:
        # AI Insight
        insights = analyze_10y_cycles(macro_df)
        with st.expander("🧠 AI Historical Cycle Analysis", expanded=True):
            for i in insights: st.write(i)
            
        # VIX Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=macro_df.index, y=macro_df['S&P 500'], name="S&P 500", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=macro_df.index, y=macro_df['VIX'], name="VIX", line=dict(color='gray', width=1), yaxis="y2"))
        danger = macro_df[macro_df['VIX'] > 30]
        fig.add_trace(go.Scatter(x=danger.index, y=danger['S&P 500'], mode='markers', marker=dict(color='red', symbol='x'), name="Crash Zone"))
        fig.update_layout(height=500, yaxis2=dict(overlaying='y', side='right'), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # Annual Returns Heatmap
        st.subheader("Annual Returns Heatmap (Restored)")
        y_mat = calculate_yearly_matrix(macro_df)
        fig_h = px.imshow(y_mat, text_auto=".1f", color_continuous_scale="RdYlGn", aspect="auto")
        st.plotly_chart(fig_h, use_container_width=True)

# TAB 3: BALKEN COMPARISON (Time Selectable)
with t_sec:
    st.subheader("Performance Rankings (Balken)")
    timeframe = st.selectbox("Select Timeframe:", ["1 Month", "3 Months", "6 Months", "1 Year"], index=0)
    
    if not macro_df.empty:
        lb_map = {"1 Month": 22, "3 Months": 66, "6 Months": 126, "1 Year": 252}
        days = lb_map[timeframe]
        idx = -days if len(macro_df) > days else 0
        ret = ((macro_df.iloc[-1] - macro_df.iloc[idx]) / macro_df.iloc[idx]) * 100
        
        # AI Sector Analyst
        st.markdown(f"<div class='ai-box'><b>🧠 Sector AI Analyst:</b> {sector_ai_analyst(ret)}</div>", unsafe_allow_html=True)
        
        # Plot Sectors
        secs = ["Tech", "Energy", "Financials", "Healthcare", "Staples", "Discretionary", "Real Estate", "Utilities", "Materials"]
        valid_s = [c for c in secs if c in ret]
        if valid_s:
            s_data = ret[valid_s].sort_values()
            fig_s = px.bar(x=s_data.values, y=s_data.index, orientation='h', title=f"Sectors ({timeframe})", color=s_data.values, color_continuous_scale="RdYlGn")
            st.plotly_chart(fig_s, use_container_width=True)

# TAB 4: INDEX COMPOSITION
with t_holdings:
    st.subheader("Index Heavyweights (Live Weight %)")
    idx_choice = st.radio("Select Index:", ["S&P 500", "Nasdaq 100", "DAX (Germany)"], horizontal=True)
    if st.button("Get Weights"):
        with st.spinner("Calculating..."):
            comp_df = fetch_index_composition(idx_choice)
            if not comp_df.empty:
                c1, c2 = st.columns([1, 1])
                c1.dataframe(comp_df, hide_index=True)
                fig_pie = px.pie(comp_df, values='Market Cap', names='Ticker', title=f'Top Constituents')
                c2.plotly_chart(fig_pie, use_container_width=True)

# TAB 5: SCANNER (Fixed + Fundamentals)
with t_scan:
    if st.button("RUN LIVE SCAN", type="primary"):
        with st.spinner(f"Scanning {univ}..."):
            stock_data = fetch_scanner_batch(univ)
            
            res = []
            if stock_data:
                for t, df in stock_data.items():
                    sc = score_stock(df)
                    # Fundamentals
                    funds = df.attrs.get('fundamentals', {})
                    pe = funds.get('PE', 0)
                    
                    res.append({
                        "Ticker": t, 
                        "Score": sc, 
                        "Price": df['Close'].iloc[-1],
                        "P/E Ratio": pe
                    })
            
            if res:
                df_r = pd.DataFrame(res)
                bulls = df_r.sort_values("Score", ascending=False).head(5)
                bears = df_r.sort_values("Score", ascending=True).head(5)
                
                c1, c2 = st.columns(2)
                
                # BULLS
                with c1:
                    st.success("🟢 Top Bulls")
                    st.dataframe(bulls, hide_index=True)
                    if not bulls.empty:
                        top = bulls.iloc[0]['Ticker']
                        st.caption(f"Monte Carlo: {top}")
                        df_p = stock_data[top]
                        d, b, s = get_monte_carlo(df_p, days=90)
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
                        st.caption(f"Monte Carlo: {top}")
                        df_p = stock_data[top]
                        d, b, s = get_monte_carlo(df_p, days=90)
                        fig = go.Figure(go.Candlestick(x=df_p.tail(200).index, open=df_p['Open'].tail(200), high=df_p['High'].tail(200), low=df_p['Low'].tail(200), close=df_p['Close'].tail(200)))
                        fig.add_trace(go.Scatter(x=d, y=b, line=dict(color='green', dash='dot'), name='Bull'))
                        fig.add_trace(go.Scatter(x=d, y=s, line=dict(color='red', dash='dot'), name='Bear'))
                        fig.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Scanner returned no data. Check internet or try a different universe.")

