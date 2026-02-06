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
    .stApp { background-color: #F0F2F5; font-family: 'Roboto', sans-serif; }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #D1D9E6;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Regime Badges & Advice Box */
    .regime-box { padding: 20px; border-radius: 8px; color: white; margin-bottom: 20px; text-align: center; }
    .regime-green { background-color: #2E7D32; } /* Dark Green */
    .regime-red { background-color: #C62828; }   /* Dark Red */
    .regime-orange { background-color: #EF6C00; } /* Dark Orange */
    
    .advice-card { background-color: #fff; padding: 15px; border-radius: 8px; border-left: 6px solid #2962FF; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE (ROBUST & LONG TERM)
# ==========================================

@st.cache_data(ttl=3600)
def fetch_long_term_data():
    """Fetches 10 YEARS of data for Deep Dive & Macro."""
    tickers = {
        # Indices
        "S&P 500": "SPY", "Nasdaq 100": "QQQ", "Russell 2000": "IWM", 
        "DAX (Germany)": "^GDAXI", "FTSE 100": "^FTSE",
        # Breadth
        "S&P Equal Weight": "RSP",
        # Volatility
        "VIX": "^VIX", "10Y Yield": "^TNX", "USD Index": "DX-Y.NYB",
        # Commodities
        "Oil": "USO", "Copper": "CPER", "Lumber": "WOOD", "Gold": "GLD", "Silver": "SLV",
        # Sectors
        "Tech": "XLK", "Energy": "XLE", "Financials": "XLF", "Staples": "XLP",
        "Discretionary": "XLY", "Utilities": "XLU", "Real Estate": "XLRE",
        "Materials": "XLB", "Industrials": "XLI", "Healthcare": "XLV"
    }
    
    try:
        data = yf.download(list(tickers.values()), period="10y", progress=False, auto_adjust=True)
        
        clean = pd.DataFrame()
        # Robust Flattening
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
    """Fetches stock data. SAFE & LIQUID lists only."""
    if universe == "US Tech / Growth":
        ticks = ["NVDA", "AMD", "TSLA", "PLTR", "COIN", "UBER", "AMZN", "GOOGL", "META", "MSFT", "AAPL", "NET", "CRWD", "RIVN", "MARA", "DKNG"]
    elif universe == "Global Macro (ETFs)":
        ticks = ["SPY", "QQQ", "IWM", "EEM", "GLD", "SLV", "USO", "TLT", "HYG", "FXI", "EWZ", "XLE", "XLK"]
    else: # DAX / Europe
        ticks = ["SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BMW.DE", "ADS.DE", "AIR.DE", "DHL.DE", "DB1.DE", "RWE.DE", "BAYN.DE"]

    data = {}
    try:
        # Download
        raw = yf.download(ticks, period="2y", group_by='ticker', progress=False, threads=True, auto_adjust=True)
        
        for t in ticks:
            try:
                # 1. Extract DataFrame safely
                if isinstance(raw.columns, pd.MultiIndex):
                    if t in raw.columns.levels[0]: 
                        df = raw[t].copy()
                    else: 
                        continue # Skip bad ticker
                elif len(ticks) == 1: 
                    df = raw.copy()
                else: 
                    continue

                # 2. Check Data Quality
                df.dropna(subset=['Close'], inplace=True)
                if len(df) < 100: continue # Need enough history for indicators
                
                # 3. Calculate Indicators
                df['SMA_50'] = ta.sma(df['Close'], length=50)
                df['SMA_200'] = ta.sma(df['Close'], length=200)
                df['RSI'] = ta.rsi(df['Close'], length=14)
                df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                bb = ta.bbands(df['Close'], length=20, std=2)
                
                if bb is not None:
                    df['BB_Upper'] = bb['BBU_20_2.0']
                    df['BB_Lower'] = bb['BBL_20_2.0']
                
                data[t] = df
            except Exception: 
                continue # Skip individual failure
    except Exception: 
        pass
    return data

@st.cache_data(ttl=3600)
def fetch_index_composition(index_name):
    """Fetches Top Holdings live."""
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
# 3. ANALYTIC ENGINES (ENHANCED MACRO)
# ==========================================

def macro_ai_analyst(df):
    """
    ADVANCED MACRO LOGIC:
    Analyzes: Trend, Volatility, Inflation (Gold/Oil), and Breadth.
    """
    if df.empty: return {}, [], []
    
    curr = df.iloc[-1]
    
    # Percent Changes (1 Month)
    lookback = -22 if len(df) > 22 else 0
    chg = ((curr - df.iloc[lookback]) / df.iloc[lookback]) * 100
    
    score = 0
    status = ""
    css = ""
    trade_call = ""
    
    insights = []
    
    # 1. MARKET TREND & MOMENTUM
    spy_trend = "UP" if curr['S&P 500'] > df['S&P 500'].rolling(200).mean().iloc[-1] else "DOWN"
    if spy_trend == "UP": 
        score += 2
        insights.append("📈 **Primary Trend:** Bullish (Price > 200 SMA).")
    else: 
        score -= 2
        insights.append("📉 **Primary Trend:** Bearish (Price < 200 SMA). Caution.")

    # 2. VOLATILITY (VIX)
    vix = curr.get('VIX', 15)
    if vix > 25:
        score -= 3
        insights.append(f"🚨 **High Fear (VIX {vix:.0f}):** Volatility is dangerous. Reduce position sizes.")
    elif vix < 18:
        score += 1
        insights.append(f"✅ **Low Fear (VIX {vix:.0f}):** Volatility supports rallying.")

    # 3. INFLATION & COMMODITIES (Gold/Silver vs Oil)
    # Are precious metals (Fear/Debasement) beating Industrial metals (Growth)?
    gold_perf = chg.get('Gold', 0)
    copper_perf = chg.get('Copper', 0)
    
    if gold_perf > copper_perf + 5.0:
        score -= 1
        insights.append("🛡️ **Defensive Rotation:** Gold is outperforming Copper. Investors seek safety.")
    elif copper_perf > gold_perf:
        score += 1
        insights.append("🏗️ **Cyclical Strength:** Copper (Economy) is outperforming Gold (Fear).")
        
    # 4. YIELDS
    tnx = curr.get('10Y Yield', 4.0)
    if tnx > 4.5:
        score -= 1
        insights.append("💸 **Rate Headwind:** 10Y Yield > 4.5% pressures tech valuations.")

    # --- DECISION ENGINE ---
    if score >= 3:
        status = "STRONG BULL MARKET"
        css = "regime-green"
        trade_call = "AGGRESSIVE LONG: Focus on Tech (XLK) and Growth."
    elif 0 <= score < 3:
        status = "NEUTRAL / CHOPPY"
        css = "regime-orange"
        trade_call = "CAUTION: Market lacks direction. Hold Cash or Quality Staples."
    else:
        status = "BEAR MARKET / CORRECTION"
        css = "regime-red"
        trade_call = "DEFENSIVE: Move to Cash, Gold, or Short Indices."
        
    return {"status": status, "css": css, "call": trade_call}, insights, chg

def score_stock(df):
    """Rank stocks from -100 (Bear) to +100 (Bull)"""
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
# 4. APP UI
# ==========================================

st.title("🦅 Titan: Omni-Market Analyst")

# SIDEBAR
st.sidebar.header("Scanner Configuration")
univ = st.sidebar.selectbox("Market Universe", ["US Tech / Growth", "Global Macro (ETFs)", "DAX / Europe"])
st.sidebar.info("Note: Macro Data is fetched automatically based on live markets.")

# LOAD DATA
with st.spinner("Analyzing Global Markets..."):
    macro_df = fetch_long_term_data()
    regime, advice, perf_data = macro_ai_analyst(macro_df)

# TABS
t_hq, t_deep, t_comp, t_holdings, t_scan = st.tabs(["🌍 Macro HQ", "📉 10Y Deep Dive", "📊 Sector Balken", "🍰 Index Composition", "🚀 Stock Scanner"])

# TAB 1: MACRO HQ (Upgraded with Bar Chart & Gold/Silver)
with t_hq:
    if not macro_df.empty:
        # Regime Banner
        st.markdown(f"<div class='regime-box {regime['css']}'><h2>{regime['status']}</h2><h3>RECOMMENDATION: {regime['call']}</h3></div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown("### 🧠 Macro AI Analyst")
            for a in advice: st.markdown(f"<div class='advice-card'>{a}</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("### 🌡️ Asset Performance (1 Month)")
            # Create Macro Bar Chart
            macro_assets = ["S&P 500", "Gold", "Silver", "Oil", "Copper", "10Y Yield", "USD Index"]
            valid_m = [c for c in macro_assets if c in perf_data]
            if valid_m:
                m_data = perf_data[valid_m].sort_values()
                fig_m = px.bar(
                    x=m_data.values, y=m_data.index, orientation='h', 
                    title="Macro Winners vs Losers (1 Month %)",
                    color=m_data.values, color_continuous_scale="RdYlGn",
                    labels={'x': 'Return (%)', 'y': 'Asset'}
                )
                st.plotly_chart(fig_m, use_container_width=True)
            
            # Key Metrics
            curr = macro_df.iloc[-1]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Gold", f"${curr.get('Gold',0):.2f}")
            m2.metric("Silver", f"${curr.get('Silver',0):.2f}")
            m3.metric("Oil", f"${curr.get('Oil',0):.2f}")
            m4.metric("10Y Yield", f"{curr.get('10Y Yield',0):.2f}%")
    else: st.error("Data Error.")

# TAB 2: 10Y DEEP DIVE (Preserved)
with t_deep:
    st.subheader("VIX vs S&P 500 (10 Years)")
    if not macro_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=macro_df.index, y=macro_df['S&P 500'], name="S&P 500", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=macro_df.index, y=macro_df['VIX'], name="VIX", line=dict(color='gray', width=1), yaxis="y2"))
        danger = macro_df[macro_df['VIX'] > 30]
        fig.add_trace(go.Scatter(x=danger.index, y=danger['S&P 500'], mode='markers', marker=dict(color='red', symbol='x'), name="Crash Zone"))
        fig.update_layout(height=500, yaxis2=dict(overlaying='y', side='right'), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap
        st.subheader("Annual Returns Heatmap")
        y_df = macro_df.resample('Y').last().pct_change() * 100
        y_df.index = y_df.index.year
        fig_h = px.imshow(y_df.T, text_auto=".1f", color_continuous_scale="RdYlGn", aspect="auto")
        st.plotly_chart(fig_h, use_container_width=True)

# TAB 3: BALKEN COMPARISON (Time Selectable)
with t_comp:
    st.subheader("Performance Rankings (Balken)")
    timeframe = st.selectbox("Select Timeframe:", ["1 Month", "3 Months", "6 Months", "1 Year"], index=0)
    
    if not macro_df.empty:
        lb_map = {"1 Month": 22, "3 Months": 66, "6 Months": 126, "1 Year": 252}
        days = lb_map[timeframe]
        idx = -days if len(macro_df) > days else 0
        ret = ((macro_df.iloc[-1] - macro_df.iloc[idx]) / macro_df.iloc[idx]) * 100
        
        # Plot Sectors
        secs = ["Tech", "Energy", "Financials", "Healthcare", "Staples", "Discretionary", "Real Estate", "Utilities", "Materials"]
        valid_s = [c for c in secs if c in ret]
        if valid_s:
            s_data = ret[valid_s].sort_values()
            fig_s = px.bar(x=s_data.values, y=s_data.index, orientation='h', title=f"Sectors ({timeframe})", color=s_data.values, color_continuous_scale="RdYlGn")
            st.plotly_chart(fig_s, use_container_width=True)

# TAB 4: INDEX COMPOSITION (Preserved)
with t_holdings:
    st.subheader("Index Heavyweights (Live Weight %)")
    idx_choice = st.radio("Select Index:", ["S&P 500", "Nasdaq 100", "DAX (Germany)"], horizontal=True)
    if st.button("Get Weights"):
        with st.spinner("Calculating..."):
            comp_df = fetch_index_composition(idx_choice)
            if not comp_df.empty:
                c1, c2 = st.columns([1, 2])
                c1.dataframe(comp_df, hide_index=True)
                fig_pie = px.pie(comp_df, values='Market Cap', names='Ticker', title=f'Top Constituents')
                c2.plotly_chart(fig_pie, use_container_width=True)

# TAB 5: SCANNER (Fixed & Robust)
with t_scan:
    if st.button("RUN LIVE SCAN", type="primary"):
        with st.spinner(f"Scanning {univ}..."):
            stock_data = fetch_scanner_batch(univ)
            
            res = []
            if stock_data:
                for t, df in stock_data.items():
                    sc = score_stock(df)
                    res.append({"Ticker": t, "Score": sc, "Price": df['Close'].iloc[-1]})
            
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

