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
    .regime-bull { border-left: 10px solid #00C853; background-color: #E8F5E9; padding: 20px; font-weight: bold; border-radius: 5px; }
    .regime-bear { border-left: 10px solid #D50000; background-color: #FFEBEE; padding: 20px; font-weight: bold; border-radius: 5px; }
    .regime-warn { border-left: 10px solid #FFAB00; background-color: #FFF3E0; padding: 20px; font-weight: bold; border-radius: 5px; }
    
    /* AI Analyst Box */
    .ai-box { background-color: #fff; padding: 15px; border-radius: 8px; border-left: 5px solid #2962FF; margin-bottom: 10px; }
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
        "VIX": "^VIX", "10Y Yield": "^TNX",
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
        # Robust Extraction
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
    """Fetches stock data. Uses safe liquid lists."""
    if universe == "US Tech / Growth":
        ticks = ["NVDA", "AMD", "TSLA", "PLTR", "COIN", "UBER", "AMZN", "GOOGL", "META", "MSFT", "AAPL", "NET", "CRWD", "RIVN"]
    elif universe == "Global Macro (ETFs)":
        ticks = ["SPY", "QQQ", "IWM", "EEM", "GLD", "SLV", "USO", "TLT", "HYG", "FXI", "EWZ"]
    else: # DAX / Europe
        ticks = ["SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BMW.DE", "ADS.DE", "AIR.DE", "DHL.DE", "DB1.DE", "RWE.DE"]

    data = {}
    try:
        raw = yf.download(ticks, period="2y", group_by='ticker', progress=False, threads=True, auto_adjust=True)
        for t in ticks:
            try:
                # Extract
                if isinstance(raw.columns, pd.MultiIndex):
                    if t in raw.columns.levels[0]: df = raw[t].copy()
                    else: continue
                elif len(ticks) == 1: df = raw.copy()
                else: continue

                df.dropna(subset=['Close'], inplace=True)
                if len(df) < 100: continue
                
                # Indicators
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

def macro_ai_analyst(df, gdp, unemp, cpi):
    """SENSITIVE Logic: Reacts quickly to VIX > 20 and Momentum drops."""
    if df.empty: return {}, []
    
    curr = df.iloc[-1]
    mom_1m = ((curr.get('S&P 500', 0) - df['S&P 500'].iloc[-22]) / df['S&P 500'].iloc[-22]) * 100
    
    advice = []
    score = 0
    
    # 1. VIX SENSITIVITY (Fear)
    vix = curr.get('VIX', 15)
    if vix > 28:
        advice.append(f"🚨 **CRASH WARNING (VIX {vix:.1f}):** Extreme fear. Markets are collapsing.")
        score -= 5
    elif vix > 20:
        advice.append(f"⚠️ **High Stress (VIX {vix:.1f}):** Volatility is elevated. Reduce risk.")
        score -= 2
    else:
        advice.append("✅ **Calm Markets:** VIX is low (<20). Bullish tailwind.")
        score += 1

    # 2. ECONOMIC INPUTS
    if gdp < 1.0 or unemp > 5.0:
        advice.append(f"⚠️ **Recession Risk:** GDP {gdp}% / Unemployment {unemp}%.")
        score -= 2
    
    # 3. INFLATION (CPI + Commodities)
    oil_trend = ((curr.get('Oil',0) - df['Oil'].iloc[-60])/df['Oil'].iloc[-60])*100
    if cpi > 4.0 or oil_trend > 10:
        advice.append(f"🔥 **Inflation Drag:** CPI {cpi}% & Oil Spiking. Bad for consumers.")
        score -= 1
    else:
        advice.append("✅ **Inflation Tame:** Commodities stable.")
        score += 1

    # VERDICT
    if score >= 1: regime = {"status": "BULLISH (BUY DIPS)", "css": "regime-bull"}
    elif score <= -2: regime = {"status": "BEARISH (PROTECT CAPITAL)", "css": "regime-bear"}
    else: regime = {"status": "CAUTION / NEUTRAL", "css": "regime-warn"}
    
    return regime, advice

def sector_ai_analyst(ret_series):
    """Analyzes Sector Rotation (Risk On/Off)."""
    if ret_series.empty: return "No data."
    
    tech = ret_series.get("Tech", 0)
    staples = ret_series.get("Staples", 0)
    utils = ret_series.get("Utilities", 0)
    
    if tech > staples and tech > utils:
        return "🐂 **Risk-On Flows:** Technology is outperforming Defensives. Investors are aggressive."
    elif staples > tech or utils > tech:
        return "🐻 **Risk-Off Flows:** Investors are hiding in Staples/Utilities. Fear is dominating."
    else:
        return "⚖️ **Mixed Rotation:** No clear sector leadership."

def analyze_10y_cycles(df):
    """Dedicated Analyst for the 10Y Chart."""
    if 'VIX' not in df: return []
    avg_vix = df['VIX'].mean()
    curr_vix = df['VIX'].iloc[-1]
    
    insights = []
    if curr_vix > 30:
        insights.append("• **Cycle Status:** CRASH MODE. Historically, VIX > 30 marks panic bottoms or capitulation.")
    elif curr_vix < 12:
        insights.append("• **Cycle Status:** COMPLACENCY. Markets are overly calm, risk of sudden spike is high.")
    else:
        insights.append(f"• **Cycle Status:** Normal Volatility. Current: {curr_vix:.1f} (Avg: {avg_vix:.1f}).")
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
    if row['Close'] > row['SMA_200']: s += 30
    else: s -= 30
    if row['RSI'] < 30: s += 30 
    elif row['RSI'] > 75: s -= 30
    if row['Close'] < row['BB_Lower']: s += 20
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
    regime, advice = macro_ai_analyst(macro_df, in_gdp, in_unemp, in_cpi)

# TABS
t_hq, t_deep, t_comp, t_scan = st.tabs(["🌍 Macro HQ", "📉 10Y Deep Dive", "📊 Sector Balken", "🚀 Stock Scanner"])

# TAB 1: MACRO HQ
with t_hq:
    if not macro_df.empty:
        st.markdown(f"<div class='{regime['css']}'><h2>REGIME: {regime['status']}</h2></div>", unsafe_allow_html=True)
        st.markdown("### 🧠 Sensitive AI Analyst")
        for a in advice: st.info(a)
        
        st.divider()
        curr = macro_df.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("VIX", f"{curr.get('VIX',0):.2f}")
        c2.metric("10Y Yield", f"{curr.get('10Y Yield',0):.2f}%")
        c3.metric("Oil", f"${curr.get('Oil',0):.2f}")
        c4.metric("Copper", f"${curr.get('Copper',0):.2f}")
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
with t_comp:
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

# TAB 4: SCANNER (Fixed + Monte Carlo)
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
                st.warning("Scanner returned no data. Markets may be closed.")

