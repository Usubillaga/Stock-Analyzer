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
st.set_page_config(layout="wide", page_title="Titan: Omni-Market Analyst")

st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; color: #212529; font-family: 'Segoe UI', sans-serif; }
    
    /* Metrics & Cards */
    div[data-testid="stMetric"] { background-color: #FFFFFF; border: 1px solid #DEE2E6; padding: 10px; border-radius: 5px; }
    .card { background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #ddd; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    
    /* Regime Indicators */
    .regime-bull { border-left: 8px solid #28a745; background-color: #e8f5e9; padding: 15px; }
    .regime-bear { border-left: 8px solid #dc3545; background-color: #f8d7da; padding: 15px; }
    .regime-warn { border-left: 8px solid #ffc107; background-color: #fff3cd; padding: 15px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE (ROBUST & COMPLETE)
# ==========================================

@st.cache_data(ttl=3600)
def fetch_global_data():
    """
    Fetches Indices, Commodities, Sectors, and Breadth data.
    """
    tickers = {
        # --- GLOBAL INDICES ---
        "S&P 500": "SPY", "Nasdaq 100": "QQQ", "Russell 2000": "IWM", 
        "DAX (Germany)": "^GDAXI", "FTSE 100 (UK)": "^FTSE", "Nikkei 225 (Japan)": "^N225",
        
        # --- MARKET BREADTH ---
        "S&P Equal Wgt": "RSP", 
        
        # --- COMMODITIES (INFLATION) ---
        "Oil (USO)": "USO", "Copper (CPER)": "CPER", 
        "Lumber (WOOD)": "WOOD", "Gold (GLD)": "GLD", "Silver (SLV)": "SLV",
        
        # --- MACRO RATES ---
        "10Y Yield": "^TNX", "VIX": "^VIX", "USD Index": "DX-Y.NYB",
        
        # --- SECTOR ETFS ---
        "Tech (XLK)": "XLK", "Energy (XLE)": "XLE", "Financials (XLF)": "XLF",
        "Healthcare (XLV)": "XLV", "Discretionary (XLY)": "XLY", "Staples (XLP)": "XLP",
        "Utilities (XLU)": "XLU", "Materials (XLB)": "XLB", "Real Estate (XLRE)": "XLRE",
        "Industrials (XLI)": "XLI", "Comms (XLC)": "XLC"
    }
    
    try:
        data = yf.download(list(tickers.values()), period="1y", progress=False, auto_adjust=True)
        
        # Robust Flattening Logic
        clean = pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            for k, v in tickers.items():
                try:
                    # Try Level 0 (Column Name) or Level 1 (Ticker)
                    if v in data.columns.levels[1]: clean[k] = data.xs(v, axis=1, level=1)['Close']
                    elif v in data.columns.levels[0]: clean[k] = data[v]['Close']
                except: pass
        else:
            for k, v in tickers.items():
                if v in data: clean[k] = data[v]
                
        clean.fillna(method='ffill', inplace=True)
        return clean, tickers
    except Exception:
        return pd.DataFrame(), tickers

@st.cache_data(ttl=600)
def fetch_scanner_batch(universe):
    """Fetches stock data for the scanner."""
    if universe == "US Tech / Growth":
        ticks = ["NVDA", "AMD", "TSLA", "PLTR", "COIN", "MARA", "MSTR", "HOOD", "SOFI", "UBER", "DKNG", "ROKU", "NET", "CRWD", "SNOW", "PANW", "RIVN", "LCID", "AMZN", "GOOGL", "META"]
    elif universe == "Mega Cap Stability":
        ticks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "BRK-B", "JPM", "V", "MA", "PG", "JNJ", "XOM", "CVX", "COST", "PEP", "KO", "MCD", "WMT", "HD", "LLY", "UNH"]
    else: # DAX / Europe
        ticks = ["SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BMW.DE", "VOW3.DE", "BAS.DE", "ADS.DE", "AIR.DE", "DHL.DE", "DB1.DE", "MUV2.DE", "IFX.DE", "RWE.DE", "EOAN.DE", "BAYN.DE"]

    data = {}
    try:
        raw = yf.download(ticks, period="1y", group_by='ticker', progress=False, threads=True, auto_adjust=True)
        for t in ticks:
            try:
                # Robust extraction
                if isinstance(raw.columns, pd.MultiIndex):
                    if t in raw.columns.levels[0]: df = raw[t].copy()
                    else: continue
                elif len(ticks) == 1: df = raw.copy()
                else: continue

                df.dropna(subset=['Close'], inplace=True)
                if len(df) < 50: continue
                
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
# 3. ANALYTIC ENGINES (MACRO + STOCK)
# ==========================================

def macro_ai_analyst(df, in_gdp, in_unemp, in_cpi):
    if df.empty: return {}, []
    
    curr = df.iloc[-1]
    prev = df.iloc[-22]
    chg = ((curr - prev)/prev)*100
    
    advice = []
    score = 0
    
    # 1. Inflation Analysis
    comm_basket = (chg.get('Oil (USO)',0) + chg.get('Copper (CPER)',0) + chg.get('Lumber (WOOD)',0)) / 3
    if in_cpi > 4.0 or comm_basket > 3.0:
        advice.append(f"🔥 **Inflation Alert:** CPI {in_cpi}% & Commodities Rising (+{comm_basket:.1f}%). Hard assets preferred.")
        score -= 2
    else:
        advice.append("✅ **Inflation Stable:** Commodities behaving. Good for Tech.")
        score += 1
        
    # 2. Economic Health
    if in_gdp < 1.0 or in_unemp > 5.0:
        advice.append("⚠️ **Recession Risk:** Weak GDP/Jobs data.")
        score -= 2
    elif chg.get('Copper (CPER)', 0) > 0:
        advice.append("🏗️ **Economic Strength:** Copper (Dr. Copper) is rising, signaling demand.")
        score += 1

    # 3. Market Breadth
    spy_chg = chg.get('S&P 500', 0)
    rsp_chg = chg.get('S&P Equal Wgt', 0)
    if spy_chg > rsp_chg + 1.0:
        advice.append("📉 **Weak Breadth:** Rally is narrow (Mega-caps only). Caution.")
        score -= 1
    else:
        advice.append("💪 **Healthy Breadth:** Equal Weight S&P is participating.")
        score += 1

    # 4. Rates
    if curr.get('10Y Yield',0) > 4.5:
        advice.append("💸 **Rate Pressure:** 10Y Yield > 4.5% hurts valuations.")
        score -= 1
        
    # Verdict
    if score >= 1: regime = {"status": "BULLISH (RISK ON)", "css": "regime-bull"}
    elif score <= -1: regime = {"status": "BEARISH (DEFENSIVE)", "css": "regime-bear"}
    else: regime = {"status": "NEUTRAL / CHOPPY", "css": "regime-warn"}
    
    return regime, advice

def score_stock(df):
    """Rank stocks from -100 (Bear) to +100 (Bull)"""
    row = df.iloc[-1]
    s = 0
    
    # Trend (40pts)
    if row['Close'] > row['SMA_200']: s += 20
    else: s -= 20
    if row['Close'] > row['SMA_50']: s += 20
    else: s -= 20
    
    # Momentum (30pts)
    if row['RSI'] < 30: s += 30 # Buy Dip
    elif row['RSI'] > 75: s -= 30 # Sell Rip
    elif 50 < row['RSI'] < 70: s += 10 # Strong
    
    # Structure (30pts)
    if row['Close'] <= row['BB_Lower'] * 1.01: s += 30
    if row['Close'] >= row['BB_Upper'] * 0.99: s -= 30
    
    return s

def get_monte_carlo(df, days=30):
    """Generates Future Cones"""
    last = df['Close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    dates = [df.index[-1] + timedelta(days=i) for i in range(1, days+1)]
    
    bull = [last + (atr * 0.5 * i) for i in range(1, days+1)]
    bear = [last - (atr * 0.5 * i) for i in range(1, days+1)]
    base = [last for i in range(1, days+1)]
    
    return dates, bull, bear, base

# ==========================================
# 4. UI LAYOUT
# ==========================================

st.title("🦅 Titan: Omni-Market Analyst")

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Economic Inputs")
in_gdp = st.sidebar.number_input("GDP Growth (%)", 2.5)
in_unemp = st.sidebar.number_input("Unemployment (%)", 3.8)
in_cpi = st.sidebar.number_input("CPI Inflation (%)", 3.2)
st.sidebar.markdown("---")
st.sidebar.header("2. Scanner Config")
univ = st.sidebar.selectbox("Universe", ["US Tech / Growth", "Mega Cap Stability", "DAX / Europe"])

# --- DATA LOADING ---
with st.spinner("Connecting to Global Exchanges..."):
    macro_df, ticker_map = fetch_global_data()
    regime, advice = macro_ai_analyst(macro_df, in_gdp, in_unemp, in_cpi)

# --- TABS ---
t_macro, t_sectors, t_scanner = st.tabs(["🌍 Macro Headquarters", "📊 Sectors & Areas", "🚀 Stock Scanner"])

# TAB 1: MACRO
with t_macro:
    if not macro_df.empty:
        st.markdown(f"<div class='{regime['css']}'><h2>REGIME: {regime['status']}</h2></div>", unsafe_allow_html=True)
        st.markdown("### 🧠 AI Analyst")
        for a in advice: st.info(a)
        
        st.divider()
        curr = macro_df.iloc[-1]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("S&P 500", f"{curr.get('S&P 500',0):.0f}")
        c2.metric("VIX", f"{curr.get('VIX',0):.2f}")
        c3.metric("10Y Yield", f"{curr.get('10Y Yield',0):.2f}%")
        c4.metric("Oil", f"${curr.get('Oil (USO)',0):.2f}")
        c5.metric("Copper", f"${curr.get('Copper (CPER)',0):.2f}")
    else:
        st.error("Macro data unavailable. API Error.")

# TAB 2: SECTORS & AREAS
with t_sectors:
    st.subheader("Global Indices Performance")
    # Indices List
    idx_cols = ["S&P 500", "Nasdaq 100", "Russell 2000", "DAX (Germany)", "FTSE 100 (UK)", "Nikkei 225 (Japan)"]
    valid_idx = [c for c in idx_cols if c in macro_df.columns]
    if valid_idx:
        norm_idx = (macro_df[valid_idx] / macro_df[valid_idx].iloc[0]) * 100
        st.line_chart(norm_idx)

    st.subheader("Sector Rotation (The 11 Areas)")
    # Sectors List
    sec_cols = ["Tech (XLK)", "Energy (XLE)", "Financials (XLF)", "Healthcare (XLV)", "Discretionary (XLY)", "Staples (XLP)", "Utilities (XLU)"]
    valid_sec = [c for c in sec_cols if c in macro_df.columns]
    if valid_sec:
        norm_sec = (macro_df[valid_sec] / macro_df[valid_sec].iloc[0]) * 100
        st.line_chart(norm_sec)

# TAB 3: SCANNER
with t_scanner:
    col_run, col_view = st.columns([1, 4])
    with col_run:
        scan_btn = st.button("RUN SCAN", type="primary", use_container_width=True)
    
    if scan_btn:
        with st.spinner(f"Scanning {univ} structure..."):
            stock_data = fetch_scanner_batch(univ)
            
            res_list = []
            for t, df in stock_data.items():
                sc = score_stock(df)
                res_list.append({
                    "Ticker": t, 
                    "Score": sc, 
                    "Price": df['Close'].iloc[-1],
                    "RSI": df['RSI'].iloc[-1]
                })
            
            if not res_list:
                st.error("No data found. Markets may be closed.")
            else:
                df_r = pd.DataFrame(res_list)
                bulls = df_r[df_r['Score']>0].sort_values('Score', ascending=False).head(10)
                bears = df_r[df_r['Score']<0].sort_values('Score', ascending=True).head(10)
                
                c_bull, c_bear = st.columns(2)
                
                # BULLS
                with c_bull:
                    st.success("🟢 Top Bulls (Long)")
                    st.dataframe(bulls, hide_index=True)
                    if not bulls.empty:
                        top = bulls.iloc[0]['Ticker']
                        st.caption(f"Projection: {top}")
                        df_p = stock_data[top]
                        d, b, s, n = get_monte_carlo(df_p)
                        fig = go.Figure(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name='Px'))
                        fig.add_trace(go.Scatter(x=d, y=b, line=dict(color='green', dash='dot'), name='Bull'))
                        fig.add_trace(go.Scatter(x=d, y=s, line=dict(color='red', dash='dot'), name='Bear'))
                        fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                # BEARS
                with c_bear:
                    st.error("🔴 Top Bears (Short)")
                    st.dataframe(bears, hide_index=True)
                    if not bears.empty:
                        top = bears.iloc[0]['Ticker']
                        st.caption(f"Projection: {top}")
                        df_p = stock_data[top]
                        d, b, s, n = get_monte_carlo(df_p)
                        fig = go.Figure(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name='Px'))
                        fig.add_trace(go.Scatter(x=d, y=b, line=dict(color='green', dash='dot'), name='Bull'))
                        fig.add_trace(go.Scatter(x=d, y=s, line=dict(color='red', dash='dot'), name='Bear'))
                        fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
