import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# 1. PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(layout="wide", page_title="Market Regime & Amplitude Scanner")

st.markdown("""
<style>
    .metric-box {
        background-color: #1e1e1e;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .bearish-box {
        border-left: 5px solid #FF5252;
    }
    .neutral-box {
        border-left: 5px solid #FFC107;
    }
    h1, h2, h3 { color: #f0f0f0; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE
# ==========================================

@st.cache_data(ttl=3600)
def get_macro_data():
    """Fetches key macro assets to determine regime."""
    # CL=F: Oil, HG=F: Copper, LBS=F: Lumber, ^TNX: 10Y Yield, GC=F: Gold
    tickers = {
        "Oil": "CL=F", 
        "Copper": "HG=F", 
        "Lumber": "LBS=F", 
        "Gold": "GC=F", 
        "10Y Yield": "^TNX",
        "SPY": "SPY",
        "Staples": "XLP",
        "Discretionary": "XLY",
        "Russell": "IWM",
        "Nasdaq": "QQQ"
    }
    
    data = yf.download(list(tickers.values()), period="6mo", progress=False)['Close']
    
    # Rename columns to friendly names
    reverse_map = {v: k for k, v in tickers.items()}
    data.columns = [reverse_map.get(c, c) for c in data.columns]
    
    return data

def get_stock_data(tickers):
    """Fetches stock data with extended amplitude metrics."""
    data_dict = {}
    valid_tickers = [t.strip().upper() for t in tickers.split(',')]
    
    if not valid_tickers:
        return {}

    # Bulk download for speed
    df_bulk = yf.download(valid_tickers, period="1y", group_by='ticker', progress=False)
    
    for t in valid_tickers:
        try:
            # Handle single ticker vs multi-ticker structure
            if len(valid_tickers) == 1:
                df = df_bulk.copy()
            else:
                df = df_bulk[t].copy()
            
            if df.empty: continue
            
            # Drop NaN
            df.dropna(inplace=True)

            # --- ADVANCED TECHNICALS (The "Amplitude" & Structure) ---
            
            # 1. Trend
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            df['SMA_200'] = ta.sma(df['Close'], length=200)
            
            # 2. Amplitude / Volatility
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
            
            # Bollinger Bands (for Overextended checks)
            bb = ta.bbands(df['Close'], length=20, std=2)
            df['BB_Upper'] = bb['BBU_20_2.0']
            df['BB_Lower'] = bb['BBL_20_2.0']
            
            # 3. Momentum
            df['RSI'] = ta.rsi(df['Close'], length=14)
            
            # 4. Volume Flow
            # Relative Volume (Current Vol vs 20 SMA)
            df['RVol'] = df['Volume'] / ta.sma(df['Volume'], length=20)
            
            data_dict[t] = df
            
        except Exception as e:
            pass
            
    return data_dict

# ==========================================
# 3. LOGIC ENGINE
# ==========================================

def analyze_macro_regime(macro_df):
    """
    Compares commodities and sectors to define the 'Weather'.
    """
    curr = macro_df.iloc[-1]
    prev_month = macro_df.iloc[-20] # approx 1 month ago
    
    # Calculates % Change
    pct_changes = ((curr - prev_month) / prev_month) * 100
    
    regime = {
        "inflation": "Neutral",
        "risk_sentiment": "Neutral",
        "economy": "Neutral",
        "alert": "None"
    }
    
    # 1. Inflation Check (Oil & Copper)
    if pct_changes['Oil'] > 5 and pct_changes['Copper'] > 5:
        regime['inflation'] = "Heating Up (Rising Costs)"
    elif pct_changes['Oil'] < -5:
        regime['inflation'] = "Cooling/Deflationary"
        
    # 2. Economic Health (Copper & Lumber vs Gold)
    # Copper is "Dr. Copper" (Economy) vs Gold (Fear)
    if pct_changes['Copper'] > pct_changes['Gold']:
        regime['economy'] = "Expansionary (Risk On)"
    else:
        regime['economy'] = "Contractionary (Safety Seek)"
        
    # 3. Market Breadth / Risk Appetite (XLY vs XLP)
    # Discretionary (spending) vs Staples (toothpaste/food)
    if pct_changes['Discretionary'] > pct_changes['Staples']:
        regime['risk_sentiment'] = "Bullish (Offense)"
    else:
        regime['risk_sentiment'] = "Bearish (Defense)"
        
    # 4. Yield Pressure
    yield_level = curr['10Y Yield']
    if yield_level > 4.5:
        regime['alert'] = "⚠️ High Rates Pressuring Tech/Growth"
    
    return regime, pct_changes

def score_confluence(df, bias="bull"):
    """
    Returns a 0-100 score based on CONFLUENCE of factors.
    Strict Logic: We don't want 'loosy' results.
    """
    row = df.iloc[-1]
    score = 0
    reasons = []
    
    # --- BULLISH LOGIC ---
    if bias == "bull":
        # 1. Structure (30 pts)
        if row['Close'] > row['SMA_200']: 
            score += 20
        if row['Close'] > row['SMA_50']: 
            score += 10
            
        # 2. Momentum (RSI) (20 pts)
        # We want not overbought yet (room to run)
        if 40 < row['RSI'] < 70:
            score += 20
            
        # 3. Volatility Compression or Breakout (30 pts)
        # If ADX > 25, trend is strong
        if row['ADX'] > 25:
            score += 15
            reasons.append("Strong Trend Strength")
        # Price near upper BB but not blown out
        if row['Close'] > row['BB_Upper'] * 0.98:
            score += 15
            reasons.append("Testing Upper Breakout")
            
        # 4. Volume (20 pts)
        if row['RVol'] > 1.2: # 20% higher volume than normal
            score += 20
            reasons.append("High Volume Interest")
            
    # --- BEARISH LOGIC ---
    elif bias == "bear":
        # 1. Structure (30 pts)
        if row['Close'] < row['SMA_200']: 
            score += 20
        if row['Close'] < row['SMA_50']: 
            score += 10
            
        # 2. Momentum (20 pts)
        if 30 < row['RSI'] < 60: # Not oversold yet
            score += 20
            
        # 3. Volatility (30 pts)
        if row['ADX'] > 25:
            score += 15
        if row['Close'] < row['BB_Lower'] * 1.02:
            score += 15
            reasons.append("Testing Lower Breakdown")
            
        # 4. Volume (20 pts)
        if row['RVol'] > 1.2:
            score += 20
            reasons.append("High Volume Selling")

    return score, ", ".join(reasons)

# ==========================================
# 4. DASHBOARD LAYOUT
# ==========================================

# A. Header & Macro View
st.title("🦅 Macro-Technical Radar")

with st.spinner("Scanning Global Markets (Commodities, Rates, Indices)..."):
    macro_data = get_macro_data()
    regime, changes = analyze_macro_regime(macro_data)

# Macro Metrics Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Oil (Inflation)", f"${macro_data['Oil'].iloc[-1]:.2f}", f"{changes['Oil']:.2f}%")
m2.metric("Copper (Economy)", f"${macro_data['Copper'].iloc[-1]:.2f}", f"{changes['Copper']:.2f}%")
m3.metric("10Y Yield", f"{macro_data['10Y Yield'].iloc[-1]:.2f}%", f"{changes['10Y Yield']:.2f}%")
m4.metric("Risk Appetite (XLY/XLP)", regime['risk_sentiment'], delta_color="off")

# Macro Context Box
st.info(f"**MARKET CONTEXT:** The economy appears **{regime['economy']}**. Inflation signals are **{regime['inflation']}**. {regime['alert']}")

if regime['risk_sentiment'].startswith("Bearish") and regime['economy'].startswith("Contractionary"):
    st.error("🚨 WARNING: MACRO ENVIRONMENT IS HOSTILE. PRIORITIZE CASH OR SHORTS.")

# B. Stock Search
st.divider()
st.subheader("🔍 Deep Technical Scanner")

col_input, col_settings = st.columns([2, 1])
with col_input:
    default_list = "NVDA, TSLA, AAPL, AMD, MSFT, AMZN, META, GOOGL, NFLX, COIN, MSTR, SPY, QQQ, IWM"
    tickers = st.text_area("Watchlist (Comma Separated)", default_list)

with col_settings:
    min_score = st.slider("Min Confluence Score (0-100)", 0, 100, 70, help="Higher = Stricter Criteria")
    show_charts = st.checkbox("Show Charts for Top Picks", value=True)

if st.button("RUN ANALYSIS"):
    stock_dict = get_stock_data(tickers)
    
    bull_results = []
    bear_results = []
    
    progress = st.progress(0)
    
    for i, (ticker, df) in enumerate(stock_dict.items()):
        # Score Bullish
        bull_score, bull_reason = score_confluence(df, bias="bull")
        if bull_score >= min_score:
            bull_results.append({
                "Ticker": ticker, 
                "Score": bull_score, 
                "Price": df['Close'].iloc[-1],
                "ATR (Vol)": df['ATR'].iloc[-1],
                "RVol": df['RVol'].iloc[-1],
                "Setup": bull_reason
            })
            
        # Score Bearish
        bear_score, bear_reason = score_confluence(df, bias="bear")
        if bear_score >= min_score:
            bear_results.append({
                "Ticker": ticker, 
                "Score": bear_score, 
                "Price": df['Close'].iloc[-1],
                "ATR (Vol)": df['ATR'].iloc[-1],
                "RVol": df['RVol'].iloc[-1],
                "Setup": bear_reason
            })
        
        progress.progress((i + 1) / len(stock_dict))

    # C. Results Display
    
    # --- BULLISH COLUMN ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### 🐂 Long Opportunities")
        if bull_results:
            df_bull = pd.DataFrame(bull_results).sort_values(by="Score", ascending=False)
            st.dataframe(
                df_bull.style.background_gradient(subset=['Score'], cmap='Greens'),
                use_container_width=True, 
                hide_index=True
            )
            
            if show_charts:
                top_bull = df_bull.iloc[0]['Ticker']
                # Bollinger + Keltner Channel visualization
                df_p = stock_dict[top_bull].tail(100)
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name='Price'))
                fig.add_trace(go.Scatter(x=df_p.index, y=df_p['BB_Upper'], line=dict(color='rgba(0, 255, 0, 0.5)'), name='Upper Band'))
                fig.add_trace(go.Scatter(x=df_p.index, y=df_p['BB_Lower'], line=dict(color='rgba(0, 255, 0, 0.5)'), name='Lower Band'))
                fig.add_trace(go.Scatter(x=df_p.index, y=df_p['SMA_50'], line=dict(color='yellow'), name='SMA 50'))
                
                fig.update_layout(title=f"Top Bull Pick: {top_bull} (Amplitude Check)", height=400, template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No assets met the Bullish criteria.")

    # --- BEARISH COLUMN ---
    with c2:
        st.markdown("### 🐻 Short Opportunities")
        if bear_results:
            df_bear = pd.DataFrame(bear_results).sort_values(by="Score", ascending=False)
            st.dataframe(
                df_bear.style.background_gradient(subset=['Score'], cmap='Reds'),
                use_container_width=True, 
                hide_index=True
            )
            
            if show_charts:
                top_bear = df_bear.iloc[0]['Ticker']
                df_p = stock_dict[top_bear].tail(100)
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name='Price'))
                fig.add_trace(go.Scatter(x=df_p.index, y=df_p['BB_Upper'], line=dict(color='rgba(255, 0, 0, 0.5)'), name='Upper Band'))
                fig.add_trace(go.Scatter(x=df_p.index, y=df_p['BB_Lower'], line=dict(color='rgba(255, 0, 0, 0.5)'), name='Lower Band'))
                fig.add_trace(go.Scatter(x=df_p.index, y=df_p['SMA_50'], line=dict(color='yellow'), name='SMA 50'))
                
                fig.update_layout(title=f"Top Bear Pick: {top_bear} (Amplitude Check)", height=400, template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No assets met the Bearish criteria.")
