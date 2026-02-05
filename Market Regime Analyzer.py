import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION & PAGE SETUP
# ==========================================
st.set_page_config(layout="wide", page_title="AI Stock Analyzer Pro")

# Custom CSS for buttons and metrics
st.markdown("""
<style>
    .bullish { color: #28a745; font-weight: bold; }
    .bearish { color: #dc3545; font-weight: bold; }
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS: TECHNICAL ANALYSIS
# ==========================================

def calculate_technicals(df):
    """Calculates RSI, Moving Averages, and Volume trends."""
    if len(df) < 200:
        return df
    
    # Moving Averages
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volume SMA
    df['Vol_SMA_20'] = df['Volume'].rolling(window=20).mean()
    
    return df

def detect_patterns(df):
    """Detects Candle Patterns and Market Structure."""
    # We need at least 3 days of data for patterns
    if len(df) < 3:
        return "Insufficient Data", "Unknown"

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    signal = "Neutral"
    reason = "No distinct pattern"
    
    # 1. CANDLESTICK PATTERNS
    
    # Bullish Engulfing
    if (prev['Close'] < prev['Open']) and \
       (curr['Close'] > curr['Open']) and \
       (curr['Close'] > prev['Open']) and \
       (curr['Open'] < prev['Close']):
        signal = "Bullish"
        reason = "Bullish Engulfing Candle"

    # Bearish Engulfing
    elif (prev['Close'] > prev['Open']) and \
         (curr['Close'] < curr['Open']) and \
         (curr['Close'] < prev['Open']) and \
         (curr['Open'] > prev['Close']):
        signal = "Bearish"
        reason = "Bearish Engulfing Candle"

    # Hammer (Bullish Reversal)
    # Body is small, lower wick is long
    body = abs(curr['Close'] - curr['Open'])
    lower_wick = min(curr['Close'], curr['Open']) - curr['Low']
    upper_wick = curr['High'] - max(curr['Close'], curr['Open'])
    
    if (lower_wick > 2 * body) and (upper_wick < body):
        signal = "Bullish"
        reason = "Hammer Candle (Potential Reversal)"

    # 2. VOLUME ANALYSIS
    # High relative volume confirms moves
    if curr['Volume'] > 1.5 * curr['Vol_SMA_20']:
        reason += " + High Volume Spike"
    
    # 3. STRUCTURE (Simple Trend Filter)
    # If price is below 200 SMA, we prioritize Bearish signals
    if curr['Close'] < curr['SMA_200']:
        structure = "Downtrend"
    else:
        structure = "Uptrend"

    return signal, reason, structure

def get_data(tickers, period="1y"):
    """Fetches data for a list of tickers."""
    data_dict = {}
    for t in tickers:
        try:
            df = yf.download(t, period=period, progress=False)
            if len(df) > 0:
                # Flat multi-index columns if necessary
                if isinstance(df.columns, pd.MultiIndex):
                     df.columns = df.columns.get_level_values(0)
                data_dict[t] = calculate_technicals(df)
        except Exception as e:
            st.warning(f"Could not fetch data for {t}")
    return data_dict

# ==========================================
# 3. SIDEBAR: MACRO & INPUTS
# ==========================================

st.sidebar.title("⚙️ Control Panel")

# A. Stock Selection
st.sidebar.header("1. Assets")
default_tickers = "NVDA, TSLA, AAPL, AMD, MSFT, GOOGL, AMZN, SPY, QQQ"
ticker_input = st.sidebar.text_area("Enter Tickers (comma separated)", default_tickers)
ticker_list = [x.strip().upper() for x in ticker_input.split(',')]

# B. Macro Environment Simulation
st.sidebar.header("2. Macro Environment")
st.sidebar.info("Since live macro APIs require paid keys, please input current estimates or 'Simulate' a scenario.")

inflation_rate = st.sidebar.slider("Inflation Rate (%)", 0.0, 20.0, 3.2)
interest_rate = st.sidebar.slider("10Y Treasury Yield (%)", 0.0, 10.0, 4.0)
vix_level = st.sidebar.number_input("VIX (Volatility Index)", value=15.0)

# C. Macro Logic
macro_status = "NEUTRAL"
macro_advice = "Proceed with caution."
macro_color = "orange"

# Logic: Hyperinflation Check
if inflation_rate > 6.0 or interest_rate > 5.0:
    macro_status = "EXTREME BEARISH (Bad Enviroment)"
    macro_advice = "⛔ HIGH RISK: Cash is King. Avoid Longs. Look for Shorts."
    macro_color = "red"
elif inflation_rate > 4.0 and vix_level > 25:
    macro_status = "BEARISH (Risk Off)"
    macro_advice = "⚠️ Caution: High Volatility & Rates. Tight stops required."
    macro_color = "orange"
elif inflation_rate < 3.0 and interest_rate < 3.5 and vix_level < 20:
    macro_status = "BULLISH (Goldilocks)"
    macro_advice = "✅ Environment is favorable for growth stocks."
    macro_color = "green"

# ==========================================
# 4. MAIN APP LOGIC
# ==========================================

st.title("📊 AI Technical & Macro Screener")
st.markdown(f"""
    **Macro Regime:** <span style='color:{macro_color}; font-size: 20px; font-weight:bold'>{macro_status}</span>  
    _{macro_advice}_
""", unsafe_allow_html=True)

if macro_status.startswith("EXTREME"):
    st.error("🚨 ALARM: MACRO CONDITIONS ARE HOSTILE. INVESTING LONG IS NOT RECOMMENDED.")

# Fetch Data
if st.button("Analyze Market"):
    with st.spinner('Downloading and crunching numbers...'):
        stock_data = get_data(ticker_list)
        
        bullish_stocks = []
        bearish_stocks = []
        
        for ticker, df in stock_data.items():
            if len(df) < 200: continue
            
            curr_price = df['Close'].iloc[-1]
            signal, reason, structure = detect_patterns(df)
            rsi = df['RSI'].iloc[-1]
            sma200 = df['SMA_200'].iloc[-1]
            
            # Logic Combiner
            
            # SCENARIO 1: BULLISH FILTER
            # Rules: Bullish Signal OR (RSI Oversold + Uptrend)
            is_bullish = (signal == "Bullish") or (rsi < 35 and structure == "Uptrend")
            # Filter out longs if Macro is Extreme Bearish
            if macro_status.startswith("EXTREME") and is_bullish:
                is_bullish = False # Vetoed by Macro
                
            if is_bullish:
                bullish_stocks.append({
                    "Ticker": ticker,
                    "Price": f"${curr_price:.2f}",
                    "Structure": structure,
                    "RSI": f"{rsi:.1f}",
                    "Reason": reason
                })

            # SCENARIO 2: BEARISH FILTER
            # Rules: Bearish Signal OR (RSI Overbought + Downtrend) OR Break of Structure
            is_bearish = (signal == "Bearish") or (rsi > 70) or (curr_price < sma200 and df['Close'].iloc[-2] > sma200)
            
            if is_bearish:
                bearish_stocks.append({
                    "Ticker": ticker,
                    "Price": f"${curr_price:.2f}",
                    "Structure": structure,
                    "RSI": f"{rsi:.1f}",
                    "Reason": reason if signal == "Bearish" else "Overextended / Structure Break"
                })

        # ==========================================
        # 5. DISPLAY RESULTS
        # ==========================================
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 Bullish Opportunities")
            if bullish_stocks:
                df_bull = pd.DataFrame(bullish_stocks)
                st.dataframe(df_bull, hide_index=True)
                
                # Chart for top Bullish pick
                top_pick = df_bull.iloc[0]['Ticker']
                st.markdown(f"**Deep Dive: {top_pick}**")
                
                # Plotly Chart
                df_chart = stock_data[top_pick].tail(100)
                fig = go.Figure(data=[go.Candlestick(x=df_chart.index,
                                open=df_chart['Open'],
                                high=df_chart['High'],
                                low=df_chart['Low'],
                                close=df_chart['Close'])])
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'))
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_200'], line=dict(color='blue', width=1), name='SMA 200'))
                fig.update_layout(title=f"{top_pick} Price Action", height=400, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No strong bullish setups found.")

        with col2:
            st.subheader("🔴 Bearish / Sell Signals")
            if bearish_stocks:
                df_bear = pd.DataFrame(bearish_stocks)
                st.dataframe(df_bear, hide_index=True)
                
                # Chart for top Bearish pick
                top_bear = df_bear.iloc[0]['Ticker']
                st.markdown(f"**Deep Dive: {top_bear}**")
                
                # Plotly Chart
                df_chart_b = stock_data[top_bear].tail(100)
                fig_b = go.Figure(data=[go.Candlestick(x=df_chart_b.index,
                                open=df_chart_b['Open'],
                                high=df_chart_b['High'],
                                low=df_chart_b['Low'],
                                close=df_chart_b['Close'])])
                fig_b.add_trace(go.Scatter(x=df_chart_b.index, y=df_chart_b['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'))
                fig_b.add_trace(go.Scatter(x=df_chart_b.index, y=df_chart_b['SMA_200'], line=dict(color='blue', width=1), name='SMA 200'))
                fig_b.update_layout(title=f"{top_bear} Price Action", height=400, template="plotly_dark")
                st.plotly_chart(fig_b, use_container_width=True)
            else:
                st.info("No strong bearish setups found.")

else:
    st.info("Adjust settings in the sidebar and click 'Analyze Market' to start.")

# Disclaimer
st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes only. It uses algorithmic rules to identify patterns but cannot predict the future. Financial markets involve risk.")
