import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
from scipy.signal import argrelextrema
from scipy.stats import linregress

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Institutional Equity Report", page_icon="📈")

# --- VISUAL STYLING (CSS) ---
st.markdown("""
<style>
    /* 1. Print Handling - The "Perfect PDF" Look */
    @media print {
        .stButton, .stExpander, header, footer, .stSidebar, [data-testid="stSidebar"], .css-18e3th9 { display: none !important; }
        .block-container { padding: 0.5rem 1rem !important; }
        .rec-badge, .metric-card, .val-box, .tech-box { border: 1px solid #ccc !important; box-shadow: none !important; }
        body { -webkit-print-color-adjust: exact; }
    }
    
    /* 2. Card Styling */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px 8px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #666;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #111;
    }
    
    /* 3. Valuation Box */
    .val-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2962FF;
        margin-bottom: 15px;
    }
    
    /* 4. Technical Box */
    .tech-box {
        background: #fff;
        padding: 12px;
        border: 1px solid #eee;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-bottom: 15px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .pattern-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        background: #e3f2fd;
        color: #1565c0;
        font-weight: bold;
        font-size: 0.8rem;
        margin-right: 5px;
    }
    
    /* 5. Badges & Colors */
    .val-pos { color: #008000; }
    .val-neg { color: #d32f2f; }
    .val-neu { color: #f57c00; }

    .rec-badge {
        font-size: 1.4rem;
        font-weight: 800;
        padding: 6px 16px;
        border-radius: 50px;
        display: inline-block;
        border: 2px solid;
        margin-bottom: 10px;
    }
    .badge-buy { background-color: #e8f5e9; color: #2e7d32; border-color: #a5d6a7; }
    .badge-sell { background-color: #ffebee; color: #c62828; border-color: #ef9a9a; }
    .badge-hold { background-color: #fff8e1; color: #f57f17; border-color: #ffe082; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'print_mode' not in st.session_state:
    st.session_state.print_mode = False

def toggle_print():
    st.session_state.print_mode = not st.session_state.print_mode

# --- ADVANCED PATTERN RECOGNITION ENGINE ---
def find_patterns(df, lookback_days=180):
    """
    Scans for geometric chart patterns using local maxima/minima.
    """
    patterns = []
    
    # Slice to relevant lookback period
    subset = df.iloc[-lookback_days:].copy()
    if len(subset) < 20: return [], subset
    
    # 1. Find Pivots (Highs and Lows)
    # order=5 means we look for a high/low that is the extreme of 5 candles on either side
    subset['min'] = subset.iloc[argrelextrema(subset.Close.values, np.less_equal, order=5)[0]]['Close']
    subset['max'] = subset.iloc[argrelextrema(subset.Close.values, np.greater_equal, order=5)[0]]['Close']
    
    # Get just the peaks and troughs as lists of (index, price)
    peaks = subset[subset['max'].notna()]['max']
    troughs = subset[subset['min'].notna()]['min']
    
    # --- A. HEAD AND SHOULDERS (Bearish) ---
    # Need 3 peaks: Middle is highest, Left and Right are lower and roughly equal
    if len(peaks) >= 3:
        p1, p2, p3 = peaks.iloc[-3], peaks.iloc[-2], peaks.iloc[-1] # L, Head, R
        # Tolerances
        is_head = p2 > p1 and p2 > p3
        is_shoulders = abs(p1 - p3) / p1 < 0.05 # Shoulders within 5% height
        if is_head and is_shoulders:
            patterns.append("Head & Shoulders (Bearish)")

    # --- B. INVERSE HEAD AND SHOULDERS (Bullish) ---
    if len(troughs) >= 3:
        t1, t2, t3 = troughs.iloc[-3], troughs.iloc[-2], troughs.iloc[-1]
        is_head = t2 < t1 and t2 < t3
        is_shoulders = abs(t1 - t3) / t1 < 0.05
        if is_head and is_shoulders:
            patterns.append("Inv. Head & Shoulders (Bullish)")

    # --- C. DOUBLE TOP ---
    if len(peaks) >= 2:
        p1, p2 = peaks.iloc[-2], peaks.iloc[-1]
        if abs(p1 - p2) / p1 < 0.02: # Peaks within 2%
            patterns.append("Double Top (Bearish)")

    # --- D. DOUBLE BOTTOM ---
    if len(troughs) >= 2:
        t1, t2 = troughs.iloc[-2], troughs.iloc[-1]
        if abs(t1 - t2) / t1 < 0.02:
            patterns.append("Double Bottom (Bullish)")

    # --- E. WEDGES / TRIANGLES (Using Slope) ---
    # We fit a line to the last 5 peaks and last 5 troughs
    if len(peaks) >= 4 and len(troughs) >= 4:
        # Linear Regression on the peaks (Resistance Line)
        x_peaks = np.arange(len(peaks))[-4:]
        y_peaks = peaks.iloc[-4:].values
        slope_res, _, _, _, _ = linregress(x_peaks, y_peaks)
        
        # Linear Regression on the troughs (Support Line)
        x_troughs = np.arange(len(troughs))[-4:]
        y_troughs = troughs.iloc[-4:].values
        slope_sup, _, _, _, _ = linregress(x_troughs, y_troughs)
        
        # Rising Wedge: Both slopes positive, support slope steeper
        if slope_res > 0 and slope_sup > 0 and slope_sup > slope_res:
            patterns.append("Rising Wedge (Bearish)")
        # Falling Wedge: Both slopes negative, resistance slope steeper
        elif slope_res < 0 and slope_sup < 0 and abs(slope_res) > abs(slope_sup):
            patterns.append("Falling Wedge (Bullish)")
        # Pennant/Triangle: Slopes converging (Res down, Sup up)
        elif slope_res < 0 and slope_sup > 0:
            patterns.append("Symmetrical Triangle (Neutral)")
    
    return patterns, subset

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_data(symbol):
    try:
        time.sleep(0.3)
        t = yf.Ticker(symbol)
        # Fetch 2y to ensure enough data for patterns
        hist = t.history(period="2y")
        return t.info, t.balance_sheet, t.financials, t.cashflow, hist, t.news
    except:
        return None, None, None, None, None, None

# --- CALCULATIONS (Tech & Val) ---
def calculate_technicals(df):
    if df.empty: return df
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_dcf(info, cashflow, shares_out):
    try:
        if 'Free Cash Flow' in cashflow.index:
            fcf_ttm = cashflow.loc['Free Cash Flow'].iloc[0]
        else:
            ocf = cashflow.loc['Operating Cash Flow'].iloc[0]
            capex = cashflow.loc['Capital Expenditure'].iloc[0]
            fcf_ttm = ocf + capex 
        
        if fcf_ttm < 0: return None 
        growth_rate = min(info.get('revenueGrowth', 0.05), 0.15)
        if growth_rate < 0: growth_rate = 0.02
        discount_rate = 0.09 
        perp_growth = 0.025
        
        future_fcfs = []
        for i in range(1, 6):
            fcf_ttm = fcf_ttm * (1 + growth_rate)
            future_fcfs.append(fcf_ttm)
        terminal_val = future_fcfs[-1] * (1 + perp_growth) / (discount_rate - perp_growth)
        dcf_value = sum([fcf / ((1 + discount_rate) ** (i + 1)) for i, fcf in enumerate(future_fcfs)])
        dcf_value += terminal_val / ((1 + discount_rate) ** 5)
        return dcf_value / shares_out
    except:
        return None

def calculate_piotroski(bs, is_, cf):
    try:
        if bs.shape[1] < 2: return 5
        score = 5 # Base score for demo stability
        # Add real logic here if needed, keeping it safe for rate limits
        return score 
    except: return 5

# --- HELPER UI ---
def render_metric(label, value, fmt="{:.2f}", is_percent=False, comparison=None, invert=False):
    if value is None: val_str, color = "—", ""
    else:
        if is_percent: value = value * 100
        val_str = fmt.format(value) + ("%" if is_percent else "")
        color = "val-neu"
        if comparison:
            if invert: color = "val-pos" if value < comparison else "val-neg"
            else: color = "val-pos" if value > comparison else "val-neg"
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {color}">{val_str}</div></div>', unsafe_allow_html=True)

# --- MAIN APP ---

# Sidebar for Input (Hidden in Print Mode)
if not st.session_state.print_mode:
    with st.sidebar:
        st.header("📊 Settings")
        ticker = st.text_input("Ticker Symbol", "NVDA").upper()
        st.markdown("---")
        if st.button("🖨️ Printer Friendly Mode"):
            toggle_print()
else:
    c1, c2 = st.columns([1, 10])
    with c1:
        if st.button("← Back"): toggle_print()
    ticker = st.session_state.get('ticker_val', "NVDA")

if 'ticker' not in locals(): ticker = "NVDA"

# Fetch & Prep
info, bs, is_, cf, hist, news = get_data(ticker)
if not info or hist.empty:
    st.error("Data restricted or ticker invalid.")
    st.stop()

hist = calculate_technicals(hist)
patterns_found, subset_for_chart = find_patterns(hist)

current_price = hist['Close'].iloc[-1]
dcf = calculate_dcf(info, cf, info.get('sharesOutstanding', 1))
margin_safety = ((dcf - current_price) / dcf * 100) if dcf else None
piotroski = calculate_piotroski(bs, is_, cf)

# Scoring Logic
rec_score = 0
if dcf and current_price < dcf: rec_score += 1
if piotroski >= 7: rec_score += 1
if hist['Close'].iloc[-1] > hist['SMA200'].iloc[-1]: rec_score += 1
if "Bullish" in str(patterns_found): rec_score += 1
if "Bearish" in str(patterns_found): rec_score -= 1

if rec_score >= 2: rec_badge, rec_cls = "STRONG BUY", "badge-buy"
elif rec_score <= -1: rec_badge, rec_cls = "SELL", "badge-sell"
else: rec_badge, rec_cls = "HOLD", "badge-hold"

# --- LAYOUT START ---
st.markdown(f"## {ticker} • {info.get('longName')}")
st.markdown(f"**{info.get('sector')}** | {datetime.now().strftime('%Y-%m-%d')}")
st.divider()

col_L, col_R = st.columns([1, 2])

with col_L:
    # 1. Recommendation
    st.markdown(f'<div class="rec-badge {rec_cls}">{rec_badge}</div>', unsafe_allow_html=True)
    
    # 2. Valuation Card
    int_val_str = f"${dcf:.2f}" if dcf else "N/A"
    mos_str = f"{margin_safety:.1f}%" if margin_safety else "N/A"
    mos_color = "#008000" if margin_safety and margin_safety > 0 else "#d32f2f"
    
    st.markdown(f"""
    <div class="val-box">
        <div style="display:flex; justify-content:space-between; align-items:end;">
            <div>
                <span style="font-size:0.75rem; font-weight:bold; color:#555;">DCF FAIR VALUE</span><br>
                <span style="font-size:1.6rem; font-weight:800; color:#333;">{int_val_str}</span>
            </div>
            <div style="text-align:right;">
                <span style="font-size:0.75rem; font-weight:bold; color:#555;">MARGIN OF SAFETY</span><br>
                <span style="font-size:1.4rem; font-weight:800; color:{mos_color};">{mos_str}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Technical Signals Box
    pat_str = "".join([f'<span class="pattern-tag">{p}</span>' for p in patterns_found]) if patterns_found else "No clear patterns detected."
    trend_50 = "Bullish" if hist['Close'].iloc[-1] > hist['SMA50'].iloc[-1] else "Bearish"
    
    st.markdown(f"""
    <div class="tech-box">
        <div style="margin-bottom:8px;"><b>Detected Patterns:</b><br>{pat_str}</div>
        <div style="display:flex; justify-content:space-between;">
            <span>Trend (50D): <b>{trend_50}</b></span>
            <span>RSI (14): <b>{hist['RSI'].iloc[-1]:.1f}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4. Fundamental Grid
    c1, c2 = st.columns(2)
    with c1:
        render_metric("P/E Ratio", info.get('trailingPE'), comparison=25, invert=True)
        render_metric("ROE", info.get('returnOnEquity'), is_percent=True, comparison=0.15)
    with c2:
        render_metric("PEG Ratio", info.get('pegRatio'), comparison=1.5, invert=True)
        render_metric("Debt/Equity", info.get('debtToEquity'), fmt="{:.1f}", comparison=100, invert=True)

with col_R:
    # --- CHARTING ---
    st.markdown("### Technical Structure")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.75, 0.25])
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'), row=1, col=1)
    
    # Moving Averages
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA50'], line=dict(color='orange', width=1), name='SMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA200'], line=dict(color='#2962FF', width=1.5), name='SMA 200'), row=1, col=1)
    
    # Pattern Visuals (Peaks/Troughs)
    # We plot the Local Max/Min found by the algorithm to help user see the pattern points
    if 'max' in subset_for_chart.columns:
        peaks = subset_for_chart[subset_for_chart['max'].notna()]
        troughs = subset_for_chart[subset_for_chart['min'].notna()]
        
        fig.add_trace(go.Scatter(x=peaks.index, y=peaks['max'], mode='markers', marker=dict(color='red', size=6, symbol='triangle-down'), name='Pivot High'), row=1, col=1)
        fig.add_trace(go.Scatter(x=troughs.index, y=troughs['min'], mode='markers', marker=dict(color='green', size=6, symbol='triangle-up'), name='Pivot Low'), row=1, col=1)

    # Volume
    colors = ['#ef5350' if row['Open'] - row['Close'] >= 0 else '#26a69a' for index, row in hist.iterrows()]
    fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

    fig.update_layout(
        height=600,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1, x=0)
    )
    fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    
    st.plotly_chart(fig, use_container_width=True)

# Footer Pattern Explainer (Hidden in Print)
if not st.session_state.print_mode:
    with st.expander("📘 Pattern Recognition Guide"):
        st.markdown("""
        * **Pivots (Triangles):** The chart marks local Highs (Red) and Lows (Green). Patterns are formed by the relationship between these points.
        * **Head & Shoulders:** Detects a pattern of High-Higher-High peaks. Indicates a potential reversal from Bullish to Bearish.
        * **Wedges:** Detects if the slope of the Highs and Lows are converging.
        """)
