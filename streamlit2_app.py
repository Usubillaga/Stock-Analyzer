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
    /* Print Handling */
    @media print {
        .stButton, .stExpander, header, footer, .stSidebar, [data-testid="stSidebar"], .css-18e3th9 { display: none !important; }
        .block-container { padding: 0.5rem 1rem !important; }
        .rec-badge, .metric-card, .val-box, .tech-box { border: 1px solid #ccc !important; box-shadow: none !important; }
        body { -webkit-print-color-adjust: exact; }
    }
    
    /* Card Styling */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #666;
        letter-spacing: 0.5px;
        margin-bottom: 2px;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #111;
    }
    
    /* Valuation Box */
    .val-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2962FF;
        margin-bottom: 15px;
    }
    
    /* Technical Box */
    .tech-box {
        background: #fff;
        padding: 10px;
        border: 1px solid #eee;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-bottom: 15px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    
    /* Badges & Colors */
    .val-pos { color: #008000; }
    .val-neg { color: #d32f2f; }
    .val-neu { color: #f57c00; }
    .pattern-tag {
        display: inline-block; padding: 2px 6px; border-radius: 4px;
        background: #e3f2fd; color: #1565c0; font-weight: bold; font-size: 0.75rem; margin-right: 4px;
    }
    .rec-badge {
        font-size: 1.4rem;
        font-weight: 800;
        padding: 8px 16px;
        border-radius: 50px;
        display: inline-block;
        border: 2px solid;
        margin-bottom: 15px;
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

# --- 1. PATTERN RECOGNITION (Updated) ---
def find_patterns(df, lookback_days=120):
    patterns = []
    subset = df.iloc[-lookback_days:].copy()
    if len(subset) < 20: return [], subset
    
    # --- MACRO PATTERNS (Geometric) ---
    # Find Pivots
    subset['min'] = subset.iloc[argrelextrema(subset.Close.values, np.less_equal, order=5)[0]]['Close']
    subset['max'] = subset.iloc[argrelextrema(subset.Close.values, np.greater_equal, order=5)[0]]['Close']
    
    peaks = subset[subset['max'].notna()]['max']
    troughs = subset[subset['min'].notna()]['min']
    
    # Logic for Patterns
    if len(peaks) >= 3:
        p1, p2, p3 = peaks.iloc[-3], peaks.iloc[-2], peaks.iloc[-1]
        if p2 > p1 and p2 > p3 and abs(p1-p3)/p1 < 0.05: patterns.append("Head & Shoulders")
    
    if len(troughs) >= 3:
        t1, t2, t3 = troughs.iloc[-3], troughs.iloc[-2], troughs.iloc[-1]
        if t2 < t1 and t2 < t3 and abs(t1-t3)/t1 < 0.05: patterns.append("Inv. Head & Shoulders")

    if len(peaks) >= 2:
        if abs(peaks.iloc[-1] - peaks.iloc[-2])/peaks.iloc[-1] < 0.015: patterns.append("Double Top")
        
    if len(troughs) >= 2:
        if abs(troughs.iloc[-1] - troughs.iloc[-2])/troughs.iloc[-1] < 0.015: patterns.append("Double Bottom")

    # Wedge Logic
    if len(peaks) >= 4 and len(troughs) >= 4:
        x_peaks = np.arange(len(peaks))[-4:]
        slope_res, _, _, _, _ = linregress(x_peaks, peaks.iloc[-4:].values)
        x_troughs = np.arange(len(troughs))[-4:]
        slope_sup, _, _, _, _ = linregress(x_troughs, troughs.iloc[-4:].values)
        
        if slope_res > 0 and slope_sup > 0 and slope_sup > slope_res: patterns.append("Rising Wedge (Bearish)")
        elif slope_res < 0 and slope_sup < 0 and abs(slope_res) > abs(slope_sup): patterns.append("Falling Wedge (Bullish)")

    # --- MICRO PATTERNS (Candlesticks) ---
    # Analyze the very last candle for single candle patterns
    last = subset.iloc[-1]
    prev = subset.iloc[-2]
    
    body = abs(last['Close'] - last['Open'])
    rng = last['High'] - last['Low']
    
    # 1. Doji
    # Body is less than 10% of total range
    if rng > 0 and body <= (rng * 0.1):
        patterns.append("Doji")
        
    # 2. Hammer / Hanging Man
    # Small body near top, long lower wick (at least 2x body)
    lower_wick = min(last['Open'], last['Close']) - last['Low']
    upper_wick = last['High'] - max(last['Open'], last['Close'])
    
    if body > 0 and lower_wick >= (2 * body) and upper_wick <= (body * 0.5):
        # Identify context: Hammer (Bullish) if detected at low, Hanging Man if at high
        # Simplistic check: is price below SMA50?
        if last['Close'] < subset['SMA50'].iloc[-1]:
            patterns.append("Hammer")
        else:
            patterns.append("Hanging Man")

    # 3. Shooting Star / Inverted Hammer
    # Small body near bottom, long upper wick
    if body > 0 and upper_wick >= (2 * body) and lower_wick <= (body * 0.5):
        if last['Close'] > subset['SMA50'].iloc[-1]:
            patterns.append("Shooting Star")
        else:
            patterns.append("Inverted Hammer")
            
    # 4. Engulfing Patterns (Requires 2 candles)
    # Bullish Engulfing: Prev red, Curr Green, Curr body engulfs Prev body
    if prev['Close'] < prev['Open'] and last['Close'] > last['Open']: # Red then Green
        if last['Open'] <= prev['Close'] and last['Close'] >= prev['Open']:
            patterns.append("Bullish Engulfing")
            
    # Bearish Engulfing: Prev Green, Curr Red, Curr body engulfs Prev body
    if prev['Close'] > prev['Open'] and last['Close'] < last['Open']: # Green then Red
        if last['Open'] >= prev['Close'] and last['Close'] <= prev['Open']:
            patterns.append("Bearish Engulfing")

    return patterns, subset

# --- 2. FUNDAMENTAL MODELS ---
def calculate_dcf(info, cashflow, shares_out):
    try:
        # FCF Calculation
        if 'Free Cash Flow' in cashflow.index: fcf = cashflow.loc['Free Cash Flow'].iloc[0]
        else: fcf = cashflow.loc['Operating Cash Flow'].iloc[0] + cashflow.loc['Capital Expenditure'].iloc[0]
        
        if fcf < 0: return None 

        growth = min(info.get('revenueGrowth', 0.05), 0.15) 
        if growth < 0: growth = 0.02
        discount = 0.09
        perp = 0.025
        
        future_fcf = [fcf * ((1+growth)**i) for i in range(1,6)]
        terminal = future_fcf[-1] * (1+perp) / (discount-perp)
        
        dcf_val = sum([f/((1+discount)**(i+1)) for i, f in enumerate(future_fcf)]) + (terminal/((1+discount)**5))
        return dcf_val / shares_out
    except: return None

def calculate_piotroski(bs, is_, cf):
    try:
        if bs.shape[1] < 2: return 5
        score = 0
        # Profit
        score += 1 if is_.loc['Net Income'].iloc[0] > 0 else 0
        score += 1 if cf.loc['Operating Cash Flow'].iloc[0] > 0 else 0
        try: score += 1 if (is_.loc['Net Income'].iloc[0] / bs.loc['Total Assets'].iloc[0]) > 0 else 0
        except: pass
        score += 1 if cf.loc['Operating Cash Flow'].iloc[0] > is_.loc['Net Income'].iloc[0] else 0
        # Leverage
        try: score += 1 if bs.loc['Long Term Debt'].iloc[0] <= bs.loc['Long Term Debt'].iloc[1] else 0
        except: pass
        try: score += 1 if (bs.loc['Current Assets'].iloc[0]/bs.loc['Current Liabilities'].iloc[0]) > (bs.loc['Current Assets'].iloc[1]/bs.loc['Current Liabilities'].iloc[1]) else 0
        except: pass
        # Efficiency
        try: score += 1 if bs.loc['Ordinary Shares Number'].iloc[0] <= bs.loc['Ordinary Shares Number'].iloc[1] else 0
        except: pass
        try: score += 1 if (is_.loc['Total Revenue'].iloc[0] - is_.loc['Cost Of Revenue'].iloc[0])/is_.loc['Total Revenue'].iloc[0] > (is_.loc['Total Revenue'].iloc[1] - is_.loc['Cost Of Revenue'].iloc[1])/is_.loc['Total Revenue'].iloc[1] else 0
        except: pass
        return score
    except: return 5

def calculate_altman(bs, is_, info):
    try:
        A = (bs.loc['Current Assets'].iloc[0] - bs.loc['Current Liabilities'].iloc[0]) / bs.loc['Total Assets'].iloc[0]
        B = bs.loc['Retained Earnings'].iloc[0] / bs.loc['Total Assets'].iloc[0] if 'Retained Earnings' in bs.index else 0
        C = (is_.loc['EBIT'].iloc[0] if 'EBIT' in is_.index else is_.loc['Net Income'].iloc[0]) / bs.loc['Total Assets'].iloc[0]
        D = info.get('marketCap',0) / bs.loc['Total Liabilities Net Minority Interest'].iloc[0]
        E = is_.loc['Total Revenue'].iloc[0] / bs.loc['Total Assets'].iloc[0]
        return 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
    except: return 3.0 

# --- 3. DATA & UI ---
@st.cache_data(ttl=3600)
def get_data(symbol):
    try:
        time.sleep(0.3)
        t = yf.Ticker(symbol)
        return t.info, t.balance_sheet, t.financials, t.cashflow, t.history(period="2y"), t.news
    except: return None, None, None, None, None, None

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
if not st.session_state.print_mode:
    with st.sidebar:
        st.header("📊 Settings")
        ticker = st.text_input("Ticker Symbol", "NVDA").upper()
        if st.button("🖨️ Printer Friendly Mode"): toggle_print()
else:
    c1, _ = st.columns([1, 10])
    with c1:
        if st.button("← Back"): toggle_print()
    ticker = st.session_state.get('ticker_val', "NVDA")

if 'ticker' not in locals(): ticker = "NVDA"

# FETCH
info, bs, is_, cf, hist, news = get_data(ticker)
if not info or hist.empty:
    st.error("Data restricted or ticker invalid.")
    st.stop()

# CALCULATE FUNDAMENTALS
dcf = calculate_dcf(info, cf, info.get('sharesOutstanding', 1))
graham = (22.5 * info.get('trailingEps',0) * info.get('bookValue',0))**0.5 if info.get('trailingEps',0)>0 else 0
piotroski = calculate_piotroski(bs, is_, cf)
altman = calculate_altman(bs, is_, info)
current_price = hist['Close'].iloc[-1]
margin_safety = ((dcf - current_price) / dcf * 100) if dcf else None

# CALCULATE TECHNICALS
hist['SMA50'] = hist['Close'].rolling(50).mean()
hist['SMA100'] = hist['Close'].rolling(100).mean() # Added SMA 100
hist['SMA200'] = hist['Close'].rolling(200).mean()
delta = hist['Close'].diff()
rs = (delta.where(delta>0,0).rolling(14).mean()) / (-delta.where(delta<0,0).rolling(14).mean())
hist['RSI'] = 100 - (100/(1+rs))
patterns_found, subset = find_patterns(hist)

# SCORING
rec_score = 0
if dcf and current_price < dcf: rec_score += 1
if piotroski >= 7: rec_score += 1
if hist['Close'].iloc[-1] > hist['SMA200'].iloc[-1]: rec_score += 1
if info.get('pegRatio', 5) < 1.5: rec_score += 1
if "Bullish" in str(patterns_found) or "Hammer" in str(patterns_found): rec_score += 1
if "Bearish" in str(patterns_found) or "Shooting" in str(patterns_found): rec_score -= 1

if rec_score >= 3: badge, b_cls = "STRONG BUY", "badge-buy"
elif rec_score >= 1: badge, b_cls = "HOLD", "badge-hold"
else: badge, b_cls = "SELL", "badge-sell"

# --- LAYOUT ---
st.markdown(f"## {ticker} • {info.get('longName')}")
st.markdown(f"**{info.get('sector')}** | {datetime.now().strftime('%Y-%m-%d')}")
st.divider()

col_L, col_R = st.columns([1, 2])

# LEFT COLUMN: FUNDAMENTALS & VISUAL PROFILE
with col_L:
    st.markdown(f'<div class="rec-badge {b_cls}">{badge}</div>', unsafe_allow_html=True)
    
    # Valuation Box
    mos_str = f"{margin_safety:.1f}%" if margin_safety else "N/A"
    mos_col = "#008000" if margin_safety and margin_safety > 0 else "#d32f2f"
    
    st.markdown(f"""
    <div class="val-box">
        <div style="display:flex; justify-content:space-between; align-items:end;">
            <div>
                <span style="font-size:0.7rem; font-weight:bold; color:#555;">DCF INTRINSIC VALUE</span><br>
                <span style="font-size:1.5rem; font-weight:800; color:#333;">${dcf:.2f}</span>
            </div>
            <div style="text-align:right;">
                <span style="font-size:0.7rem; font-weight:bold; color:#555;">MARGIN OF SAFETY</span><br>
                <span style="font-size:1.4rem; font-weight:800; color:{mos_col};">{mos_str}</span>
            </div>
        </div>
        <hr style="margin:8px 0; border-top:1px solid #ddd;">
        <div style="font-size:0.85rem; display:flex; justify-content:space-between;">
            <span>Current: <b>${current_price:.2f}</b></span>
            <span>Graham No: <b>${graham:.2f}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics Grid (Added Growth Rate here)
    st.markdown("### 🏗️ Fundamentals")
    c1, c2 = st.columns(2)
    with c1:
        render_metric("Revenue Growth", info.get('revenueGrowth'), is_percent=True, comparison=0.10)
        render_metric("P/E Ratio", info.get('trailingPE'), comparison=25, invert=True)
        render_metric("Piotroski F", piotroski, fmt="{:.0f}", comparison=6)
        render_metric("ROE", info.get('returnOnEquity'), is_percent=True, comparison=0.15)
    with c2:
        render_metric("PEG Ratio", info.get('pegRatio'), comparison=1.5, invert=True)
        render_metric("Altman Z", altman, fmt="{:.2f}", comparison=2.99)
        render_metric("Debt/Equity", info.get('debtToEquity'), fmt="{:.1f}", comparison=100, invert=True)
        render_metric("Profit Margin", info.get('profitMargins'), is_percent=True, comparison=0.10)

    # 360 RADAR CHART
    st.markdown("### 🎯 360° Analysis")
    vals = [
        min(100, (1/(info.get('trailingPE', 50)+1))*2000), 
        min(100, info.get('revenueGrowth', 0)*300), 
        min(100, info.get('returnOnEquity', 0)*300), 
        min(100, (altman/4)*100), 
        min(100, info.get('grossMargins', 0)*150)
    ]
    radar_cats = ['Value','Growth','Profit','Health','Moat']
    fig_r = go.Figure(go.Scatterpolar(r=vals, theta=radar_cats, fill='toself', name=ticker))
    fig_r.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])), 
        margin=dict(l=20, r=20, t=20, b=20), 
        height=250,
        showlegend=False
    )
    st.plotly_chart(fig_r, use_container_width=True)

# RIGHT COLUMN: TECHNICALS & PATTERNS
with col_R:
    st.markdown("### 📐 Technical Structure")
    
    # Chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'), row=1, col=1)
    
    # ADDED SMA 50, 100, 200 traces
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA50'], line=dict(color='orange', width=1), name='SMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA100'], line=dict(color='purple', width=1), name='SMA 100'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA200'], line=dict(color='#2962FF', width=1.5), name='SMA 200'), row=1, col=1)
    
    if 'max' in subset.columns:
        peaks = subset[subset['max'].notna()]
        troughs = subset[subset['min'].notna()]
        fig.add_trace(go.Scatter(x=peaks.index, y=peaks['max'], mode='markers', marker=dict(color='red', size=6, symbol='triangle-down'), name='Pivot High'), row=1, col=1)
        fig.add_trace(go.Scatter(x=troughs.index, y=troughs['min'], mode='markers', marker=dict(color='green', size=6, symbol='triangle-up'), name='Pivot Low'), row=1, col=1)

    colors = ['#ef5350' if row['Open'] - row['Close'] >= 0 else '#26a69a' for index, row in hist.iterrows()]
    fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

    fig.update_layout(height=550, margin=dict(l=0,r=0,t=10,b=0), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    pat_str = "".join([f'<span class="pattern-tag">{p}</span>' for p in patterns_found]) if patterns_found else "No Chart Patterns Detected."
    trend = "Bullish" if hist['Close'].iloc[-1] > hist['SMA200'].iloc[-1] else "Bearish"
    
    st.markdown(f"""
    <div class="tech-box">
        <div style="margin-bottom:5px;"><b>Detected Patterns:</b> {pat_str}</div>
        <div style="display:flex; justify-content:space-between; border-top:1px solid #eee; padding-top:5px;">
            <span>Long Trend: <b>{trend}</b></span>
            <span>RSI (14): <b>{hist['RSI'].iloc[-1]:.1f}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
if not st.session_state.print_mode:
    with st.expander("📘 Guide: How to Read This Report"):
        st.markdown("""
        * **360 Analysis:** A visual profile of the stock's strengths (Scale 0-100).
        * **Margin of Safety:** Difference between Intrinsic Value (DCF) and Price.
        * **Patterns:** Automatic detection of Geometric (Head & Shoulders, Wedges) and Candlestick (Doji, Hammer, Engulfing) patterns.
        """)
