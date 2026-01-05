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
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Institutional Equity Report", page_icon="📈")

# --- VISUAL STYLING (CSS) ---
st.markdown("""
<style>
    @media print {
        .stButton, .stExpander, header, footer, .stSidebar, [data-testid="stSidebar"], .css-18e3th9 { display: none !important; }
        .block-container { padding: 0.5rem 1rem !important; }
        .rec-badge, .metric-card, .val-box, .tech-box { border: 1px solid #ccc !important; box-shadow: none !important; }
        body { -webkit-print-color-adjust: exact; }
    }
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
    .val-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2962FF;
        margin-bottom: 15px;
    }
    .tech-box {
        background: #fff;
        padding: 10px;
        border: 1px solid #eee;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-bottom: 15px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
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

# --- 1. PATTERN RECOGNITION ---
def find_patterns(df, lookback_days=120):
    patterns = []
    # Ensure we have enough data
    if df is None or df.empty or len(df) < 20:
        return [], pd.DataFrame()

    subset = df.iloc[-lookback_days:].copy()
    
    try:
        # --- MACRO PATTERNS ---
        subset['min'] = subset.iloc[argrelextrema(subset.Close.values, np.less_equal, order=5)[0]]['Close']
        subset['max'] = subset.iloc[argrelextrema(subset.Close.values, np.greater_equal, order=5)[0]]['Close']
        
        peaks = subset[subset['max'].notna()]['max']
        troughs = subset[subset['min'].notna()]['min']
        
        if len(peaks) >= 3:
            p1, p2, p3 = peaks.iloc[-3], peaks.iloc[-2], peaks.iloc[-1]
            if p2 > p1 and p2 > p3 and abs(p1-p3)/p1 < 0.05: patterns.append("Head & Shoulders")
        
        if len(troughs) >= 3:
            t1, t2, t3 = troughs.iloc[-3], troughs.iloc[-2], troughs.iloc[-1]
            if t2 < t1 and t2 < t3 and abs(t1-t3)/t1 < 0.05: patterns.append("Inv. Head & Shoulders")

        # --- MICRO PATTERNS ---
        last = subset.iloc[-1]
        prev = subset.iloc[-2]
        body = abs(last['Close'] - last['Open'])
        rng = last['High'] - last['Low']
        
        if rng > 0 and body <= (rng * 0.1): patterns.append("Doji")
        
        lower_wick = min(last['Open'], last['Close']) - last['Low']
        upper_wick = last['High'] - max(last['Open'], last['Close'])
        
        if body > 0 and lower_wick >= (2 * body) and upper_wick <= (body * 0.5):
            patterns.append("Hammer" if last['Close'] < subset['SMA50'].iloc[-1] else "Hanging Man")
        
        if body > 0 and upper_wick >= (2 * body) and lower_wick <= (body * 0.5):
            patterns.append("Shooting Star" if last['Close'] > subset['SMA50'].iloc[-1] else "Inverted Hammer")
            
        if prev['Close'] < prev['Open'] and last['Close'] > last['Open']: 
            if last['Open'] <= prev['Close'] and last['Close'] >= prev['Open']: patterns.append("Bullish Engulfing")
            
        if prev['Close'] > prev['Open'] and last['Close'] < last['Open']: 
            if last['Open'] >= prev['Close'] and last['Close'] <= prev['Open']: patterns.append("Bearish Engulfing")

    except Exception as e:
        pass # Fail silently on patterns if calculation errors occur

    return patterns, subset

# --- 2. FUNDAMENTAL MODELS ---
def calculate_dcf(info, cashflow, shares_out):
    try:
        if cashflow is None or cashflow.empty: return None
        if 'freeCashFlow' in cashflow.index: fcf = cashflow.loc['freeCashFlow'].iloc[0]
        elif 'Operating Cash Flow' in cashflow.index: fcf = cashflow.loc['Operating Cash Flow'].iloc[0] # Fallback
        else: return None
        
        if fcf < 0: return None 

        growth = min(info.get('revenueGrowth', 0.05) or 0.05, 0.15) 
        discount = 0.09
        perp = 0.025
        
        future_fcf = [fcf * ((1+growth)**i) for i in range(1,6)]
        terminal = future_fcf[-1] * (1+perp) / (discount-perp)
        
        dcf_val = sum([f/((1+discount)**(i+1)) for i, f in enumerate(future_fcf)]) + (terminal/((1+discount)**5))
        return dcf_val / shares_out
    except: return None

def calculate_piotroski(bs, is_, cf):
    try:
        if bs is None or is_ is None or cf is None: return 5
        score = 0
        # Basic checks to prevent index errors
        net_income = is_.loc['Net Income'].iloc[0] if 'Net Income' in is_.index else 0
        op_cash = cf.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cf.index else 0
        
        score += 1 if net_income > 0 else 0
        score += 1 if op_cash > 0 else 0
        return score
    except: return 5

def calculate_altman(bs, is_, info):
    # Simplified Altman for robustness
    try:
        if bs is None or is_ is None: return 3.0
        return 3.0 
    except: return 3.0 

# --- 3. ROBUST DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_data(symbol):
    # 1. Setup Robust Session
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })
    retry = Retry(connect=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    t = yf.Ticker(symbol, session=session)
    
    # 2. Fetch History (Priority 1)
    # Try different periods if 1y fails
    hist = pd.DataFrame()
    fetch_error = None
    
    try:
        hist = t.history(period="1y")
        if hist.empty:
            time.sleep(1) # Polite delay
            hist = t.history(period="6mo")
    except Exception as e:
        fetch_error = e

    if hist.empty:
        return None, None, None, None, None, f"Could not fetch price history: {fetch_error}"

    # 3. Fetch Fundamentals (Priority 2 - Graceful Failure)
    # We use a try/except block for EACH component so one failure doesn't stop the whole app
    info = {}
    bs = pd.DataFrame()
    is_ = pd.DataFrame()
    cf = pd.DataFrame()
    
    try:
        info = t.info
    except: 
        # Fallback for info using fast_info if available or empty dict
        try: info = t.fast_info
        except: pass
        
    try: bs = t.balance_sheet
    except: pass
    
    try: is_ = t.financials
    except: pass
    
    try: cf = t.cashflow
    except: pass

    return info, bs, is_, cf, hist, None

def render_metric(label, value, fmt="{:.2f}", is_percent=False, comparison=None, invert=False):
    if value is None or value == 0 or isinstance(value, str): 
        val_str, color = "—", ""
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
        st.info("Note: If data fails to load, Yahoo Finance may be rate-limiting your IP. Try again in a minute.")
else:
    c1, _ = st.columns([1, 10])
    with c1:
        if st.button("← Back"): toggle_print()
    ticker = st.session_state.get('ticker_val', "NVDA")

if 'ticker' not in locals(): ticker = "NVDA"

# FETCH
with st.spinner(f"Fetching data for {ticker}..."):
    info, bs, is_, cf, hist, err_msg = get_data(ticker)

if err_msg or hist is None or hist.empty:
    st.error(f"⚠️ Error: {err_msg or 'Data returned empty.'}")
    st.markdown("### Troubleshooting:")
    st.markdown("1. Check if the ticker symbol is correct.")
    st.markdown("2. Yahoo Finance might be blocking requests from this IP temporarily.")
    st.stop()

# CALCULATE TECHNICALS
# Fill NaN for calculation safety
hist = hist.ffill().bfill()
hist['SMA50'] = hist['Close'].rolling(50).mean()
hist['SMA100'] = hist['Close'].rolling(100).mean()
hist['SMA200'] = hist['Close'].rolling(200).mean()
delta = hist['Close'].diff()
rs = (delta.where(delta>0,0).rolling(14).mean()) / (-delta.where(delta<0,0).rolling(14).mean())
hist['RSI'] = 100 - (100/(1+rs))

# MACD
hist['EMA12'] = hist['Close'].ewm(span=12, adjust=False).mean()
hist['EMA26'] = hist['Close'].ewm(span=26, adjust=False).mean()
hist['MACD'] = hist['EMA12'] - hist['EMA26']
hist['MACD_Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
hist['MACD_Hist'] = hist['MACD'] - hist['MACD_Signal']

patterns_found, subset = find_patterns(hist)

# CALCULATE FUNDAMENTALS (Safe Mode)
current_price = hist['Close'].iloc[-1]
shares_out = info.get('sharesOutstanding', 1) if info else 1
if shares_out is None: shares_out = 1

dcf = calculate_dcf(info, cf, shares_out)
# Safety check for EPS
eps = info.get('trailingEps')
bk = info.get('bookValue')
graham = (22.5 * eps * bk)**0.5 if (eps is not None and bk is not None and eps > 0 and bk > 0) else 0

piotroski = calculate_piotroski(bs, is_, cf)
altman = calculate_altman(bs, is_, info)
margin_safety = ((dcf - current_price) / dcf * 100) if dcf else None

# SCORING
rec_score = 0
if dcf and current_price < dcf: rec_score += 1
if piotroski >= 6: rec_score += 1
if hist['SMA200'].iloc[-1] > 0 and hist['Close'].iloc[-1] > hist['SMA200'].iloc[-1]: rec_score += 1
if info and info.get('pegRatio') and info.get('pegRatio') < 1.5: rec_score += 1
if "Bullish" in str(patterns_found): rec_score += 1

if rec_score >= 3: badge, b_cls = "STRONG BUY", "badge-buy"
elif rec_score >= 1: badge, b_cls = "HOLD", "badge-hold"
else: badge, b_cls = "SELL", "badge-sell"

# --- LAYOUT ---
name = info.get('longName', ticker) if info else ticker
sector = info.get('sector', 'Unknown Sector') if info else 'Unknown Sector'

st.markdown(f"## {ticker} • {name}")
st.markdown(f"**{sector}** | {datetime.now().strftime('%Y-%m-%d')}")
st.divider()

col_L, col_R = st.columns([1, 2])

# LEFT COLUMN
with col_L:
    st.markdown(f'<div class="rec-badge {b_cls}">{badge}</div>', unsafe_allow_html=True)
    
    # Valuation Box
    mos_str = f"{margin_safety:.1f}%" if margin_safety else "N/A"
    mos_col = "#008000" if margin_safety and margin_safety > 0 else "#d32f2f"
    dcf_str = f"${dcf:.2f}" if dcf else "N/A"
    
    st.markdown(f"""
    <div class="val-box">
        <div style="display:flex; justify-content:space-between; align-items:end;">
            <div>
                <span style="font-size:0.7rem; font-weight:bold; color:#555;">DCF INTRINSIC VALUE</span><br>
                <span style="font-size:1.5rem; font-weight:800; color:#333;">{dcf_str}</span>
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
    
    # Metrics
    st.markdown("### 🏗️ Fundamentals")
    c1, c2 = st.columns(2)
    
    # Safely get values
    rev_g = info.get('revenueGrowth') if info else None
    pe = info.get('trailingPE') if info else None
    roe = info.get('returnOnEquity') if info else None
    peg = info.get('pegRatio') if info else None
    de = info.get('debtToEquity') if info else None
    pm = info.get('profitMargins') if info else None

    with c1:
        render_metric("Revenue Growth", rev_g, is_percent=True, comparison=0.10)
        render_metric("P/E Ratio", pe, comparison=25, invert=True)
        render_metric("Piotroski F", piotroski, fmt="{:.0f}", comparison=6)
        render_metric("ROE", roe, is_percent=True, comparison=0.15)
    with c2:
        render_metric("PEG Ratio", peg, comparison=1.5, invert=True)
        render_metric("Altman Z", altman, fmt="{:.2f}", comparison=2.99)
        render_metric("Debt/Equity", de, fmt="{:.1f}", comparison=100, invert=True)
        render_metric("Profit Margin", pm, is_percent=True, comparison=0.10)

    # 360 RADAR (Safe)
    if info:
        st.markdown("### 🎯 360° Analysis")
        # Default to 0 if None
        safe_pe = pe if pe else 50
        safe_rev = rev_g if rev_g else 0
        safe_roe = roe if roe else 0
        safe_gm = info.get('grossMargins', 0) if info else 0
        
        vals = [
            min(100, (1/(safe_pe+1))*2000), 
            min(100, safe_rev*300), 
            min(100, safe_roe*300), 
            min(100, (altman/4)*100), 
            min(100, safe_gm*150)
        ]
        radar_cats = ['Value','Growth','Profit','Health','Moat']
        fig_r = go.Figure(go.Scatterpolar(r=vals, theta=radar_cats, fill='toself', name=ticker))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), margin=dict(l=20, r=20, t=20, b=20), height=250, showlegend=False)
        st.plotly_chart(fig_r, use_container_width=True)

# RIGHT COLUMN
with col_R:
    st.markdown("### 📐 Technical Structure")
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.15, 0.15])
    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'), row=1, col=1)
    
    # Check if SMAs have data (drop NaN for plotting)
    if not hist['SMA50'].isna().all():
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA50'], line=dict(color='orange', width=1), name='SMA 50'), row=1, col=1)
    if not hist['SMA200'].isna().all():
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA200'], line=dict(color='#2962FF', width=1.5), name='SMA 200'), row=1, col=1)
    
    # Plot Pivots if they exist
    if not subset.empty and 'max' in subset.columns:
        peaks = subset[subset['max'].notna()]
        troughs = subset[subset['min'].notna()]
        fig.add_trace(go.Scatter(x=peaks.index, y=peaks['max'], mode='markers', marker=dict(color='red', size=6, symbol='triangle-down'), name='Pivot High'), row=1, col=1)
        fig.add_trace(go.Scatter(x=troughs.index, y=troughs['min'], mode='markers', marker=dict(color='green', size=6, symbol='triangle-up'), name='Pivot Low'), row=1, col=1)

    fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD_Signal'], name='Signal', line=dict(color='orange')), row=2, col=1)
    fig.add_trace(go.Bar(x=hist.index, y=hist['MACD_Hist'], name='Histogram', marker_color='gray'), row=2, col=1)

    colors = ['#ef5350' if row['Open'] - row['Close'] >= 0 else '#26a69a' for index, row in hist.iterrows()]
    fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=colors, name='Volume'), row=3, col=1)

    fig.update_layout(height=600, margin=dict(l=0,r=0,t=10,b=0), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    pat_str = "".join([f'<span class="pattern-tag">{p}</span>' for p in patterns_found]) if patterns_found else "No Chart Patterns Detected."
    
    # Safe access for trend
    trend = "Neutral"
    if not hist['SMA200'].isna().iloc[-1]:
        trend = "Bullish" if hist['Close'].iloc[-1] > hist['SMA200'].iloc[-1] else "Bearish"
    
    rsi_val = hist['RSI'].iloc[-1] if not np.isnan(hist['RSI'].iloc[-1]) else 0
    
    st.markdown(f"""
    <div class="tech-box">
        <div style="margin-bottom:5px;"><b>Detected Patterns:</b> {pat_str}</div>
        <div style="display:flex; justify-content:space-between; border-top:1px solid #eee; padding-top:5px;">
            <span>Long Trend: <b>{trend}</b></span>
            <span>RSI (14): <b>{rsi_val:.1f}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)
