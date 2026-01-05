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
import random

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

# --- SESSION STATE & CACHING ---
if 'print_mode' not in st.session_state:
    st.session_state.print_mode = False
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}

def toggle_print():
    st.session_state.print_mode = not st.session_state.print_mode

# --- HELPER: ROBUST FINANCIAL LOOKUP ---
def get_val(df, keys, col_index=0):
    """Safely retrieves a value from a DataFrame using a list of possible keys."""
    if df is None or df.empty: return 0
    
    # Try exact keys first
    for k in keys:
        if k in df.index:
            try:
                return df.loc[k].iloc[col_index]
            except:
                pass
    
    # Try searching index containing string
    for k in keys:
        matches = [idx for idx in df.index if k.lower() in str(idx).lower()]
        if matches:
            try:
                return df.loc[matches[0]].iloc[col_index]
            except:
                pass
    return 0

# --- 1. PATTERN RECOGNITION ---
def find_patterns(df, lookback_days=120):
    patterns = []
    if df is None or df.empty or len(df) < 20:
        return [], pd.DataFrame()

    subset = df.iloc[-lookback_days:].copy()
    
    try:
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

        if len(peaks) >= 2:
            if abs(peaks.iloc[-1] - peaks.iloc[-2])/peaks.iloc[-1] < 0.015: patterns.append("Double Top")
        if len(troughs) >= 2:
            if abs(troughs.iloc[-1] - troughs.iloc[-2])/troughs.iloc[-1] < 0.015: patterns.append("Double Bottom")

        # Candlesticks
        last = subset.iloc[-1]
        prev = subset.iloc[-2]
        body = abs(last['Close'] - last['Open'])
        rng = last['High'] - last['Low']
        
        if rng > 0 and body <= (rng * 0.1): patterns.append("Doji")
            
        lower_wick = min(last['Open'], last['Close']) - last['Low']
        upper_wick = last['High'] - max(last['Open'], last['Close'])
        
        if body > 0 and lower_wick >= (2 * body) and upper_wick <= (body * 0.5):
            if last['Close'] < subset['SMA50'].iloc[-1]: patterns.append("Hammer")
            else: patterns.append("Hanging Man")

        if body > 0 and upper_wick >= (2 * body) and lower_wick <= (body * 0.5):
            if last['Close'] > subset['SMA50'].iloc[-1]: patterns.append("Shooting Star")
            else: patterns.append("Inverted Hammer")
                
        if prev['Close'] < prev['Open'] and last['Close'] > last['Open']: 
            if last['Open'] <= prev['Close'] and last['Close'] >= prev['Open']: patterns.append("Bullish Engulfing")
                
        if prev['Close'] > prev['Open'] and last['Close'] < last['Open']: 
            if last['Open'] >= prev['Close'] and last['Close'] <= prev['Open']: patterns.append("Bearish Engulfing")
                
    except Exception as e:
        pass

    return patterns, subset

# --- 2. FUNDAMENTAL MODELS (UPDATED EQUATIONS) ---
def calculate_dcf(info, cashflow, shares_out):
    try:
        if cashflow is None or cashflow.empty: return None
        
        # Robust FCF
        fcf = get_val(cashflow, ['Free Cash Flow', 'freeCashFlow', 'FreeCashFlow'])
        if fcf == 0:
            ocf = get_val(cashflow, ['Operating Cash Flow', 'Total Cash From Operating Activities'])
            capex = get_val(cashflow, ['Capital Expenditure', 'Total Capitalization']) # Usually negative
            fcf = ocf + capex
            
        if fcf <= 0: return None 

        growth = min(info.get('revenueGrowth', 0.05) or 0.05, 0.15) 
        if growth < 0: growth = 0.02
        
        discount = 0.09
        perp = 0.025
        
        future_fcf = [fcf * ((1+growth)**i) for i in range(1,6)]
        terminal = future_fcf[-1] * (1+perp) / (discount-perp)
        
        dcf_val = sum([f/((1+discount)**(i+1)) for i, f in enumerate(future_fcf)]) + (terminal/((1+discount)**5))
        return dcf_val / shares_out
    except: return None

def calculate_piotroski(bs, is_, cf):
    """
    Calculates the 9-point Piotroski F-Score.
    Requires Current Year (0) and Previous Year (1) data.
    """
    try:
        if bs is None or is_ is None or cf is None: return 5
        if bs.shape[1] < 2 or is_.shape[1] < 2: return 5 # Need 2 years of data
        
        score = 0
        
        # 1. Net Income > 0
        ni_curr = get_val(is_, ['Net Income', 'Net Income Common Stockholders'], 0)
        ni_prev = get_val(is_, ['Net Income', 'Net Income Common Stockholders'], 1)
        score += 1 if ni_curr > 0 else 0
        
        # 2. Operating Cash Flow > 0
        ocf_curr = get_val(cf, ['Operating Cash Flow', 'Total Cash From Operating Activities'], 0)
        score += 1 if ocf_curr > 0 else 0
        
        # 3. ROA > 0 (Implied by NI > 0 if Assets > 0, but strictly NI/Assets)
        assets_curr = get_val(bs, ['Total Assets'], 0)
        assets_prev = get_val(bs, ['Total Assets'], 1)
        score += 1 if (ni_curr / assets_curr) > 0 else 0
        
        # 4. Operating Cash Flow > Net Income (Quality of Earnings)
        score += 1 if ocf_curr > ni_curr else 0
        
        # 5. Long Term Debt Lower (Decreased Leverage)
        ltd_curr = get_val(bs, ['Long Term Debt', 'Total Non Current Liabilities Net Minority Interest'], 0)
        ltd_prev = get_val(bs, ['Long Term Debt', 'Total Non Current Liabilities Net Minority Interest'], 1)
        score += 1 if ltd_curr <= ltd_prev else 0
        
        # 6. Current Ratio Higher (Increased Liquidity)
        cur_assets_c = get_val(bs, ['Total Current Assets'], 0)
        cur_liab_c = get_val(bs, ['Total Current Liabilities'], 0)
        cur_assets_p = get_val(bs, ['Total Current Assets'], 1)
        cur_liab_p = get_val(bs, ['Total Current Liabilities'], 1)
        
        # Avoid div by zero
        cr_curr = cur_assets_c / cur_liab_c if cur_liab_c else 0
        cr_prev = cur_assets_p / cur_liab_p if cur_liab_p else 0
        score += 1 if cr_curr > cr_prev else 0
        
        # 7. No New Shares Issued (Dilution)
        shares_curr = get_val(bs, ['Share Issued', 'Ordinary Shares Number', 'Common Stock'], 0)
        shares_prev = get_val(bs, ['Share Issued', 'Ordinary Shares Number', 'Common Stock'], 1)
        score += 1 if shares_curr <= shares_prev else 0
        
        # 8. Gross Margin Higher
        rev_curr = get_val(is_, ['Total Revenue', 'Total Revenue'], 0)
        gp_curr = get_val(is_, ['Gross Profit'], 0)
        rev_prev = get_val(is_, ['Total Revenue'], 1)
        gp_prev = get_val(is_, ['Gross Profit'], 1)
        
        gm_curr = gp_curr / rev_curr if rev_curr else 0
        gm_prev = gp_prev / rev_prev if rev_prev else 0
        score += 1 if gm_curr > gm_prev else 0
        
        # 9. Asset Turnover Higher
        at_curr = rev_curr / assets_curr if assets_curr else 0
        at_prev = rev_prev / assets_prev if assets_prev else 0
        score += 1 if at_curr > at_prev else 0
        
        return score
    except: return 5 # Default if calculation crashes

def calculate_altman(bs, is_, info):
    """
    Calculates Altman Z-Score for Manufacturing (Standard Formula).
    Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
    """
    try:
        if bs is None or is_ is None: return 3.0
        
        total_assets = get_val(bs, ['Total Assets'], 0)
        if total_assets == 0: return 3.0
        
        # A: Working Capital / Total Assets
        curr_assets = get_val(bs, ['Total Current Assets'], 0)
        curr_liab = get_val(bs, ['Total Current Liabilities'], 0)
        A = (curr_assets - curr_liab) / total_assets
        
        # B: Retained Earnings / Total Assets
        ret_earnings = get_val(bs, ['Retained Earnings', 'Retained Earnings Accum Deficit'], 0)
        B = ret_earnings / total_assets
        
        # C: EBIT / Total Assets
        ebit = get_val(is_, ['EBIT', 'Operating Income', 'Pretax Income'], 0) # Fallbacks
        C = ebit / total_assets
        
        # D: Market Value Equity / Total Liabilities
        mkt_cap = info.get('marketCap', 0)
        total_liab = get_val(bs, ['Total Liabilities Net Minority Interest', 'Total Liabilities'], 0)
        D = mkt_cap / total_liab if total_liab else 0
        
        # E: Sales / Total Assets
        revenue = get_val(is_, ['Total Revenue'], 0)
        E = revenue / total_assets
        
        z_score = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
        return z_score
    except: return 3.0

# --- 3. DATA FETCHING (SAFE) ---
def get_data_safe(symbol):
    if symbol in st.session_state.data_cache:
        return st.session_state.data_cache[symbol]

    retries = 3
    delay = 2
    last_error = None

    for i in range(retries):
        try:
            t = yf.Ticker(symbol)
            
            hist = t.history(period="1y")
            if hist.empty:
                time.sleep(1)
                hist = t.history(period="6mo")
            if hist.empty: raise ValueError("Empty history")

            info = {}
            try: 
                info = t.info
                if info is None: info = {}
            except: 
                try: 
                    f = t.fast_info
                    info = {'sharesOutstanding': f.shares, 'longName': symbol, 'sector': 'Unknown', 'previousClose': f.previous_close, 'marketCap': f.market_cap}
                except: info = {}
            
            bs, is_, cf = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            try: bs = t.balance_sheet
            except: pass
            try: is_ = t.financials
            except: pass
            try: cf = t.cashflow
            except: pass

            data_bundle = (info, bs, is_, cf, hist, None)
            st.session_state.data_cache[symbol] = data_bundle
            return data_bundle

        except Exception as e:
            last_error = e
            time.sleep(delay * (i + 1)) 
    
    return None, None, None, None, None, f"Failed after {retries} retries. Rate Limit or Network Error."

def render_metric(label, value, fmt="{:.2f}", is_percent=False, comparison=None, invert=False):
    if value is None or value == 0 or (isinstance(value, str) and value == "N/A"): 
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
        ticker_input = st.text_input("Ticker Symbol", "NVDA").upper()
        
        if 'last_ticker' not in st.session_state or st.session_state.last_ticker != ticker_input:
            st.session_state.data_cache = {} 
            st.session_state.last_ticker = ticker_input
            
        if st.button("🖨️ Printer Friendly Mode"): toggle_print()
else:
    c1, _ = st.columns([1, 10])
    with c1:
        if st.button("← Back"): toggle_print()
    ticker_input = st.session_state.get('last_ticker', "NVDA")

ticker = ticker_input

# FETCH
with st.spinner(f"Fetching data for {ticker}..."):
    info, bs, is_, cf, hist, err_msg = get_data_safe(ticker)

if err_msg or hist is None or hist.empty:
    st.warning("⚠️ Yahoo Finance is limiting requests or ticker is invalid.")
    st.error(f"Error Details: {err_msg}")
    st.stop()

# CALCULATE TECHNICALS
hist = hist.ffill().bfill()
hist['SMA50'] = hist['Close'].rolling(50).mean()
hist['SMA100'] = hist['Close'].rolling(100).mean()
hist['SMA200'] = hist['Close'].rolling(200).mean()
delta = hist['Close'].diff()
rs = (delta.where(delta>0,0).rolling(14).mean()) / (-delta.where(delta<0,0).rolling(14).mean())
hist['RSI'] = 100 - (100/(1+rs))

hist['EMA12'] = hist['Close'].ewm(span=12, adjust=False).mean()
hist['EMA26'] = hist['Close'].ewm(span=26, adjust=False).mean()
hist['MACD'] = hist['EMA12'] - hist['EMA26']
hist['MACD_Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
hist['MACD_Hist'] = hist['MACD'] - hist['MACD_Signal']

patterns_found, subset = find_patterns(hist)

# CALCULATE FUNDAMENTALS
current_price = hist['Close'].iloc[-1]
shares_out = info.get('sharesOutstanding', 1) if info else 1
if shares_out is None: shares_out = 1

dcf = calculate_dcf(info, cf, shares_out)

eps = info.get('trailingEps') if info else None
bk = info.get('bookValue') if info else None
graham = 0
if eps is not None and bk is not None and eps > 0 and bk > 0:
    graham = (22.5 * eps * bk)**0.5

piotroski = calculate_piotroski(bs, is_, cf)
altman = calculate_altman(bs, is_, info)
margin_safety = ((dcf - current_price) / dcf * 100) if dcf else None

# SCORING
rec_score = 0
if dcf and current_price < dcf: rec_score += 1
if piotroski >= 6: rec_score += 1
if not hist['SMA200'].isna().all() and hist['Close'].iloc[-1] > hist['SMA200'].iloc[-1]: rec_score += 1
if info and info.get('pegRatio') and info.get('pegRatio') < 1.5: rec_score += 1
if "Bullish" in str(patterns_found): rec_score += 1
if "Bearish" in str(patterns_found): rec_score -= 1

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

with col_L:
    st.markdown(f'<div class="rec-badge {b_cls}">{badge}</div>', unsafe_allow_html=True)
    
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
    
    st.markdown("### 🏗️ Fundamentals")
    c1, c2 = st.columns(2)
    
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

    if info:
        st.markdown("### 🎯 360° Analysis")
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
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])), 
            margin=dict(l=20, r=20, t=20, b=20), 
            height=250,
            showlegend=False
        )
        st.plotly_chart(fig_r, use_container_width=True)

with col_R:
    st.markdown("### 📐 Technical Structure")
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.15, 0.15])
    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'), row=1, col=1)
    
    if not hist['SMA50'].isna().all():
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA50'], line=dict(color='orange', width=1), name='SMA 50'), row=1, col=1)
    if not hist['SMA100'].isna().all():
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA100'], line=dict(color='purple', width=1), name='SMA 100'), row=1, col=1)
    if not hist['SMA200'].isna().all():
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA200'], line=dict(color='#2962FF', width=1.5), name='SMA 200'), row=1, col=1)
    
    if not subset.empty and 'max' in subset.columns:
        peaks = subset[subset['max'].notna()]
        troughs = subset[subset['min'].notna()]
        fig.add_trace(go.Scatter(x=peaks.index, y=peaks['max'], mode='markers', marker=dict(color='red', size=8, symbol='triangle-down'), name='Pivot High'), row=1, col=1)
        fig.add_trace(go.Scatter(x=troughs.index, y=troughs['min'], mode='markers', marker=dict(color='green', size=8, symbol='triangle-up'), name='Pivot Low'), row=1, col=1)

    if patterns_found:
        candle_patterns = [p for p in patterns_found if p in ["Doji", "Hammer", "Hanging Man", "Shooting Star", "Inverted Hammer", "Bullish Engulfing", "Bearish Engulfing"]]
        if candle_patterns:
            last_date = hist.index[-1]
            last_price = hist['High'].iloc[-1]
            txt_label = ", ".join(candle_patterns)
            
            fig.add_annotation(
                x=last_date, y=last_price,
                text=f"▼ {txt_label}",
                showarrow=False,
                yshift=10,
                font=dict(color="black", size=10, family="Arial Black"),
                bgcolor="#ffeb3b",
                row=1, col=1
            )

    fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD_Signal'], name='Signal', line=dict(color='orange')), row=2, col=1)
    fig.add_trace(go.Bar(x=hist.index, y=hist['MACD_Hist'], name='Histogram', marker_color='gray'), row=2, col=1)

    colors = ['#ef5350' if row['Open'] - row['Close'] >= 0 else '#26a69a' for index, row in hist.iterrows()]
    fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=colors, name='Volume'), row=3, col=1)

    fig.update_layout(height=600, margin=dict(l=0,r=0,t=10,b=0), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    pat_str = "".join([f'<span class="pattern-tag">{p}</span>' for p in patterns_found]) if patterns_found else "No Chart Patterns Detected."
    
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

if not st.session_state.print_mode:
    with st.expander("📘 Guide: How to Read This Report", expanded=True):
        st.markdown("""
        ### **1. Score Explanations**
        * **Piotroski F-Score (0-9):** Measures financial strength across profitability, leverage, and efficiency.
            * **8-9:** Very Strong. 
            * **0-2:** Weak/Distressed.
        * **Altman Z-Score:** Predicts bankruptcy risk based on working capital, retained earnings, EBIT, market value, and sales.
            * **> 3.0:** **Safe Zone.** * **1.8 - 3.0:** **Grey Zone.** * **< 1.8:** **Distress Zone.**
            
        ### **2. Other Metrics**
        * **360 Analysis:** A visual profile of the stock's strengths (Scale 0-100).
        * **Margin of Safety:** Difference between Intrinsic Value (DCF) and Price.
        * **Patterns:** Geometric (Head & Shoulders, Wedges) and Candlestick (Doji, Hammer) patterns are highlighted on the chart.
        """)
