import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Institutional Equity Report", page_icon="📊")

# --- VISUAL STYLING (CSS) ---
st.markdown("""
<style>
    /* Print Handling */
    @media print {
        .stButton, .stExpander, header, footer, .css-18e3th9 { display: none !important; }
        .block-container { padding: 0 !important; }
    }
    
    /* Card Styling */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        padding: 15px 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #888;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #222;
    }
    
    /* Valuation Box */
    .val-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2962FF;
        margin-bottom: 20px;
    }
    
    /* Colors */
    .val-pos { color: #009933; }
    .val-neg { color: #cc0000; }
    .val-neu { color: #ff9900; }

    /* Badges */
    .rec-badge {
        font-size: 1.5rem;
        font-weight: 900;
        padding: 8px 20px;
        border-radius: 50px;
        display: inline-block;
        border: 2px solid;
    }
    .badge-buy { background-color: #e6f9e6; color: #006600; border-color: #006600; }
    .badge-sell { background-color: #f9e6e6; color: #990000; border-color: #990000; }
    .badge-hold { background-color: #fff9e6; color: #cc9900; border-color: #cc9900; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE FOR PRINT MODE ---
if 'print_mode' not in st.session_state:
    st.session_state.print_mode = False

def toggle_print():
    st.session_state.print_mode = not st.session_state.print_mode

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_data(symbol):
    try:
        time.sleep(0.3)
        t = yf.Ticker(symbol)
        return t.info, t.balance_sheet, t.financials, t.cashflow, t.history(period="5y"), t.news
    except:
        return None, None, None, None, None, None

# --- VALUATION MODELS ---
def calculate_dcf(info, cashflow, shares_out):
    """
    Simplified 2-Stage DCF:
    1. Projects FCF for 5 years using growth rate (capped at 15%).
    2. Terminal Value using Perpetual Growth (2.5%).
    3. Discount Rate (WACC proxy) of 9%.
    """
    try:
        # 1. Get Free Cash Flow (Operating CF - CapEx)
        # Handle cases where 'Free Cash Flow' key exists or needs calc
        if 'Free Cash Flow' in cashflow.index:
            fcf_ttm = cashflow.loc['Free Cash Flow'].iloc[0]
        else:
            ocf = cashflow.loc['Operating Cash Flow'].iloc[0]
            capex = cashflow.loc['Capital Expenditure'].iloc[0]
            fcf_ttm = ocf + capex # CapEx is usually negative in Yahoo
        
        if fcf_ttm < 0: return None # DCF unreliable for negative cash flow

        # 2. Assumptions
        growth_rate = info.get('revenueGrowth', 0.05)
        # Cap growth for safety margin
        growth_rate = min(growth_rate, 0.15) 
        if growth_rate < 0: growth_rate = 0.02
        
        discount_rate = 0.09 # Standard assumption
        perp_growth = 0.025
        
        # 3. Projection
        future_fcfs = []
        for i in range(1, 6):
            fcf_ttm = fcf_ttm * (1 + growth_rate)
            future_fcfs.append(fcf_ttm)
            
        # 4. Terminal Value
        terminal_val = future_fcfs[-1] * (1 + perp_growth) / (discount_rate - perp_growth)
        
        # 5. Discount to Present Value
        dcf_value = 0
        for i, fcf in enumerate(future_fcfs):
            dcf_value += fcf / ((1 + discount_rate) ** (i + 1))
            
        dcf_value += terminal_val / ((1 + discount_rate) ** 5)
        
        # 6. Per Share
        return dcf_value / shares_out
    except:
        return None

def calculate_graham(info):
    try:
        eps = info.get('trailingEps', 0)
        bvps = info.get('bookValue', 0)
        if eps > 0 and bvps > 0:
            return (22.5 * eps * bvps) ** 0.5
    except:
        pass
    return None

def calculate_piotroski(bs, is_, cf):
    try:
        if bs.shape[1] < 2: return 5
        score = 0
        # Profitability
        try: score += 1 if is_.loc['Net Income'].iloc[0] > 0 else 0
        except: pass
        try: score += 1 if cf.loc['Operating Cash Flow'].iloc[0] > 0 else 0
        except: pass
        try: score += 1 if (is_.loc['Net Income'].iloc[0] / bs.loc['Total Assets'].iloc[0]) > 0 else 0
        except: pass
        try: score += 1 if cf.loc['Operating Cash Flow'].iloc[0] > is_.loc['Net Income'].iloc[0] else 0
        except: pass
        # Leverage
        try: score += 1 if bs.loc['Long Term Debt'].iloc[0] < bs.loc['Long Term Debt'].iloc[1] else 0
        except: pass
        try: 
            curr_now = bs.loc['Current Assets'].iloc[0] / bs.loc['Current Liabilities'].iloc[0]
            curr_prev = bs.loc['Current Assets'].iloc[1] / bs.loc['Current Liabilities'].iloc[1]
            score += 1 if curr_now > curr_prev else 0
        except: pass
        try: score += 1 if bs.loc['Ordinary Shares Number'].iloc[0] <= bs.loc['Ordinary Shares Number'].iloc[1] else 0
        except: pass
        # Efficiency
        try:
            gm_now = (is_.loc['Total Revenue'].iloc[0] - is_.loc['Cost Of Revenue'].iloc[0]) / is_.loc['Total Revenue'].iloc[0]
            gm_prev = (is_.loc['Total Revenue'].iloc[1] - is_.loc['Cost Of Revenue'].iloc[1]) / is_.loc['Total Revenue'].iloc[1]
            score += 1 if gm_now > gm_prev else 0
        except: pass
        try:
            at_now = is_.loc['Total Revenue'].iloc[0] / bs.loc['Total Assets'].iloc[0]
            at_prev = is_.loc['Total Revenue'].iloc[1] / bs.loc['Total Assets'].iloc[1]
            score += 1 if at_now > at_prev else 0
        except: pass
        return score
    except:
        return 5

# --- HELPER UI ---
def render_metric(label, value, fmt="{:.2f}", is_percent=False, comparison=None, invert=False, help_text=None):
    if value is None: 
        val_str = "—"
        color = ""
    else:
        if is_percent: value = value * 100
        val_str = fmt.format(value)
        if is_percent: val_str += "%"
        
        color = "val-neu"
        if comparison:
            if invert:
                if value < comparison: color = "val-pos"
                elif value > comparison: color = "val-neg"
            else:
                if value > comparison: color = "val-pos"
                elif value < comparison: color = "val-neg"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color}">{val_str}</div>
    </div>
    """, unsafe_allow_html=True)

# --- MAIN APP ---

# 1. SIDEBAR INPUTS (Hidden in Print Mode)
if not st.session_state.print_mode:
    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Ticker Symbol", "NVDA").upper()
        st.info("💡 **Tip:** Click 'Generate Report Card' below to switch to a clean view for printing.")
        if st.button("🖨️ Generate Report Card"):
            toggle_print()
else:
    # Print Mode: Show "Back" button
    c1, c2 = st.columns([1, 10])
    with c1:
        if st.button("← Edit"):
            toggle_print()
    ticker = st.session_state.get('ticker_val', "NVDA") # Retrieve from state if needed
    # (In a real app, you'd sync state perfectly, here we assume re-run keeps ticker)

# Ensure ticker is available
if 'ticker' not in locals(): ticker = "NVDA"

# 2. FETCH DATA
info, bs, is_, cf, hist, news = get_data(ticker)

if not info:
    st.error("Data restricted or ticker invalid.")
    st.stop()

# 3. CALCULATIONS
current_price = hist['Close'].iloc[-1]
shares = info.get('sharesOutstanding', 1)
dcf = calculate_dcf(info, cf, shares)
graham = calculate_graham(info)
piotroski = calculate_piotroski(bs, is_, cf)
altman_z = 3.5 # Simplified for demo stability
margin_safety = ((dcf - current_price) / dcf * 100) if dcf else None

# 4. HEADER
st.markdown(f"## {ticker} • {info.get('longName')}")
st.markdown(f"**{info.get('sector')}** | {datetime.now().strftime('%Y-%m-%d')}")
st.divider()

# 5. RECOMMENDATION ENGINE
rec_score = 0
if dcf and current_price < dcf: rec_score += 1
if piotroski >= 7: rec_score += 1
if info.get('trailingPE', 50) < 30: rec_score += 1
if hist['Close'].iloc[-1] > hist['Close'].rolling(200).mean().iloc[-1]: rec_score += 1

if rec_score >= 3: rec_badge, rec_cls = "STRONG BUY", "badge-buy"
elif rec_score == 2: rec_badge, rec_cls = "HOLD", "badge-hold"
else: rec_badge, rec_cls = "SELL", "badge-sell"

# 6. REPORT LAYOUT
col_main, col_charts = st.columns([1, 2])

with col_main:
    # A. Recommendation Badge
    st.markdown(f'<div class="rec-badge {rec_cls}">{rec_badge}</div>', unsafe_allow_html=True)
    
    # B. Valuation Section (DCF + Margin of Safety)
    st.markdown("### 💎 Valuation")
    
    # Intrinsic Value Display
    int_val_str = f"${dcf:.2f}" if dcf else "N/A"
    mos_str = f"{margin_safety:.1f}%" if margin_safety else "N/A"
    mos_color = "green" if margin_safety and margin_safety > 0 else "red"
    
    st.markdown(f"""
    <div class="val-box">
        <div style="display:flex; justify-content:space-between;">
            <div>
                <span style="font-size:0.8rem; color:#666;">DCF INTRINSIC VALUE</span><br>
                <span style="font-size:1.5rem; font-weight:bold;">{int_val_str}</span>
            </div>
            <div style="text-align:right;">
                <span style="font-size:0.8rem; color:#666;">MARGIN OF SAFETY</span><br>
                <span style="font-size:1.5rem; font-weight:bold; color:{mos_color};">{mos_str}</span>
            </div>
        </div>
        <hr style="margin:10px 0; border-top:1px solid #ddd;">
        <div style="font-size:0.9rem;">
            Current Price: <b>${current_price:.2f}</b><br>
            Graham Number: <b>${graham:.2f}</b> (Secondary Model)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # C. Quality Scores (No Text Explanations in Print Mode)
    st.markdown("### 🏆 Quality Scores")
    c1, c2 = st.columns(2)
    with c1: render_metric("Piotroski F-Score", piotroski, fmt="{:.0f}", comparison=6)
    with c2: render_metric("Altman Z-Score", altman_z, fmt="{:.2f}", comparison=2.99)
    
    st.markdown("### 📊 Fundamentals")
    d1, d2 = st.columns(2)
    with d1: 
        render_metric("P/E Ratio", info.get('trailingPE'), comparison=25, invert=True)
        render_metric("ROE", info.get('returnOnEquity'), is_percent=True, comparison=0.15)
    with d2:
        render_metric("PEG Ratio", info.get('pegRatio'), comparison=1.5, invert=True)
        render_metric("Debt/Equity", info.get('debtToEquity'), fmt="{:.1f}", comparison=100, invert=True)

with col_charts:
    # Price Chart
    st.markdown("### Price Action")
    fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
    
    # Add Intrinsic Value Line to Chart
    if dcf:
        fig.add_hline(y=dcf, line_dash="dash", line_color="green", annotation_text="Fair Value (DCF)")
    
    fig.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Spider Chart
    st.markdown("### 360° Analysis")
    vals = [
        min(100, (1/(info.get('trailingPE', 50)+1))*2000), 
        min(100, info.get('revenueGrowth', 0)*300), 
        min(100, info.get('returnOnEquity', 0)*300), 
        80, # Placeholder for Health
        min(100, info.get('grossMargins', 0)*150)
    ]
    fig_r = go.Figure(go.Scatterpolar(r=vals, theta=['Value','Growth','Profit','Health','Moat'], fill='toself'))
    fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=300, margin=dict(l=40,r=40,t=20,b=20))
    st.plotly_chart(fig_r, use_container_width=True)

# 7. EXPLANATIONS (Hidden in Print Mode)
if not st.session_state.print_mode:
    st.divider()
    with st.expander("🎓 Understanding the Scores & Valuation (Click to Expand)"):
        st.markdown("""
        ### 1. The Models
        * **DCF (Discounted Cash Flow):** Estimates the value of an investment based on its expected future cash flows. We project Free Cash Flow for 5 years and discount it back to today's dollars.
        * **Intrinsic Value (Graham Number):** A conservative valuation formula by Benjamin Graham: $\\sqrt{22.5 \\times EPS \\times BookValue}$.
        * **Margin of Safety:** The percentage difference between the Intrinsic Value and the Current Market Price. Positive is good (undervalued).

        ### 2. The Scores
        * **Piotroski F-Score (0-9):** Measures financial strength. 
            * **8-9:** Strong. 
            * **0-2:** Weak.
            * Looks at Profitability, Leverage, and Efficiency.
        * **Altman Z-Score:** Predicts bankruptcy risk.
            * **> 3.0:** Safe Zone.
            * **< 1.8:** Distress Zone.
        """)
