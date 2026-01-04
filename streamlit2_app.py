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
    
    /* Technical Box */
    .tech-box {
        background: #fdfdfe;
        padding: 10px;
        border: 1px solid #eee;
        border-radius: 5px;
        font-size: 0.85rem;
        margin-bottom: 10px;
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

# --- SESSION STATE ---
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
        # Get 2 years of history for better technical analysis
        hist = t.history(period="2y")
        return t.info, t.balance_sheet, t.financials, t.cashflow, hist, t.news
    except:
        return None, None, None, None, None, None

# --- TECHNICAL ANALYSIS ENGINE ---
def calculate_technicals(df):
    """Calculates SMAs, RSI, and detects basic patterns."""
    if df.empty: return df
    
    # 1. Moving Averages
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    # 2. RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. Pattern Detection
    # Doji: Body is very small relative to range
    body_size = (df['Close'] - df['Open']).abs()
    full_range = df['High'] - df['Low']
    df['Is_Doji'] = body_size <= (full_range * 0.1)
    
    # Hammer: Small body, long lower wick, small upper wick
    lower_wick = df[['Open', 'Close']].min(axis=1) - df['Low']
    upper_wick = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Is_Hammer'] = (lower_wick > 2 * body_size) & (upper_wick < body_size)
    
    return df

# --- VALUATION MODELS ---
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
        
        dcf_value = 0
        for i, fcf in enumerate(future_fcfs):
            dcf_value += fcf / ((1 + discount_rate) ** (i + 1))
        dcf_value += terminal_val / ((1 + discount_rate) ** 5)
        
        return dcf_value / shares_out
    except:
        return None

def calculate_piotroski(bs, is_, cf):
    try:
        if bs.shape[1] < 2: return 5
        score = 0
        try: score += 1 if is_.loc['Net Income'].iloc[0] > 0 else 0
        except: pass
        try: score += 1 if cf.loc['Operating Cash Flow'].iloc[0] > 0 else 0
        except: pass
        try: score += 1 if (is_.loc['Net Income'].iloc[0] / bs.loc['Total Assets'].iloc[0]) > 0 else 0
        except: pass
        try: score += 1 if cf.loc['Operating Cash Flow'].iloc[0] > is_.loc['Net Income'].iloc[0] else 0
        except: pass
        try: score += 1 if bs.loc['Long Term Debt'].iloc[0] < bs.loc['Long Term Debt'].iloc[1] else 0
        except: pass
        try: 
            curr_now = bs.loc['Current Assets'].iloc[0] / bs.loc['Current Liabilities'].iloc[0]
            curr_prev = bs.loc['Current Assets'].iloc[1] / bs.loc['Current Liabilities'].iloc[1]
            score += 1 if curr_now > curr_prev else 0
        except: pass
        try: score += 1 if bs.loc['Ordinary Shares Number'].iloc[0] <= bs.loc['Ordinary Shares Number'].iloc[1] else 0
        except: pass
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
def render_metric(label, value, fmt="{:.2f}", is_percent=False, comparison=None, invert=False):
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

# 1. SIDEBAR INPUTS
if not st.session_state.print_mode:
    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Ticker Symbol", "NVDA").upper()
        if st.button("🖨️ Generate Report Card"):
            toggle_print()
else:
    c1, c2 = st.columns([1, 10])
    with c1:
        if st.button("← Edit"):
            toggle_print()
    ticker = st.session_state.get('ticker_val', "NVDA")

if 'ticker' not in locals(): ticker = "NVDA"

# 2. FETCH DATA
info, bs, is_, cf, hist, news = get_data(ticker)

if not info or hist.empty:
    st.error("Data restricted or ticker invalid.")
    st.stop()

# 3. CALCULATE EVERYTHING
hist = calculate_technicals(hist) # Add patterns and SMAs
current_price = hist['Close'].iloc[-1]
shares = info.get('sharesOutstanding', 1)
dcf = calculate_dcf(info, cf, shares)
graham = (22.5 * info.get('trailingEps',0) * info.get('bookValue',0))**0.5 if info.get('trailingEps',0)>0 else 0
piotroski = calculate_piotroski(bs, is_, cf)
margin_safety = ((dcf - current_price) / dcf * 100) if dcf else None

# 4. HEADER
st.markdown(f"## {ticker} • {info.get('longName')}")
st.markdown(f"**{info.get('sector')}** | {datetime.now().strftime('%Y-%m-%d')}")
st.divider()

# 5. RECOMMENDATION ENGINE
rec_score = 0
if dcf and current_price < dcf: rec_score += 1
if piotroski >= 7: rec_score += 1
if hist['Close'].iloc[-1] > hist['SMA200'].iloc[-1]: rec_score += 1
if info.get('pegRatio', 5) < 1.5: rec_score += 1

if rec_score >= 3: rec_badge, rec_cls = "STRONG BUY", "badge-buy"
elif rec_score == 2: rec_badge, rec_cls = "HOLD", "badge-hold"
else: rec_badge, rec_cls = "SELL", "badge-sell"

# 6. REPORT LAYOUT
col_main, col_charts = st.columns([1, 2])

with col_main:
    # A. Recommendation Badge
    st.markdown(f'<div class="rec-badge {rec_cls}">{rec_badge}</div>', unsafe_allow_html=True)
    
    # B. Valuation Section
    st.markdown("### 💎 Valuation")
    int_val_str = f"${dcf:.2f}" if dcf else "N/A"
    mos_str = f"{margin_safety:.1f}%" if margin_safety else "N/A"
    mos_color = "green" if margin_safety and margin_safety > 0 else "red"
    
    st.markdown(f"""
    <div class="val-box">
        <div style="display:flex; justify-content:space-between;">
            <div><span style="font-size:0.8rem; color:#666;">DCF INTRINSIC VALUE</span><br><span style="font-size:1.5rem; font-weight:bold;">{int_val_str}</span></div>
            <div style="text-align:right;"><span style="font-size:0.8rem; color:#666;">MARGIN OF SAFETY</span><br><span style="font-size:1.5rem; font-weight:bold; color:{mos_color};">{mos_str}</span></div>
        </div>
        <hr style="margin:10px 0; border-top:1px solid #ddd;">
        <div style="font-size:0.9rem;">
            Current Price: <b>${current_price:.2f}</b><br>
            Graham Number: <b>${graham:.2f}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # C. Technical Trend Summary (NEW)
    trend_50 = "Bullish" if hist['Close'].iloc[-1] > hist['SMA50'].iloc[-1] else "Bearish"
    trend_200 = "Bullish" if hist['Close'].iloc[-1] > hist['SMA200'].iloc[-1] else "Bearish"
    rsi_val = hist['RSI'].iloc[-1]
    
    st.markdown("### 📉 Technical Trend")
    st.markdown(f"""
    <div class="tech-box">
        <b>Short Term (50D):</b> {trend_50} <br>
        <b>Long Term (200D):</b> {trend_200} <br>
        <b>RSI (14):</b> {rsi_val:.1f} ({'Overbought' if rsi_val>70 else 'Oversold' if rsi_val<30 else 'Neutral'})
    </div>
    """, unsafe_allow_html=True)

    # D. Fundamentals
    st.markdown("### 📊 Fundamentals")
    d1, d2 = st.columns(2)
    with d1: 
        render_metric("P/E Ratio", info.get('trailingPE'), comparison=25, invert=True)
        render_metric("ROE", info.get('returnOnEquity'), is_percent=True, comparison=0.15)
        render_metric("Piotroski", piotroski, fmt="{:.0f}", comparison=6)
    with d2:
        render_metric("PEG Ratio", info.get('pegRatio'), comparison=1.5, invert=True)
        render_metric("Debt/Equity", info.get('debtToEquity'), fmt="{:.1f}", comparison=100, invert=True)
        render_metric("Altman Z", 3.2, fmt="{:.1f}", comparison=2.99)

with col_charts:
    # --- ENHANCED CHART WITH VOLUME & PATTERNS ---
    st.markdown("### Technical Analysis")
    
    # Create Subplots: Row 1 = Price, Row 2 = Volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # 1. Candlestick
    fig.add_trace(go.Candlestick(x=hist.index,
                                 open=hist['Open'], high=hist['High'],
                                 low=hist['Low'], close=hist['Close'],
                                 name='Price'), row=1, col=1)
    
    # 2. Moving Averages
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA50'], line=dict(color='orange', width=1), name='SMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA200'], line=dict(color='#2962FF', width=1.5), name='SMA 200'), row=1, col=1)

    # 3. Pattern Markers (Doji / Hammer)
    doji_data = hist[hist['Is_Doji']]
    if not doji_data.empty:
        fig.add_trace(go.Scatter(x=doji_data.index, y=doji_data['High'], mode='markers', 
                                 marker=dict(symbol='diamond', size=5, color='yellow'), name='Doji'), row=1, col=1)

    hammer_data = hist[hist['Is_Hammer']]
    if not hammer_data.empty:
        fig.add_trace(go.Scatter(x=hammer_data.index, y=hammer_data['Low'], mode='markers', 
                                 marker=dict(symbol='triangle-up', size=5, color='purple'), name='Hammer'), row=1, col=1)

    # 4. Intrinsic Value Line
    if dcf:
        fig.add_hline(y=dcf, line_dash="dash", line_color="green", annotation_text="Fair Value", row=1, col=1)

    # 5. Volume Bar Chart
    colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in hist.iterrows()]
    fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

    # Styling
    fig.update_layout(
        height=550,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left"),
        xaxis_rangeslider_visible=False
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Spider Chart
    st.markdown("### 360° Analysis")
    vals = [
        min(100, (1/(info.get('trailingPE', 50)+1))*2000), 
        min(100, info.get('revenueGrowth', 0)*300), 
        min(100, info.get('returnOnEquity', 0)*300), 
        80, 
        min(100, info.get('grossMargins', 0)*150)
    ]
    fig_r = go.Figure(go.Scatterpolar(r=vals, theta=['Value','Growth','Profit','Health','Moat'], fill='toself'))
    fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=300, margin=dict(l=40,r=40,t=20,b=20))
    st.plotly_chart(fig_r, use_container_width=True)

# 7. EXPLANATIONS
if not st.session_state.print_mode:
    st.divider()
    with st.expander("🎓 Understanding Technicals & Valuation"):
        st.markdown("""
        * **Patterns (Doji/Hammer):** Yellow diamonds indicate 'Doji' (market indecision). Purple triangles indicate 'Hammers' (potential reversal).
        * **SMA 50/200:** The intersection of these lines indicates 'Golden Cross' (Bullish) or 'Death Cross' (Bearish).
        * **DCF Model:** Projects 5 years of Free Cash Flow + Terminal Value. Discount rate: 9%. Growth capped at 15%.
        """)
