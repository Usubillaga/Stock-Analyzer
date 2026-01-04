import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Professional Stock Analyst", page_icon="📈")

# --- Custom CSS ---
st.markdown("""
<style>
    .metric-container {
        border: 1px solid #e0e0e0;
        padding: 10px;
        border-radius: 5px;
        background-color: #ffffff;
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #666;
        margin-bottom: 2px;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: bold;
        color: #000;
    }
    .positive { color: #008000; }
    .negative { color: #d32f2f; }
    .neutral { color: #f57c00; }
    .section-header {
        font-size: 1rem;
        font-weight: bold;
        background-color: #f0f2f6;
        padding: 5px 10px;
        margin-top: 20px;
        margin-bottom: 10px;
        border-left: 5px solid #000;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. ROBUST DATA FETCHING (Split into parts) ---

@st.cache_data(ttl=3600)
def get_price_history(symbol):
    """Fetches only price history (Low risk of block)."""
    try:
        ticker = yf.Ticker(symbol)
        return ticker.history(period="5y")
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_fundamentals(symbol):
    """Fetches 'Heavy' data (High risk of block). Returns None if blocked."""
    try:
        # Add a tiny delay to be polite to the API
        time.sleep(0.5) 
        ticker = yf.Ticker(symbol)
        
        # We explicitly access these to trigger the download
        info = ticker.info
        bs = ticker.balance_sheet
        fin = ticker.financials
        cf = ticker.cashflow
        news = ticker.news
        return info, bs, fin, cf, news
    except Exception:
        return None, None, None, None, None

# --- 2. CALCULATION HELPERS (Safe Handling) ---

def safe_get(data, key, default=0):
    """Safely gets a value from a dict or returns default."""
    if not data: return default
    return data.get(key, default)

def calculate_piotroski_f_score(bs, is_, cf):
    if bs is None or is_ is None or cf is None: return 0
    try:
        if bs.shape[1] < 2 or is_.shape[1] < 2 or cf.shape[1] < 2: return 5
        curr, prev = 0, 1
        score = 0
        
        # Profitability
        try: score += 1 if is_.loc['Net Income'].iloc[curr] > 0 else 0
        except: pass
        try: score += 1 if (is_.loc['Net Income'].iloc[curr] / bs.loc['Total Assets'].iloc[curr]) > 0 else 0
        except: pass
        try: score += 1 if cf.loc['Operating Cash Flow'].iloc[curr] > 0 else 0
        except: pass
        try: score += 1 if cf.loc['Operating Cash Flow'].iloc[curr] > is_.loc['Net Income'].iloc[curr] else 0
        except: pass
        
        # Leverage
        try: score += 1 if bs.loc['Long Term Debt'].iloc[curr] < bs.loc['Long Term Debt'].iloc[prev] else 0
        except: pass
        try: 
            curr_r_now = bs.loc['Current Assets'].iloc[curr] / bs.loc['Current Liabilities'].iloc[curr]
            curr_r_prev = bs.loc['Current Assets'].iloc[prev] / bs.loc['Current Liabilities'].iloc[prev]
            score += 1 if curr_r_now > curr_r_prev else 0
        except: pass
        
        # Efficiency
        try: score += 1 if bs.loc['Ordinary Shares Number'].iloc[curr] <= bs.loc['Ordinary Shares Number'].iloc[prev] else 0
        except: pass
        try:
            gm_now = (is_.loc['Total Revenue'].iloc[curr] - is_.loc['Cost Of Revenue'].iloc[curr]) / is_.loc['Total Revenue'].iloc[curr]
            gm_prev = (is_.loc['Total Revenue'].iloc[prev] - is_.loc['Cost Of Revenue'].iloc[prev]) / is_.loc['Total Revenue'].iloc[prev]
            score += 1 if gm_now > gm_prev else 0
        except: pass
        
        return score
    except:
        return 5

def calculate_altman_z(bs, is_, info):
    if bs is None or is_ is None or not info: return 0
    try:
        tot_assets = bs.loc['Total Assets'].iloc[0]
        ebit = is_.loc['EBIT'].iloc[0] if 'EBIT' in is_.index else is_.loc['Net Income'].iloc[0]
        retained = bs.loc['Retained Earnings'].iloc[0] if 'Retained Earnings' in bs.index else 0
        wk_cap = bs.loc['Current Assets'].iloc[0] - bs.loc['Current Liabilities'].iloc[0]
        rev = is_.loc['Total Revenue'].iloc[0]
        mkt_cap = info.get('marketCap', 0)
        tot_liab = bs.loc['Total Liabilities Net Minority Interest'].iloc[0]
        
        A = wk_cap / tot_assets
        B = retained / tot_assets
        C = ebit / tot_assets
        D = mkt_cap / tot_liab
        E = rev / tot_assets
        
        return 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
    except:
        return 0

# --- 3. UI COMPONENTS ---

def render_card(label, value, comparison=None, fmt="{:.2f}", is_score=False):
    color = ""
    if value is None:
        val_str = "N/A"
    else:
        val_str = fmt.format(value)
        if comparison is not None:
            color = "positive" if value > comparison else "negative"
        if is_score:
            color = "positive" if value > 6 else ("negative" if value < 3 else "neutral")

    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color}">{val_str}</div>
    </div>
    """, unsafe_allow_html=True)

# --- MAIN APP ---

st.title("Equity Research Dashboard")
col1, col2 = st.columns([1, 3])
ticker = col1.text_input("Ticker", "NVO").upper()

if ticker:
    # 1. Fetch History (Chart) - This rarely fails
    hist = get_price_history(ticker)
    
    if hist.empty:
        st.error("Could not find ticker symbol.")
        st.stop()
        
    # 2. Fetch Fundamentals (Info) - This often fails
    info, bs, is_, cf, news = get_fundamentals(ticker)
    
    # Determine if we are in "Restricted Mode" (Rate Limited)
    is_restricted = info is None
    
    if is_restricted:
        st.warning("⚠️ Deep Fundamental Data is currently rate-limited by Yahoo. Showing Price Chart only.")
        
    # --- HEADER ---
    name = safe_get(info, 'longName', ticker)
    sector = safe_get(info, 'sector', 'Unknown')
    st.markdown(f"## {ticker} - {name}")
    st.caption(f"Sector: {sector}")
    st.divider()

    # --- METRICS (Only show if not restricted) ---
    if not is_restricted:
        piotroski = calculate_piotroski_f_score(bs, is_, cf)
        altman = calculate_altman_z(bs, is_, info)
        
        st.markdown('<div class="section-header">Valuation & Quality</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Price", f"${hist['Close'].iloc[-1]:.2f}")
        render_card("P/E Ratio", safe_get(info, 'trailingPE'), 25)
        render_card("Piotroski F", piotroski, is_score=True, fmt="{:.0f}")
        render_card("Altman Z", altman, is_score=True)
        render_card("ROE", safe_get(info, 'returnOnEquity') * 100, 15, "{:.1f}%")
        render_card("Profit Margin", safe_get(info, 'profitMargins') * 100, 10, "{:.1f}%")

    # --- CHARTS (Always show this) ---
    st.divider()
    g1, g2 = st.columns([2, 1])
    
    with g1:
        st.subheader("Price Action")
        fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
        fig.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with g2:
        if not is_restricted:
            st.subheader("Analysis Profile")
            # Simple Radar
            vals = [
                min(100, (1/(safe_get(info,'trailingPE',20)+1))*1500), # Norm PE
                min(100, safe_get(info,'revenueGrowth',0)*200),
                min(100, safe_get(info,'returnOnEquity',0)*300),
                min(100, (altman/4)*100),
                min(100, safe_get(info,'grossMargins',0)*100)
            ]
            fig_r = go.Figure(go.Scatterpolar(r=vals, theta=['Valuation','Growth','Profit','Health','Moat'], fill='toself'))
            fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=350, margin=dict(l=30,r=30,t=20,b=20))
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("Analysis Chart unavailable due to rate limits.")

    # --- NEWS ---
    if news:
        st.subheader("Latest News")
        for n in news[:3]:
            st.markdown(f"**[{n['title']}]({n['link']})**")
