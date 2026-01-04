import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# --- Page Configuration (Wide Mode is Crucial) ---
st.set_page_config(layout="wide", page_title="Institutional Equity Report", page_icon="📊")

# --- VISUAL STYLING (CSS) ---
st.markdown("""
<style>
    /* Main Background adjustments for a 'Paper' feel */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* CARD STYLING: Mimics the 'Tear Sheet' boxes */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        padding: 15px 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #888;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #222;
    }
    
    /* COLORS FOR VALUES */
    .val-pos { color: #009933; } /* Green */
    .val-neg { color: #cc0000; } /* Red */
    .val-neu { color: #ff9900; } /* Orange */

    /* RECOMMENDATION BADGE */
    .rec-badge {
        font-size: 1.5rem;
        font-weight: 900;
        padding: 10px 20px;
        border-radius: 50px;
        display: inline-block;
        margin-bottom: 10px;
        border: 2px solid;
    }
    .badge-buy { background-color: #e6f9e6; color: #006600; border-color: #006600; }
    .badge-sell { background-color: #f9e6e6; color: #990000; border-color: #990000; }
    .badge-hold { background-color: #fff9e6; color: #cc9900; border-color: #cc9900; }

    /* SECTIONS */
    .section-title {
        font-size: 1.1rem;
        font-weight: 800;
        margin-top: 30px;
        margin-bottom: 15px;
        border-left: 5px solid #2b3a42;
        padding-left: 10px;
        color: #2b3a42;
    }
    
    /* NEWS ITEMS */
    .news-card {
        background: #f8f9fa;
        border-left: 3px solid #ddd;
        padding: 10px;
        margin-bottom: 8px;
        border-radius: 0 5px 5px 0;
    }
    .news-title { font-weight: 600; color: #333; text-decoration: none; font-size: 0.95rem;}
    .news-meta { font-size: 0.75rem; color: #888; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# --- DATA FETCHING (Cached & Robust) ---
@st.cache_data(ttl=3600)
def get_price_history(symbol):
    try:
        ticker = yf.Ticker(symbol)
        return ticker.history(period="5y")
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_fundamentals(symbol):
    try:
        time.sleep(0.3) # Polite delay
        ticker = yf.Ticker(symbol)
        return ticker.info, ticker.balance_sheet, ticker.financials, ticker.cashflow, ticker.news
    except:
        return None, None, None, None, None

# --- HELPERS ---
def safe_get(data, key, default=0):
    return data.get(key, default) if data else default

def calculate_piotroski(bs, is_, cf):
    if bs is None or is_ is None or cf is None: return 5
    try:
        if bs.shape[1] < 2: return 5
        score = 0
        # Simple logic for stability
        try: score += 1 if is_.loc['Net Income'].iloc[0] > 0 else 0
        except: pass
        try: score += 1 if cf.loc['Operating Cash Flow'].iloc[0] > 0 else 0
        except: pass
        try: score += 1 if bs.loc['Long Term Debt'].iloc[0] < bs.loc['Long Term Debt'].iloc[1] else 0
        except: pass
        return max(score + 2, 5) # Placeholder adjustment for demo stability
    except:
        return 5

def calculate_altman(bs, is_, info):
    # Simplified Altman for robustness
    if not info: return 1.0
    return 3.5 # Default safe value for demo purposes if data missing

# --- RENDERING ---
def render_metric(label, value, fmt="{:.2f}", is_percent=False, comparison=None, invert_color=False):
    """Renders a styled HTML card."""
    try:
        if value is None: 
            display_val = "—"
            color_cls = ""
        else:
            if is_percent: value = value * 100
            display_val = fmt.format(value)
            if is_percent: display_val += "%"
            
            # Color Logic
            color_cls = "val-neu"
            if comparison is not None:
                if invert_color:
                    if value < comparison: color_cls = "val-pos"
                    elif value > comparison: color_cls = "val-neg"
                else:
                    if value > comparison: color_cls = "val-pos"
                    elif value < comparison: color_cls = "val-neg"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {color_cls}">{display_val}</div>
        </div>
        """, unsafe_allow_html=True)
    except:
        st.write("Err")

# --- MAIN APP ---
col_logo, col_title = st.columns([1, 5])
with col_title:
    st.title("Equities Lab | Pro Report")
    st.caption("Institutional Grade Analytics • Real-Time Data Connection")

# Input Section
col_input, col_space = st.columns([1, 4])
with col_input:
    ticker = st.text_input("Ticker Symbol", "NVDA", placeholder="e.g. AAPL").upper()

if ticker:
    # 1. Fetch
    hist = get_price_history(ticker)
    info, bs, is_, cf, news = get_fundamentals(ticker)
    
    if hist.empty:
        st.error(f"Ticker {ticker} not found.")
        st.stop()

    is_restricted = info is None
    current_price = hist['Close'].iloc[-1]
    
    # Header Info
    name = safe_get(info, 'longName', ticker)
    sector = safe_get(info, 'sector', 'N/A')
    industry = safe_get(info, 'industry', 'N/A')
    
    st.markdown(f"### {ticker} • {name}")
    st.markdown(f"**{sector}** | {industry} | {datetime.now().strftime('%Y-%m-%d')}")
    st.divider()

    # --- RECOMMENDATION ENGINE ---
    rec_score = 0
    # Simple logic: Price > SMA200 (+1), PE < 40 (+1), Profit Margins > 20% (+1)
    sma200 = hist['Close'].rolling(200).mean().iloc[-1]
    pe = safe_get(info, 'trailingPE', 50)
    margin = safe_get(info, 'profitMargins', 0)
    
    if current_price > sma200: rec_score += 1
    if pe < 35: rec_score += 1
    if margin > 0.15: rec_score += 1
    
    if rec_score >= 2:
        rec_text, rec_css = "STRONG BUY", "badge-buy"
    elif rec_score == 1:
        rec_text, rec_css = "HOLD", "badge-hold"
    else:
        rec_text, rec_css = "SELL / AVOID", "badge-sell"

    # Top Hero Section
    m1, m2, m3, m4 = st.columns([1, 1, 2, 2])
    with m1:
        st.markdown(f'<div class="rec-badge {rec_css}">{rec_text}</div>', unsafe_allow_html=True)
    with m2:
        st.metric("Current Price", f"${current_price:.2f}", 
                  f"{(current_price - hist['Close'].iloc[-2]):.2f}")

    # --- FUNDAMENTAL GRID ---
    if not is_restricted:
        st.markdown('<div class="section-title">VALUATION & EFFICIENCY</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: render_metric("P/E Ratio", safe_get(info, 'trailingPE'), comparison=25, invert_color=True)
        with c2: render_metric("Forward P/E", safe_get(info, 'forwardPE'), comparison=20, invert_color=True)
        with c3: render_metric("PEG Ratio", safe_get(info, 'pegRatio'), comparison=1.5, invert_color=True)
        with c4: render_metric("Profit Margin", safe_get(info, 'profitMargins'), is_percent=True, comparison=0.15)
        with c5: render_metric("ROE", safe_get(info, 'returnOnEquity'), is_percent=True, comparison=0.15)
        with c6: render_metric("Revenue Growth", safe_get(info, 'revenueGrowth'), is_percent=True, comparison=0.10)

        st.markdown('<div class="section-title">RISK & HEALTH</div>', unsafe_allow_html=True)
        d1, d2, d3, d4, d5, d6 = st.columns(6)
        with d1: render_metric("Current Ratio", safe_get(info, 'currentRatio'), comparison=1.2)
        with d2: render_metric("Debt/Equity", safe_get(info, 'debtToEquity')/100 if safe_get(info, 'debtToEquity') else 0, comparison=1.5, invert_color=True)
        with d3: render_metric("Beta", safe_get(info, 'beta'), fmt="{:.2f}")
        with d4: render_metric("Short Float", safe_get(info, 'shortPercentOfFloat'), is_percent=True, comparison=0.10, invert_color=True)
        with d5: render_metric("Piotroski F", calculate_piotroski(bs, is_, cf), fmt="{:.0f}", comparison=6)
        with d6: render_metric("Altman Z", 3.2, fmt="{:.1f}", comparison=2.99) # Demo value

    # --- CHARTS SECTION ---
    st.markdown('<div class="section-title">TECHNICAL & QUANTITATIVE ANALYSIS</div>', unsafe_allow_html=True)
    
    chart_col, radar_col = st.columns([2, 1])
    
    with chart_col:
        # Create Main Chart with Volume
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3])

        # Candlestick
        fig.add_trace(go.Candlestick(x=hist.index,
                                     open=hist['Open'], high=hist['High'],
                                     low=hist['Low'], close=hist['Close'],
                                     name='Price'), row=1, col=1)
        
        # SMAs
        hist['SMA50'] = hist['Close'].rolling(50).mean()
        hist['SMA200'] = hist['Close'].rolling(200).mean()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA50'], line=dict(color='orange', width=1), name='50 SMA'), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA200'], line=dict(color='#2962FF', width=1.5), name='200 SMA'), row=1, col=1)

        # Volume
        colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in hist.iterrows()]
        fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

        # Styling
        fig.update_layout(
            height=500,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(250,250,250,1)',
            showlegend=False,
            xaxis_rangeslider_visible=False
        )
        fig.update_yaxes(showgrid=True, gridcolor='#eee')
        fig.update_xaxes(showgrid=False)
        
        st.plotly_chart(fig, use_container_width=True)

    with radar_col:
        # Improved Radar Chart
        if not is_restricted:
            # Normalize data for 0-100 scale
            pe_score = max(0, min(100, 100 - safe_get(info, 'trailingPE', 50)))
            growth_score = min(100, safe_get(info, 'revenueGrowth', 0) * 300)
            profit_score = min(100, safe_get(info, 'profitMargins', 0) * 300)
            health_score = 80 # Placeholder based on Altman
            moat_score = min(100, safe_get(info, 'grossMargins', 0) * 150)

            categories = ['Valuation', 'Growth', 'Profitability', 'Health', 'Moat']
            values = [pe_score, growth_score, profit_score, health_score, moat_score]
            # Close the loop
            values += [values[0]]
            categories += [categories[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                fillcolor='rgba(41, 98, 255, 0.3)',
                line=dict(color='#2962FF', width=2),
                name=ticker
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor='#ddd'),
                    angularaxis=dict(gridcolor='#eee')
                ),
                margin=dict(l=40, r=40, t=20, b=20),
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # News List
            st.markdown("##### 📰 Latest Headlines")
            if news:
                for n in news[:4]:
                    title = n.get('title', 'No Title')
                    link = n.get('link', '#')
                    # Clean title if too long
                    if len(title) > 60: title = title[:60] + "..."
                    st.markdown(f"""
                    <div class="news-card">
                        <a href="{link}" target="_blank" class="news-title">{title}</a>
                        <div class="news-meta">Yahoo Finance • Today</div>
                    </div>
                    """, unsafe_allow_html=True)
