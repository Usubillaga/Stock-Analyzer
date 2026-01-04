import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

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
    .news-item {
        font-size: 0.9rem;
        padding: 5px 0;
        border-bottom: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# --- CACHING FUNCTION (The Fix for Rate Limits) ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_stock_data(symbol):
    """
    Fetches all necessary data at once to avoid repeated API calls.
    """
    try:
        stock = yf.Ticker(symbol)
        
        # Trigger data downloads inside the cache function
        info = stock.info
        hist = stock.history(period="5y")
        news = stock.news
        bs = stock.balance_sheet
        financials = stock.financials
        cashflow = stock.cashflow
        
        return info, hist, news, bs, financials, cashflow
    except Exception as e:
        return None, None, None, None, None, None

# --- Helper Functions (Updated to accept DataFrames directly) ---

def calculate_piotroski_f_score(bs, is_, cf):
    """Calculates the Piotroski F-Score (0-9) using provided dataframes."""
    try:
        # Need at least 2 years of data
        if bs.shape[1] < 2 or is_.shape[1] < 2 or cf.shape[1] < 2:
            return 5

        curr = 0
        prev = 1
        score = 0
        
        # 1. Profitability
        try: net_income = is_.loc['Net Income'].iloc[curr]; score += 1 if net_income > 0 else 0
        except: pass
        
        try: roa = is_.loc['Net Income'].iloc[curr] / bs.loc['Total Assets'].iloc[curr]; score += 1 if roa > 0 else 0
        except: pass
        
        try: cfo = cf.loc['Operating Cash Flow'].iloc[curr]; score += 1 if cfo > 0 else 0
        except: pass
        
        try: score += 1 if cfo > net_income else 0
        except: pass
        
        # 2. Leverage/Liquidity
        try: 
            lt_debt_curr = bs.loc['Long Term Debt'].iloc[curr] if 'Long Term Debt' in bs.index else 0
            lt_debt_prev = bs.loc['Long Term Debt'].iloc[prev] if 'Long Term Debt' in bs.index else 0
            score += 1 if lt_debt_curr < lt_debt_prev else 0
        except: pass
        
        try:
            cur_ratio_curr = bs.loc['Current Assets'].iloc[curr] / bs.loc['Current Liabilities'].iloc[curr]
            cur_ratio_prev = bs.loc['Current Assets'].iloc[prev] / bs.loc['Current Liabilities'].iloc[prev]
            score += 1 if cur_ratio_curr > cur_ratio_prev else 0
        except: pass
        
        try:
            shares_curr = bs.loc['Ordinary Shares Number'].iloc[curr]
            shares_prev = bs.loc['Ordinary Shares Number'].iloc[prev]
            score += 1 if shares_curr <= shares_prev else 0
        except: pass
        
        # 3. Efficiency
        try:
            gm_curr = (is_.loc['Total Revenue'].iloc[curr] - is_.loc['Cost Of Revenue'].iloc[curr]) / is_.loc['Total Revenue'].iloc[curr]
            gm_prev = (is_.loc['Total Revenue'].iloc[prev] - is_.loc['Cost Of Revenue'].iloc[prev]) / is_.loc['Total Revenue'].iloc[prev]
            score += 1 if gm_curr > gm_prev else 0
        except: pass
        
        try:
            asset_turn_curr = is_.loc['Total Revenue'].iloc[curr] / bs.loc['Total Assets'].iloc[curr]
            asset_turn_prev = is_.loc['Total Revenue'].iloc[prev] / bs.loc['Total Assets'].iloc[prev]
            score += 1 if asset_turn_curr > asset_turn_prev else 0
        except: pass

        return score
    except Exception:
        return 5

def calculate_altman_z_score(bs, is_, info):
    """Calculates Altman Z-Score."""
    try:
        if bs.empty or is_.empty: return 0

        # Most recent data
        total_assets = bs.loc['Total Assets'].iloc[0]
        curr_assets = bs.loc['Current Assets'].iloc[0]
        curr_liab = bs.loc['Current Liabilities'].iloc[0]
        working_cap = curr_assets - curr_liab
        retained_earnings = bs.loc['Retained Earnings'].iloc[0] if 'Retained Earnings' in bs.index else 0
        ebit = is_.loc['EBIT'].iloc[0] if 'EBIT' in is_.index else is_.loc['Net Income'].iloc[0]
        market_cap = info.get('marketCap', 0)
        total_liabilities = bs.loc['Total Liabilities Net Minority Interest'].iloc[0]
        revenue = is_.loc['Total Revenue'].iloc[0]

        A = working_cap / total_assets
        B = retained_earnings / total_assets
        C = ebit / total_assets
        D = market_cap / total_liabilities
        E = revenue / total_assets

        z_score = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
        return z_score
    except:
        return 0

def get_intrinsic_value(info):
    try:
        eps = info.get('trailingEps', 0)
        book_value = info.get('bookValue', 0)
        if eps > 0 and book_value > 0:
            return (22.5 * eps * book_value) ** 0.5
    except:
        pass
    return info.get('currentPrice', 0)

# --- UI Components ---

def render_metric_card(label, value, comparison=None, fmt="{:.2f}", color_logic="high_good"):
    color_class = ""
    if comparison is not None:
        if color_logic == "high_good":
            color_class = "positive" if value > comparison else "negative"
        elif color_logic == "low_good":
            color_class = "positive" if value < comparison else "negative"
            
    if label == "Piotroski F":
        color_class = "positive" if value >= 7 else ("negative" if value <= 3 else "neutral")
    elif label == "Altman Z":
        color_class = "positive" if value > 2.99 else ("negative" if value < 1.8 else "neutral")

    display_value = value
    if isinstance(value, (float, int)):
        display_value = fmt.format(value)

    html = f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{display_value}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# --- Main App ---

st.title("Equity Research Dashboard")
st.caption("Replicating professional institutional analysis sheets using real-time data.")

col1, col2 = st.columns([1, 3])
with col1:
    ticker = st.text_input("Enter Ticker Symbol", value="NVO").upper()
with col2:
    st.write("") # Spacer

if ticker:
    # --- FETCH DATA WITH CACHING ---
    info, hist, news, bs, financials, cashflow = fetch_stock_data(ticker)
    
    if info is None or 'currentPrice' not in info:
        st.error("Error: Unable to fetch data. Yahoo Finance may be rate-limiting requests. Please try again in 5 minutes.")
        st.stop()
            
    # --- HEADER ---
    st.markdown(f"## {ticker} - {info.get('longName', 'N/A')}")
    st.markdown(f"**Report Date:** {datetime.now().strftime('%Y-%m-%d')} | **Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
    
    st.divider()

    # --- DATA PREP ---
    price = info.get('currentPrice', 0)
    mkt_cap = info.get('marketCap', 0)
    pe_ratio = info.get('trailingPE', 0)
    fwd_pe = info.get('forwardPE', 0)
    peg = info.get('pegRatio', 0)
    
    # Calculate Scores
    piotroski = calculate_piotroski_f_score(bs, financials, cashflow)
    altman = calculate_altman_z_score(bs, financials, info)
    intrinsic_val = get_intrinsic_value(info)
    
    # --- METRICS GRID ---
    st.markdown('<div class="section-header">Price & Valuation / Financial Performance</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    
    with c1: render_metric_card("Price", price, fmt="${:.2f}")
    with c2: render_metric_card("Market Cap", mkt_cap / 1e9, fmt="${:.1f}B")
    with c3: render_metric_card("Trailing P/E", pe_ratio, comparison=20, color_logic="low_good")
    with c4: render_metric_card("Forward P/E", fwd_pe, comparison=20, color_logic="low_good")
    with c5: render_metric_card("PEG Ratio", peg, comparison=1, color_logic="low_good")
    with c6: render_metric_card("ROE", info.get('returnOnEquity', 0)*100, comparison=15, fmt="{:.1f}%")
    with c7: render_metric_card("Profit Margin", info.get('profitMargins', 0)*100, comparison=10, fmt="{:.1f}%")
    with c8: render_metric_card("ROIC", info.get('returnOnAssets', 0)*100, comparison=8, fmt="{:.1f}%")

    st.markdown('<div class="section-header">Risk Indicators & Quality Scores</div>', unsafe_allow_html=True)
    r2_1, r2_2, r2_3, r2_4, r2_5, r2_6, r2_7, r2_8 = st.columns(8)
    
    with r2_1: render_metric_card("Current Ratio", info.get('currentRatio', 0), comparison=1.5)
    with r2_2: render_metric_card("Debt/Equity", info.get('debtToEquity', 0)/100, comparison=1.0, color_logic="low_good")
    with r2_3: render_metric_card("Beta", info.get('beta', 1))
    with r2_4: render_metric_card("Piotroski F", piotroski)
    with r2_5: render_metric_card("Altman Z", altman, fmt="{:.2f}")
    with r2_6: render_metric_card("Short Float", info.get('shortPercentOfFloat', 0)*100, comparison=5, color_logic="low_good", fmt="{:.2f}%")
    with r2_7: render_metric_card("Intrinsic Val", intrinsic_val, comparison=price, fmt="${:.2f}")
    with r2_8: 
        gap = ((intrinsic_val - price) / price) * 100
        render_metric_card("Upside", gap, comparison=0, fmt="{:.1f}%")

    # --- CHARTS SECTION ---
    st.divider()
    chart_col1, chart_col2 = st.columns([2, 1])

    with chart_col1:
        st.subheader("Linear Price Chart & Fair Value")
        
        fig_price = go.Figure()
        fig_price.add_trace(go.Candlestick(x=hist.index,
                        open=hist['Open'], high=hist['High'],
                        low=hist['Low'], close=hist['Close'],
                        name='Market Price'))
        
        fig_price.add_trace(go.Scatter(x=hist.index, y=[intrinsic_val]*len(hist), 
                                 mode='lines', name='Intrinsic Value',
                                 line=dict(color='green', dash='dash')))
        
        hist['SMA50'] = hist['Close'].rolling(window=50).mean()
        hist['SMA200'] = hist['Close'].rolling(window=200).mean()
        fig_price.add_trace(go.Scatter(x=hist.index, y=hist['SMA50'], name='50 SMA', line=dict(width=1)))
        fig_price.add_trace(go.Scatter(x=hist.index, y=hist['SMA200'], name='200 SMA', line=dict(width=1)))

        fig_price.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_price, use_container_width=True)

        st.subheader("Technical Indicators")
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
        exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD'] = exp1 - exp2
        hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()

        fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.5, 0.5])
        
        fig_tech.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name='RSI', line=dict(color='purple')), row=1, col=1)
        fig_tech.add_hline(y=70, line_dash="dot", row=1, col=1, line_color="red")
        fig_tech.add_hline(y=30, line_dash="dot", row=1, col=1, line_color="green")
        
        fig_tech.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
        fig_tech.add_trace(go.Scatter(x=hist.index, y=hist['Signal'], name='Signal', line=dict(color='orange')), row=2, col=1)
        
        fig_tech.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_tech, use_container_width=True)

    with chart_col2:
        st.subheader("Analysis Spider Chart")
        
        def normalize(val, min_v, max_v):
            return min(100, max(0, (val - min_v) / (max_v - min_v) * 100))

        scores = {
            'Valuation': normalize(1/pe_ratio if pe_ratio > 0 else 0, 0, 0.1),
            'Growth': normalize(info.get('revenueGrowth', 0), 0, 0.3),
            'Profitability': normalize(info.get('returnOnEquity', 0), 0, 0.2),
            'Financial Health': normalize(altman, 0, 3),
            'Moat/Margins': normalize(info.get('grossMargins', 0), 0, 0.6)
        }
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=list(scores.values()),
            theta=list(scores.keys()),
            fill='toself',
            name=ticker
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            height=350,
            margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.subheader("Top News")
        if news:
            for n in news[:5]:
                title = n.get('title', 'No Title')
                link = n.get('link', '#')
                pub_date = datetime.fromtimestamp(n.get('providerPublishTime', 0)).strftime('%Y-%m-%d')
                st.markdown(f"""
                <div class="news-item">
                    <a href="{link}" target="_blank" style="text-decoration:none; color:#000; font-weight:bold;">{title}</a>
                    <br><span style="color:#888; font-size:0.8rem;">{pub_date}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("No news found.")
