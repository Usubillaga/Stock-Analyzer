import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. CLEAN LIGHT-THEME UI CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="Titan: Global Macro & Technical Engine (Light)")

# Custom CSS for a clean, professional Light Theme
st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background-color: #FFFFFF;
        color: #333333;
    }
    
    /* Styling for metric boxes to make them pop against white */
    div[data-testid="stMetric"] {
        background-color: #F8F9FA;
        border: 1px solid #E9ECEF;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricValue"] { color: #212529 !important; font-weight: 600; }
    div[data-testid="stMetricLabel"] { color: #6C757D !important; }
    
    /* Custom Card Containers */
    .card {
        background-color: #F8F9FA;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #E0E0E0;
        margin-bottom: 20px;
    }
    /* Accents for status */
    .bull-accent { border-left: 5px solid #28a745; }
    .bear-accent { border-left: 5px solid #dc3545; }
    .neutral-accent { border-left: 5px solid #ffc107; }
    
    /* Header styling */
    h1, h2, h3, h4 { color: #333333 !important; font-weight: 600; }
    .highlight { color: #007bff; font-weight: bold; }
    
    /* Dataframe adjustments */
    .stDataFrame { border: 1px solid #E0E0E0; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ADVANCED DATA ENGINE (UNCHANGED)
# ==========================================

@st.cache_data(ttl=3600)
def get_macro_data():
    """Fetches macro assets: Commodities, Yields, and 11 Sector ETFs."""
    tickers = {
        # Commodities & Yields
        "Oil": "CL=F", "Gold": "GC=F", "Copper": "HG=F", "10Y Yield": "^TNX", "VIX": "^VIX",
        # Benchmark
        "S&P 500": "SPY",
        # Sectors
        "Tech": "XLK", "Financials": "XLF", "Energy": "XLE", "Healthcare": "XLV",
        "Discretionary": "XLY", "Staples": "XLP", "Materials": "XLB", "Utilities": "XLU",
        "Industrials": "XLI", "Real Estate": "XLRE", "Comms": "XLC"
    }
    df = yf.download(list(tickers.values()), period="6mo", progress=False)['Close']
    rev_map = {v: k for k, v in tickers.items()}
    df.columns = [rev_map.get(c, c) for c in df.columns]
    return df

@st.cache_data(ttl=86400)
def get_market_universe(choice):
    """Fetches tickers for S&P 500 or Nasdaq 100."""
    try:
        if choice == "S&P 500":
            payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            return [x.replace('.', '-') for x in payload['Symbol'].tolist()]
        elif choice == "Nasdaq 100":
            payload = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
            for t in payload:
                if 'Ticker' in t.columns: return [x.replace('.', '-') for x in t['Ticker'].tolist()]
                if 'Symbol' in t.columns: return [x.replace('.', '-') for x in t['Symbol'].tolist()]
    except:
        return ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD"] # Fallback

@st.cache_data(ttl=3600)
def batch_technical_analysis(ticker_list):
    """Massive batch processor for technicals."""
    if not ticker_list: return {}
    df_bulk = yf.download(ticker_list, period="1y", group_by='ticker', progress=False, threads=True)
    results = {}
    
    for t in ticker_list:
        try:
            df = df_bulk[t].copy() if len(ticker_list) > 1 else df_bulk.copy()
            if df.empty or df['Close'].isnull().all(): continue
            df.dropna(inplace=True)
            if len(df) < 100: continue
            
            # --- CALCULATIONS ---
            # Structure
            df.ta.sma(length=50, append=True)
            df.ta.sma(length=200, append=True)
            # Momentum
            df.ta.rsi(length=14, append=True)
            # Volatility
            df.ta.bbands(length=20, std=2, append=True)
            df.ta.adx(length=14, append=True)
            # Volume
            df['Vol_SMA'] = df['Volume'].rolling(20).mean()
            df['RVol'] = df['Volume'] / df['Vol_SMA']
            
            # Friendly Names
            df['BB_Upper'] = df['BBU_20_2.0']
            df['BB_Lower'] = df['BBL_20_2.0']
            
            results[t] = df
        except: continue
    return results

# ==========================================
# 3. LOGIC & INTELLIGENCE (UNCHANGED)
# ==========================================

def analyze_sectors(macro_df):
    """Compares Sector Performance relative to SPY (Alpha)."""
    curr = macro_df.iloc[-1]
    start = macro_df.iloc[-22] # 1 month ago
    returns = ((curr - start) / start) * 100
    
    spy_ret = returns['S&P 500']
    alpha = returns - spy_ret
    
    risk_ratio = returns['Discretionary'] - returns['Staples']
    inflation_pressure = returns['Energy'] + returns['Materials']
    
    analysis = {
        "returns": returns, "alpha": alpha, "risk_on": risk_ratio > 0,
        "inflationary": inflation_pressure > 5.0,
        "leader": returns.idxmax(), "laggard": returns.idxmin()
    }
    return analysis

def get_market_narrative(analysis, macro_df):
    """Generates the AI textual analysis."""
    narrative = []
    # Risk Sentiment
    if analysis['risk_on']:
        narrative.append("🟢 **Risk-On Environment:** Investors are favoring Cyclicals (Discretionary) over Staples. This suggests confidence.")
        narrative.append("👉 **Approach:** Look for pullbacks in Tech/Discretionary to go Long.")
    else:
        narrative.append("🔴 **Risk-Off / Defensive:** Investors hiding in Staples/Utilities. Fear is present.")
        narrative.append("👉 **Approach:** Avoid High Beta. Look for shorts in Discretionary or longs in Defensive sectors.")

    # Inflation/Rates
    yield_val = macro_df['10Y Yield'].iloc[-1]
    if yield_val > 4.5:
        narrative.append(f"⚠️ **High Rate Alert ({yield_val:.2f}%):** Yields are pressuring valuations in Real Estate & Tech.")
    
    if analysis['inflationary']:
        narrative.append("🔥 **Inflation Trade Active:** Energy and Materials outperform. Look at Commodity stocks.")
        
    return narrative

def score_stock(df):
    """0-100 Score based on technical confluence."""
    row = df.iloc[-1]
    score_bull = 0
    score_bear = 0
    
    # --- BULLISH SCORING ---
    if row['Close'] > row['SMA_200']: score_bull += 30 # Structure
    if row['Close'] > row['SMA_50']: score_bull += 10
    if 40 < row['RSI_14'] < 60: score_bull += 20 # Momentum sweet spot
    if row['RSI_14'] < 30: score_bull += 10 # Oversold bounce potential
    if row['ADX_14'] > 25: score_bull += 10 # Trend strength
    if row['Close'] > row['BB_Upper'] * 0.99: score_bull += 20 # Breakout test
    
    # --- BEARISH SCORING ---
    if row['Close'] < row['SMA_200']: score_bear += 30
    if row['Close'] < row['SMA_50']: score_bear += 10
    if 40 < row['RSI_14'] < 60: score_bear += 20
    if row['RSI_14'] > 70: score_bear += 10 # Overbought
    if row['Close'] < row['BB_Lower'] * 1.01: score_bear += 20 # Breakdown test
    
    return score_bull, score_bear

# ==========================================
# 4. MAIN APP LAYOUT (RE-STYLED)
# ==========================================

st.title("🦅 TITAN | Macro-Quant Analytics")
st.markdown("---")

# Load Data
with st.spinner("Establishing connection to global market data..."):
    macro_data = get_macro_data()
    sector_analysis = analyze_sectors(macro_data)
    narrative = get_market_narrative(sector_analysis, macro_data)

# TABS for Clean UI
tab_macro, tab_scanner = st.tabs(["🌍 Macro Headquarters", "🔭 Stock Sniper Scope"])

# ==========================================
# TAB 1: MACRO HEADQUARTERS
# ==========================================
with tab_macro:
    st.subheader("Global Dashboard")
    # 1. Top Level Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    curr = macro_data.iloc[-1]
    prev = macro_data.iloc[-2]
    
    col1.metric("S&P 500", f"{curr['S&P 500']:.2f}", f"{(curr['S&P 500']/prev['S&P 500']-1)*100:.2f}%")
    col2.metric("10Y Yield", f"{curr['10Y Yield']:.2f}%", f"{(curr['10Y Yield']-prev['10Y Yield']):.2f}")
    col3.metric("Oil (WTI)", f"${curr['Oil']:.2f}", f"{(curr['Oil']/prev['Oil']-1)*100:.2f}%")
    col4.metric("VIX (Fear)", f"{curr['VIX']:.2f}", f"{(curr['VIX']-prev['VIX']):.2f}", delta_color="inverse")
    col5.metric("Gold", f"${curr['Gold']:.2f}", f"{(curr['Gold']/prev['Gold']-1)*100:.2f}%")

    st.divider()
    
    # 2. Sector Rotation Visualization & Narrative
    c_chart, c_text = st.columns([2, 1])
    
    with c_chart:
        st.markdown("#### Sector Relative Performance (1 Month Alpha vs SPY)")
        sector_cols = ["Tech", "Financials", "Energy", "Healthcare", "Discretionary", 
                       "Staples", "Materials", "Utilities", "Industrials", "Real Estate", "Comms"]
        alpha_data = sector_analysis['alpha'][sector_cols].sort_values(ascending=True)
        
        # Color logic for light mode (Green for leaders, Red for laggards)
        colors = ['#dc3545' if x < 0 else '#28a745' for x in alpha_data.values]
        
        fig = go.Figure(go.Bar(x=alpha_data.values, y=alpha_data.index, orientation='h', marker_color=colors))
        # Use a bright white template
        fig.update_layout(template="plotly_white", xaxis_title="Relative Strength (%)", height=400, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)
        
    with c_text:
        # Use the new card style defined in CSS
        st.markdown("<div class='card neutral-accent'>", unsafe_allow_html=True)
        st.markdown("### 🧠 AI Market Analyst")
        for line in narrative:
            st.markdown(line)
        st.divider()
        st.write(f"**Top Sector:** {sector_analysis['leader']} (+{sector_analysis['returns'][sector_analysis['leader']]:.1f}%)")
        st.write(f"**Weakest Link:** {sector_analysis['laggard']} ({sector_analysis['returns'][sector_analysis['laggard']]:.1f}%)")
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 2: STOCK SNIPER SCOPE
# ==========================================
with tab_scanner:
    
    # Controls in a clean light card
    with st.expander("⚙️ Scanner Controls & Settings", expanded=True):
        st.info("Select your universe and criteria to scan for high-confluence setups based on structure, momentum, and volatility.")
        c1, c2, c3 = st.columns(3)
        universe = c1.selectbox("1. Market Universe", ["Nasdaq 100", "S&P 500", "Custom List"])
        if universe == "Custom List":
            custom_txt = st.text_area("Tickers (comma separated)", "NVDA, TSLA, AMD, AAPL, MSFT, COIN")
        min_score = c2.slider("2. Min Confidence Score (0-100)", 0, 100, 75)
        max_stocks = c3.number_input("3. Speed Limit (Max Stocks)", 10, 600, 100, help="Lower for faster scans.")
        
        run_btn = st.button("🚀 INITIATE SCAN", type="primary", use_container_width=True)

    if run_btn:
        status_area = st.empty()
        status_area.info("📡 Initializing Scan... Downloading market data.")
        
        if universe == "Custom List":
            tickers = [x.strip().upper() for x in custom_txt.split(',')]
        else:
            tickers = get_market_universe(universe)[:max_stocks]
            
        data_dict = batch_technical_analysis(tickers)
        
        bull_list = []
        bear_list = []
        
        # Scoring Loop
        status_area.info(f"📊 Analyzing {len(data_dict)} assets against technical models...")
        progress = st.progress(0)
        for i, (t, df) in enumerate(data_dict.items()):
            b_score, s_score = score_stock(df)
            last_price = df['Close'].iloc[-1]
            rsi = df['RSI_14'].iloc[-1]
            adx = df['ADX_14'].iloc[-1]
            
            if b_score >= min_score:
                bull_list.append({"Ticker": t, "Score": b_score, "Price": last_price, "RSI": rsi, "ADX (Trend)": adx})
            if s_score >= min_score:
                bear_list.append({"Ticker": t, "Score": s_score, "Price": last_price, "RSI": rsi, "ADX (Trend)": adx})
            progress.progress((i+1)/len(data_dict))
            
        progress.empty()
        status_area.success("Scan Complete. Results grouped below.")
        
        # Display Results
        col_bull, col_bear = st.columns(2)
        
        # BULLISH RESULTS
        with col_bull:
            # Use the light theme card style with green accent
            st.markdown("<div class='card bull-accent'><h3>🐂 Bullish Opportunities</h3></div>", unsafe_allow_html=True)
            if bull_list:
                df_bull = pd.DataFrame(bull_list).sort_values("Score", ascending=False)
                st.dataframe(df_bull.style.background_gradient(subset=["Score"], cmap="Greens"), use_container_width=True, hide_index=True)
                
                # Charting Best Bull
                top_bull = df_bull.iloc[0]['Ticker']
                st.markdown(f"**Deep Dive: {top_bull}**")
                df_p = data_dict[top_bull].tail(120)
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name="Price"))
                fig.add_trace(go.Scatter(x=df_p.index, y=df_p['BB_Upper'], line=dict(color='green', width=1, dash='dot'), name="Upper Band"))
                fig.add_trace(go.Scatter(x=df_p.index, y=df_p['SMA_200'], line=dict(color='blue', width=2), name="200 SMA"))
                # Use plotly_white template
                fig.update_layout(title=f"{top_bull} Technicals", height=400, template="plotly_white", margin=dict(l=0,r=0,t=40,b=0), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No stocks met the minimum bullish criteria.")

        # BEARISH RESULTS
        with col_bear:
            # Use the light theme card style with red accent
            st.markdown("<div class='card bear-accent'><h3>🐻 Bearish Opportunities</h3></div>", unsafe_allow_html=True)
            if bear_list:
                df_bear = pd.DataFrame(bear_list).sort_values("Score", ascending=False)
                st.dataframe(df_bear.style.background_gradient(subset=["Score"], cmap="Reds"), use_container_width=True, hide_index=True)
                
                # Charting Best Bear
                top_bear = df_bear.iloc[0]['Ticker']
                st.markdown(f"**Deep Dive: {top_bear}**")
                df_p = data_dict[top_bear].tail(120)
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name="Price"))
                fig.add_trace(go.Scatter(x=df_p.index, y=df_p['BB_Lower'], line=dict(color='red', width=1, dash='dot'), name="Lower Band"))
                fig.add_trace(go.Scatter(x=df_p.index, y=df_p['SMA_200'], line=dict(color='blue', width=2), name="200 SMA"))
                # Use plotly_white template
                fig.update_layout(title=f"{top_bear} Technicals", height=400, template="plotly_white", margin=dict(l=0,r=0,t=40,b=0), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No stocks met the minimum bearish criteria.")
