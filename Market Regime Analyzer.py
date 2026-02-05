import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

# ==========================================
# 1. UI CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="Titan: Global Macro & Technical Engine")

st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #333333; }
    div[data-testid="stMetric"] { background-color: #F8F9FA; border: 1px solid #E9ECEF; padding: 15px; border-radius: 8px; }
    .card { background-color: #F8F9FA; padding: 20px; border-radius: 10px; border: 1px solid #E0E0E0; margin-bottom: 20px; }
    .bull-accent { border-left: 5px solid #28a745; }
    .bear-accent { border-left: 5px solid #dc3545; }
    .neutral-accent { border-left: 5px solid #ffc107; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ROBUST DATA ENGINE (FIXED)
# ==========================================

@st.cache_data(ttl=3600)
def get_macro_data():
    """
    Fetches macro assets.
    FIX: Switched Futures (CL=F) to ETFs (USO) for reliability.
    FIX: Flattens MultiIndex columns to prevent empty dataframes.
    """
    tickers = {
        # Changed to ETFs for better stability on free API
        "Oil": "USO", "Gold": "GLD", "Copper": "CPER", "10Y Yield": "^TNX", "VIX": "^VIX",
        "S&P 500": "SPY",
        # Sectors
        "Tech": "XLK", "Financials": "XLF", "Energy": "XLE", "Healthcare": "XLV",
        "Discretionary": "XLY", "Staples": "XLP", "Materials": "XLB", "Utilities": "XLU",
        "Industrials": "XLI", "Real Estate": "XLRE", "Comms": "XLC"
    }
    
    # Download with auto_adjust to get proper closing prices
    try:
        df = yf.download(list(tickers.values()), period="6mo", progress=False, group_by='ticker', auto_adjust=True)
        
        # FIX: Flatten the confusing MultiIndex structure from yfinance
        # yfinance often returns: ('AAPL', 'Close'), ('AAPL', 'Volume')
        # We just want a simple DataFrame where columns are the Tickers and values are Close prices.
        
        close_data = pd.DataFrame()
        
        for friendly_name, ticker_symbol in tickers.items():
            try:
                # Try fetching the 'Close' column for this ticker
                # Depending on yfinance version, it might be at top level or second level
                if ticker_symbol in df.columns.levels[0]:
                     series = df[ticker_symbol]['Close']
                else:
                     # Fallback if structure is different
                     series = df['Close'][ticker_symbol]
                
                close_data[friendly_name] = series
            except Exception:
                pass # Skip if one ticker fails
        
        # Fill missing values to prevent empty charts
        close_data.fillna(method='ffill', inplace=True)
        return close_data
        
    except Exception as e:
        st.error(f"Data connection failed: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_market_universe(choice):
    """Fetches tickers or returns a robust default list if scraping fails."""
    try:
        if choice == "S&P 500":
            payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            return [x.replace('.', '-') for x in payload['Symbol'].tolist()]
        elif choice == "Nasdaq 100":
            payload = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
            for t in payload:
                if 'Ticker' in t.columns: return [x.replace('.', '-') for x in t['Ticker'].tolist()]
    except:
        pass
    
    # Robust Fallback List
    return ["NVDA", "TSLA", "AAPL", "AMD", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "COIN", "MSTR", "JPM", "XOM", "LLY"]

@st.cache_data(ttl=3600)
def batch_technical_analysis(ticker_list):
    if not ticker_list: return {}
    
    results = {}
    
    # Process in smaller chunks to avoid timeouts
    chunk_size = 20
    for i in range(0, len(ticker_list), chunk_size):
        chunk = ticker_list[i:i+chunk_size]
        try:
            df_bulk = yf.download(chunk, period="1y", group_by='ticker', progress=False, threads=True, auto_adjust=True)
            
            for t in chunk:
                try:
                    # Extract single dataframe safely
                    if len(chunk) > 1:
                        if t not in df_bulk.columns.levels[0]: continue
                        df = df_bulk[t].copy()
                    else:
                        df = df_bulk.copy()
                    
                    df = df.dropna(subset=['Close'])
                    if len(df) < 50: continue
                    
                    # Technicals
                    df['SMA_50'] = ta.sma(df['Close'], length=50)
                    df['SMA_200'] = ta.sma(df['Close'], length=200)
                    df['RSI_14'] = ta.rsi(df['Close'], length=14)
                    df['ADX_14'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
                    
                    bb = ta.bbands(df['Close'], length=20, std=2)
                    df['BB_Upper'] = bb['BBU_20_2.0']
                    df['BB_Lower'] = bb['BBL_20_2.0']
                    
                    results[t] = df
                except: continue
        except: continue
        
    return results

# ==========================================
# 3. ANALYSIS LOGIC
# ==========================================

def analyze_sectors(macro_df):
    if macro_df.empty: return {}
    
    curr = macro_df.iloc[-1]
    # Handle cases where data is too short
    idx = -22 if len(macro_df) > 22 else 0
    start = macro_df.iloc[idx] 
    
    returns = ((curr - start) / start) * 100
    
    # Safe Alpha Calculation
    if 'S&P 500' in returns:
        spy_ret = returns['S&P 500']
        alpha = returns - spy_ret
    else:
        alpha = returns # Fallback
    
    # Safe Access to keys
    def get_ret(key): return returns.get(key, 0)
    
    risk_ratio = get_ret('Discretionary') - get_ret('Staples')
    inflation_pressure = get_ret('Energy') + get_ret('Materials')
    
    return {
        "returns": returns, "alpha": alpha, "risk_on": risk_ratio > 0,
        "inflationary": inflation_pressure > 5.0,
        "leader": returns.idxmax(), "laggard": returns.idxmin()
    }

def get_market_narrative(analysis, macro_df):
    if not analysis: return ["⚠️ Data insufficient for analysis."]
    
    narrative = []
    if analysis['risk_on']:
        narrative.append("🟢 **Risk-On (Bullish):** Investors are buying Growth/Cyclicals.")
        narrative.append("👉 **Strategy:** Buy dips in Tech & Discretionary.")
    else:
        narrative.append("🔴 **Risk-Off (Defensive):** Investors are fleeing to Staples/Utilities.")
        narrative.append("👉 **Strategy:** Defensive stocks or Cash.")

    if '^TNX' in macro_df.columns:
        yield_val = macro_df['^TNX'].iloc[-1]
        if yield_val > 4.5: narrative.append(f"⚠️ **High Rates ({yield_val:.2f}%):** Headwind for Tech/Real Estate.")
    
    return narrative

def score_stock(df):
    row = df.iloc[-1]
    score_bull = 0
    score_bear = 0
    
    # Safe checks
    if pd.isna(row['SMA_200']): return 0, 0
    
    # Bullish
    if row['Close'] > row['SMA_200']: score_bull += 30
    if row['Close'] > row['SMA_50']: score_bull += 10
    if 40 < row['RSI_14'] < 60: score_bull += 20
    if row['Close'] > row['BB_Upper']: score_bull += 20
    
    # Bearish
    if row['Close'] < row['SMA_200']: score_bear += 30
    if row['Close'] < row['SMA_50']: score_bear += 10
    if row['RSI_14'] > 70: score_bear += 20
    if row['Close'] < row['BB_Lower']: score_bear += 20
    
    return score_bull, score_bear

# ==========================================
# 4. DASHBOARD LAYOUT
# ==========================================

st.title("🦅 TITAN | Macro-Quant Analytics")
st.markdown("---")

# Load Data
with st.spinner("Connecting to Market Data Feeds..."):
    macro_data = get_macro_data()
    
    if macro_data.empty:
        st.error("❌ Failed to load Market Data. Yahoo Finance API might be blocking requests. Try again later.")
        st.stop()
        
    sector_analysis = analyze_sectors(macro_data)
    narrative = get_market_narrative(sector_analysis, macro_data)

# TABS
tab_macro, tab_scanner = st.tabs(["🌍 Macro Headquarters", "🔭 Stock Sniper Scope"])

# --- TAB 1: MACRO ---
with tab_macro:
    # Safe Metrics
    def safe_metric(label, key, is_currency=False):
        if key in macro_data.columns:
            val = macro_data[key].iloc[-1]
            prev = macro_data[key].iloc[-2]
            delta = (val - prev) / prev * 100
            fmt = f"${val:.2f}" if is_currency else f"{val:.2f}"
            col.metric(label, fmt, f"{delta:.2f}%")
        else:
            col.metric(label, "N/A", "0%")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: col = st.empty(); safe_metric("S&P 500", "S&P 500")
    with col2: col = st.empty(); safe_metric("10Y Yield", "10Y Yield")
    with col3: col = st.empty(); safe_metric("Oil (USO)", "Oil", True)
    with col4: col = st.empty(); safe_metric("VIX", "VIX")
    with col5: col = st.empty(); safe_metric("Gold (GLD)", "Gold", True)

    st.divider()
    
    c_chart, c_text = st.columns([2, 1])
    
    with c_chart:
        st.markdown("#### Sector Relative Strength (vs SPY)")
        if not sector_analysis:
            st.warning("Chart Unavailable")
        else:
            sector_cols = ["Tech", "Financials", "Energy", "Healthcare", "Discretionary", 
                           "Staples", "Materials", "Utilities", "Industrials", "Real Estate", "Comms"]
            # Filter cols that actually exist in our data
            valid_cols = [c for c in sector_cols if c in sector_analysis['alpha']]
            
            if valid_cols:
                alpha_data = sector_analysis['alpha'][valid_cols].sort_values()
                colors = ['#dc3545' if x < 0 else '#28a745' for x in alpha_data.values]
                
                fig = go.Figure(go.Bar(x=alpha_data.values, y=alpha_data.index, orientation='h', marker_color=colors))
                fig.update_layout(template="plotly_white", margin=dict(l=0,r=0,t=20,b=0), height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with c_text:
        st.markdown("<div class='card neutral-accent'>", unsafe_allow_html=True)
        st.markdown("### 🧠 AI Analyst")
        for line in narrative: st.markdown(line)
        st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 2: SCANNER ---
with tab_scanner:
    with st.expander("⚙️ Scanner Settings", expanded=True):
        c1, c2, c3 = st.columns(3)
        universe = c1.selectbox("Universe", ["Nasdaq 100", "S&P 500", "Custom List"])
        custom_txt = ""
        if universe == "Custom List":
            custom_txt = c1.text_area("Tickers", "NVDA, TSLA, AMD, AAPL, MSFT, COIN, MSTR")
        min_score = c2.slider("Min Score", 0, 100, 70)
        max_stocks = c3.number_input("Max Stocks", 10, 500, 50)
        run_btn = st.button("🚀 INITIATE SCAN", type="primary", use_container_width=True)

    if run_btn:
        st.info("📡 Scanning...")
        if universe == "Custom List":
            tickers = [x.strip().upper() for x in custom_txt.split(',')]
        else:
            tickers = get_market_universe(universe)[:max_stocks]
            
        data_dict = batch_technical_analysis(tickers)
        
        bull_list, bear_list = [], []
        
        progress = st.progress(0)
        for i, (t, df) in enumerate(data_dict.items()):
            b, s = score_stock(df)
            last = df['Close'].iloc[-1]
            if b >= min_score: bull_list.append({"Ticker": t, "Score": b, "Price": last})
            if s >= min_score: bear_list.append({"Ticker": t, "Score": s, "Price": last})
            progress.progress((i+1)/len(data_dict))
        progress.empty()
        
        c_bull, c_bear = st.columns(2)
        
        with c_bull:
            st.markdown("<div class='card bull-accent'><h3>🐂 Bullish</h3></div>", unsafe_allow_html=True)
            if bull_list:
                df_b = pd.DataFrame(bull_list).sort_values("Score", ascending=False)
                st.dataframe(df_b.style.background_gradient(subset=["Score"], cmap="Greens"), use_container_width=True, hide_index=True)
                
                # Chart top pick
                top = df_b.iloc[0]['Ticker']
                df_p = data_dict[top].tail(100)
                fig = go.Figure(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close']))
                fig.update_layout(title=f"Top Bull: {top}", template="plotly_white", height=350, margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No results.")
                
        with c_bear:
            st.markdown("<div class='card bear-accent'><h3>🐻 Bearish</h3></div>", unsafe_allow_html=True)
            if bear_list:
                df_s = pd.DataFrame(bear_list).sort_values("Score", ascending=False)
                st.dataframe(df_s.style.background_gradient(subset=["Score"], cmap="Reds"), use_container_width=True, hide_index=True)
                
                top = df_s.iloc[0]['Ticker']
                df_p = data_dict[top].tail(100)
                fig = go.Figure(go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close']))
                fig.update_layout(title=f"Top Bear: {top}", template="plotly_white", height=350, margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No results.")
