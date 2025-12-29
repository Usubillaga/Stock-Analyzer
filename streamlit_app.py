import streamlit as st
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Stock Analyzer with DCF", layout="centered")

st.title("📈 Stock Fundamental Analyzer with DCF Valuation")
st.markdown("Enter a ticker to get key metrics, analyst growth forecast, and a simple DCF valuation.")

# Sidebar inputs
ticker = st.text_input("Ticker symbol (e.g. AAPL, TSLA, MSFT)", "AAPL").upper().strip()

col1, col2 = st.columns(2)
with col1:
    default_growth = st.number_input("Override analyst growth rate (%)", min_value=0.0, value=0.0, help="Leave 0 to use analyst forecast")
with col2:
    perp_growth = st.number_input("Perpetual growth rate (%)", min_value=0.0, max_value=10.0, value=2.0) / 100

if st.button("Analyze"):
    with st.spinner("Fetching data..."):
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or 'longName' not in info:
            st.error("Invalid ticker or no data available.")
            st.stop()

        # Header
        st.header(f"{info.get('longName', ticker)} ({ticker})")
        st.caption(f"Sector: {info.get('sector', 'N/A')} • Updated: {datetime.now().strftime('%Y-%m-%d')}")

        price = info.get('currentPrice') or info.get('regularMarketPrice')
        st.metric("Current Price", f"${price:.2f}" if price else "N/A")

        # Key metrics
        st.subheader("Key Financial Metrics")
        cols = st.columns(4)
        metrics = [
            ("Gross Margin", info.get('grossMargins'), "{:.1%}"),
            ("Net Margin", info.get('profitMargins'), "{:.1%}"),
            ("ROE", info.get('returnOnEquity'), "{:.1%}"),
            ("ROA", info.get('returnOnAssets'), "{:.1%}"),
            ("P/E Ratio", info.get('trailingPE'), "{:.1f}"),
            ("P/B Ratio", info.get('priceToBook'), "{:.1f}"),
            ("EV/EBITDA", info.get('enterpriseToEbitda'), "{:.1f}"),
            ("Debt/Equity", info.get('debtToEquity'), "{:.1f}"),
        ]
        for i, (label, value, fmt) in enumerate(metrics):
            with cols[i % 4]:
                st.metric(label, fmt.format(value) if value else "N/A")

        # Analyst growth
        st.subheader("Growth Assumptions")
        analyst_growth_pct = None
        try:
            url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=earningsTrend"
            data = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).json()
            for t in data.get('quoteSummary', {}).get('result', [{}])[0].get('earningsTrend', {}).get('trend', []):
                if t.get('period') == '+5y':
                    analyst_growth_pct = t.get('growth', {}).get('raw')
                    if analyst_growth_pct:
                        analyst_growth_pct *= 100
                        break
        except:
            pass

        growth_rate = (default_growth / 100) if default_growth > 0 else (analyst_growth_pct / 100 if analyst_growth_pct else 0.08)

        if analyst_growth_pct:
            st.success(f"Analyst 5-year growth forecast: {analyst_growth_pct:.1f}%")
        else:
            st.info("No analyst growth forecast found → using 8% default")

        # Simple WACC
        beta = info.get('beta', 1.0)
        rf = 0.043  # ~10-year Treasury
        wacc = rf + beta * 0.05
        debt = info.get('totalDebt', 0)
        cap = info.get('marketCap', 0)
        if cap > 0:
            total = cap + debt
            wacc = (cap/total)*(rf + beta*0.05) + (debt/total)*0.05*(1-0.21)
        st.caption(f"Estimated WACC: {wacc:.1%}")

        # DCF Valuation
        st.subheader("DCF Valuation")
        cf = stock.cashflow
        if cf.empty:
            st.warning("Cash flow data unavailable → DCF skipped")
        else:
            if 'Free Cash Flow' in cf.index:
                fcf_series = cf.loc['Free Cash Flow'].dropna()
            else:
                ocf = cf.loc['Operating Cash Flow'].dropna() if 'Operating Cash Flow' in cf.index else pd.Series()
                capex = cf.loc['Capital Expenditures'].dropna() if 'Capital Expenditures' in cf.index else pd.Series()
                fcf_series = ocf + capex

            if fcf_series.empty or fcf_series.iloc[0] <= 0:
                st.warning("No positive Free Cash Flow → DCF not possible")
            else:
                avg_fcf = fcf_series.iloc[:3].mean()
                years = 5
                projected = [avg_fcf * (1 + growth_rate)**(y+1) for y in range(years)]
                terminal = projected[-1] * (1 + perp_growth) / (wacc - perp_growth)
                discounted = [f / (1 + wacc)**(y+1) for y, f in enumerate(projected)]
                tv_disc = terminal / (1 + wacc)**years

                enterprise_value = sum(discounted) + tv_disc
                net_debt = info.get('totalDebt', 0) - info.get('cash', 0)
                equity_value = max(enterprise_value - net_debt, 0)
                shares = info.get('sharesOutstanding', 1)
                intrinsic = equity_value / shares

                st.metric("Intrinsic Value (DCF)", f"${intrinsic:,.2f}")

                if price:
                    margin = (intrinsic - price) / price * 100
                    if margin > 20:
                        st.success(f"Potentially Undervalued (+{margin:.0f}% upside)")
                    elif margin < -20:
                        st.error(f"Potentially Overvalued ({margin:.0f}% downside)")
                    else:
                        st.info(f"Fairly valued ({margin:+.0f}% vs market)")

st.caption("Data from Yahoo Finance • Simple model for educational purposes")
