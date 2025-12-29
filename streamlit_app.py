import streamlit as st
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Stock DCF Analyzer", layout="centered")

st.title("📈 Stock Analyzer with DCF Valuation")
st.markdown("Analyze any stock: key metrics + analyst growth + simple DCF.")

ticker = st.text_input("Enter ticker (e.g. AAPL, TSLA)", "AAPL").upper().strip()

col1, col2 = st.columns(2)
with col1:
    override_growth = st.number_input("Override analyst growth (%) – 0 to auto-use", min_value=0.0, value=0.0)
with col2:
    perp_growth = st.number_input("Perpetual growth rate (%)", value=2.0) / 100

if st.button("🚀 Analyze Stock"):
    with st.spinner("Loading data from Yahoo Finance..."):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
        except Exception:
            st.error("Invalid ticker or connection issue.")
            st.stop()

        if 'longName' not in info:
            st.error("No data found for this ticker.")
            st.stop()

        # Header
        st.header(f"{info.get('longName', ticker)} ({ticker})")
        st.caption(f"Sector: {info.get('sector', 'N/A')}")

        price = info.get('currentPrice') or info.get('regularMarketPrice')
        if price:
            st.metric("Current Price", f"${price:.2f}")

        # Metrics
        st.subheader("Key Metrics")
        cols = st.columns(4)
        metrics = [
            ("Gross Margin", info.get('grossMargins'), "{:.1%}"),
            ("Net Margin", info.get('profitMargins'), "{:.1%}"),
            ("ROE", info.get('returnOnEquity'), "{:.1%}"),
            ("ROA", info.get('returnOnAssets'), "{:.1%}"),
            ("P/E", info.get('trailingPE'), "{:.1f}"),
            ("P/B", info.get('priceToBook'), "{:.1f}"),
            ("EV/EBITDA", info.get('enterpriseToEbitda'), "{:.1f}"),
            ("Debt/Equity", info.get('debtToEquity'), "{:.1f}"),
        ]
        for i, (label, val, fmt) in enumerate(metrics):
            with cols[i % 4]:
                st.metric(label, fmt.format(val) if val else "N/A")

        # Analyst growth
        st.subheader("Growth Rate Used")
        analyst_growth = None
        try:
            url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=earningsTrend"
            data = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10).json()
            for t in data.get('quoteSummary', {}).get('result', [{}])[0].get('earningsTrend', {}).get('trend', []):
                if t.get('period') == '+5y':
                    analyst_growth = t.get('growth', {}).get('raw')
                    if analyst_growth:
                        analyst_growth *= 100
                    break
        except:
            pass

        growth_rate = override_growth / 100 if override_growth > 0 else (analyst_growth / 100 if analyst_growth else 0.08)

        if analyst_growth:
            st.success(f"Analyst 5-yr growth: {analyst_growth:.1f}% (used)")
        else:
            st.info("No analyst forecast → using 8% default")

        # WACC
        beta = info.get('beta', 1.0)
        rf = 0.043
        wacc = rf + beta * 0.05
        debt = info.get('totalDebt', 0)
        cap = info.get('marketCap', 0)
        if cap:
            total = cap + debt
            wacc = (cap/total)*(rf + beta*0.05) + (debt/total)*0.05*(1-0.21)
        st.caption(f"Estimated WACC: {wacc:.1%}")

        # DCF
        st.subheader("DCF Intrinsic Value")
        cf = stock.cashflow
        if cf.empty:
            st.warning("Cash flow data missing")
        else:
            if 'Free Cash Flow' in cf.index:
                fcf = cf.loc['Free Cash Flow'].dropna()
            else:
                ocf = cf.loc['Operating Cash Flow'].dropna() if 'Operating Cash Flow' in cf.index else pd.Series()
                capex = cf.loc['Capital Expenditures'].dropna() if 'Capital Expenditures' in cf.index else pd.Series()
                fcf = ocf + capex

            if fcf.empty or fcf.iloc[0] <= 0:
                st.warning("No positive FCF → DCF unavailable")
            else:
                avg_fcf = fcf.iloc[:3].mean()
                years = 5
                proj = [avg_fcf * (1 + growth_rate)**(y+1) for y in range(years)]
                tv = proj[-1] * (1 + perp_growth) / (wacc - perp_growth)
                disc = [f / (1 + wacc)**(y+1) for y, f in enumerate(proj)]
                tv_disc = tv / (1 + wacc)**years

                ev = sum(disc) + tv_disc
                net_debt = info.get('totalDebt', 0) - info.get('cash', 0)
                equity = max(ev - net_debt, 0)
                shares = info.get('sharesOutstanding', 1)
                intrinsic = equity / shares

                st.metric("Intrinsic Value", f"${intrinsic:,.2f}")

                if price:
                    upside = (intrinsic - price) / price * 100
                    if upside > 20:
                        st.success(f"Potentially Undervalued (+{upside:.0f}% upside)")
                    elif upside < -20:
                        st.error(f"Potentially Overvalued ({upside:.0f}% downside)")
                    else:
                        st.info(f"Fairly priced ({upside:+.0f}% difference)")

st.caption("Data: Yahoo Finance • Simple educational model • Dec 2025")
