import streamlit as st
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Advanced Stock DCF Analyzer", layout="centered")

st.title("📈 Advanced Stock Analyzer with DCF & Margin of Safety")
st.markdown("Professional-grade valuation: conservative growth, realistic WACC, and Benjamin Graham-style margin of safety.")

ticker = st.text_input("Enter ticker (e.g. AAPL, BABA, UNH)", "AAPL").upper().strip()

col1, col2, col3 = st.columns(3)
with col1:
    override_growth = st.number_input("Override analyst 5-yr growth (%) – 0 to auto-use", min_value=0.0, value=0.0, step=0.5)
with col2:
    perp_growth = st.slider("Perpetual growth rate (%)", min_value=0.5, max_value=4.0, value=2.0, step=0.5) / 100
with col3:
    margin_of_safety = st.slider("Desired Margin of Safety (%)", min_value=10, max_value=70, value=30, step=5,
                                 help="Buy price = Intrinsic Value × (1 - MoS)")

if st.button("🚀 Analyze Stock"):
    with st.spinner("Fetching data from Yahoo Finance..."):
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
        st.caption(f"Sector: {info.get('sector', 'N/A')} • Data as of {datetime.now().strftime('%B %d, %Y')}")

        price = info.get('currentPrice') or info.get('regularMarketPrice')
        if price:
            st.metric("Current Price", f"${price:.2f}")

        # Key Metrics
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
        for i, (label, val, fmt) in enumerate(metrics):
            with cols[i % 4]:
                st.metric(label, fmt.format(val) if val else "N/A")

        # Analyst Growth (more robust)
        st.subheader("Growth Assumptions")
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

        growth_rate = override_growth / 100 if override_growth > 0 else (analyst_growth / 100 if analyst_growth else 0.06)  # Conservative default 6%

        if analyst_growth:
            st.success(f"Analyst 5-year EPS growth forecast: {analyst_growth:.1f}%")
        else:
            st.info("No analyst forecast available → using conservative 6% default")

        # Improved WACC
        beta = info.get('beta', 1.0)
        rf = 0.043  # 10-year Treasury ~4.3% (Dec 2025)
        erp = 0.055  # Slightly higher equity risk premium for realism
        cost_equity = rf + beta * erp

        debt = info.get('totalDebt', 0)
        market_cap = info.get('marketCap', 0)
        if market_cap and market_cap > 0:
            enterprise_value = market_cap + debt
            weight_equity = market_cap / enterprise_value
            weight_debt = debt / enterprise_value
            cost_debt = 0.05  # Conservative pre-tax cost of debt
            tax_rate = 0.21
            wacc = weight_equity * cost_equity + weight_debt * cost_debt * (1 - tax_rate)
        else:
            wacc = 0.09  # Conservative fallback

        st.caption(f"Estimated WACC: {wacc:.1%} (Rf: 4.3%, ERP: 5.5%)")

        # Improved DCF
        st.subheader("DCF Valuation & Margin of Safety")
        cf = stock.cashflow
        if cf.empty:
            st.warning("Cash flow data unavailable")
        else:
            # Prefer direct FCF, fallback to OCF - CapEx
            if 'Free Cash Flow' in cf.index:
                fcf_series = cf.loc['Free Cash Flow'].dropna()
            else:
                ocf = cf.loc['Operating Cash Flow'].dropna() if 'Operating Cash Flow' in cf.index else pd.Series()
                capex = cf.loc['Capital Expenditures'].dropna() if 'Capital Expenditures' in cf.index else pd.Series()
                fcf_series = ocf + capex

            if fcf_series.empty or fcf_series.iloc[0] <= 0:
                st.warning("No positive Free Cash Flow → DCF not reliable")
            else:
                # Use 3-year average for stability
                avg_fcf = fcf_series.iloc[:3].mean()

                years = 10  # 10-year explicit forecast for more accuracy
                projected_fcf = [avg_fcf * (1 + growth_rate) ** (i + 1) for i in range(years)]

                # Terminal value (Gordon Growth)
                terminal_value = projected_fcf[-1] * (1 + perp_growth) / (wacc - perp_growth)

                # Discount cash flows
                discounted_fcf = [fcf / (1 + wacc) ** (i + 1) for i, fcf in enumerate(projected_fcf)]
                discounted_tv = terminal_value / (1 + wacc) ** years

                enterprise_value = sum(discounted_fcf) + discounted_tv
                net_debt = info.get('totalDebt', 0) - info.get('cash', 0)
                equity_value = max(enterprise_value - net_debt, 0)
                shares = info.get('sharesOutstanding', 1)
                intrinsic_value = equity_value / shares

                # Margin of Safety
                target_buy_price = intrinsic_value * (1 - margin_of_safety / 100)

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Intrinsic Value (DCF)", f"${intrinsic_value:,.2f}")
                col_b.metric(f"Target Buy Price ({margin_of_safety}% MoS)", f"${target_buy_price:,.2f}")
                col_c.metric("Current Price", f"${price:.2f}" if price else "N/A")

                if price:
                    upside_to_intrinsic = (intrinsic_value - price) / price * 100
                    if price <= target_buy_price:
                        st.success(f"🚀 STRONG BUY SIGNAL: Price below target with {margin_of_safety}% margin of safety "
                                   f"(Potential upside: {upside_to_intrinsic:.0f}%)")
                    elif price <= intrinsic_value:
                        st.info(f"Moderate opportunity: {upside_to_intrinsic:.0f}% upside to intrinsic value, "
                                f"but below your {margin_of_safety}% safety threshold")
                    else:
                        st.error(f"Overvalued: Trading {abs(upside_to_intrinsic):.0f}% above intrinsic value")

        # Historical Price Chart
        st.subheader("1-Year Price History")
        try:
            hist = stock.history(period="1y")
            if not hist.empty:
                st.line_chart(hist['Close'], use_container_width=True)
            else:
                st.info("No price history available.")
        except:
            st.info("Chart unavailable.")

st.caption("Data: Yahoo Finance • Conservative 10-year DCF model • Educational purposes only • December 2025")
