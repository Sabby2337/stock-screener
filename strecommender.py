import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import re

# Optional: import requests and BeautifulSoup for scraping Screener.in (promoter holding)
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

# Default list of Nifty 50 stock tickers (NSE)56
nifty50_tickers = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS",
    "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS", "ITC.NS",
    "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "LTIM.NS",
    "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SHRIRAMFIN.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS"
]

@st.cache_data
def fetch_stock_data(ticker):
    """Fetch financial data and technical indicators for a given stock ticker."""
    data = {"Ticker": ticker}
    try:
        stock = yf.Ticker(ticker)
        # Historical price data (1 year) for technical indicators
        hist = stock.history(period="1y")
        if not hist.empty:
            # Calculate technical indicators using pandas-ta
            hist.ta.rsi(length=14, append=True)
            hist.ta.macd(append=True)
            hist.ta.sma(length=50, append=True)
            hist.ta.sma(length=200, append=True)
            # RSI (last value)
            if "RSI_14" in hist.columns:
                data["RSI"] = float(hist["RSI_14"].iloc[-1])
            # MACD (difference between MACD and signal line last value)
            if "MACD_12_26_9" in hist.columns and "MACDs_12_26_9" in hist.columns:
                macd_val = hist["MACD_12_26_9"].iloc[-1]
                signal_val = hist["MACDs_12_26_9"].iloc[-1]
                if pd.notna(macd_val) and pd.notna(signal_val):
                    data["MACD_signal_diff"] = float(macd_val - signal_val)
            # Price vs moving averages
            last_price = hist["Close"].iloc[-1]
            if "SMA_50" in hist.columns and pd.notna(hist["SMA_50"].iloc[-1]):
                data["Price_above_50MA"] = bool(last_price > hist["SMA_50"].iloc[-1])
            if "SMA_200" in hist.columns and pd.notna(hist["SMA_200"].iloc[-1]):
                data["Price_above_200MA"] = bool(last_price > hist["SMA_200"].iloc[-1])
        # Get valuation and basic info metrics
        info = stock.info
        data["P/E"] = info.get("trailingPE")
        data["PEG"] = info.get("pegRatio")
        ev = info.get("enterpriseValue")
        ebitda = info.get("ebitda")
        data["EV/EBITDA"] = float(ev / ebitda) if ev and ebitda else None
        data["Beta"] = info.get("beta")
        # Dividend yield (as percentage)
        if info.get("dividendYield") is not None:
            data["DividendYield"] = float(info["dividendYield"])
        elif info.get("trailingAnnualDividendYield") is not None:
            data["DividendYield"] = float(info["trailingAnnualDividendYield"])
        # Debt to Equity from info if available
        if info.get("debtToEquity") is not None:
            data["Debt/Equity"] = float(info["debtToEquity"])
        # Fetch financial statements
        fin = stock.financials  # annual income statement
        bal = stock.balance_sheet  # annual balance sheet
        cf = stock.cashflow       # annual cash flow
        # Compute 5-year CAGR for Revenue and EPS (if 4+ years of data)
        netincomes = None
        if not fin.empty:
            fin.columns = fin.columns.astype(str)  # ensure columns as strings (years)
            if "Total Revenue" in fin.index:
                revenues = fin.loc["Total Revenue"]
                if revenues.shape[0] >= 2:
                    # Use first and last available values for CAGR
                    first_rev = revenues.iloc[-1]  # earliest year
                    last_rev = revenues.iloc[0]    # latest year
                    years = revenues.shape[0] - 1
                    if first_rev and last_rev and years > 0:
                        data["RevCAGR"] = float(((last_rev/first_rev) ** (1/years) - 1) * 100)
            if "Net Income" in fin.index:
                netincomes = fin.loc["Net Income"]
                if netincomes.shape[0] >= 2:
                    first_ni = netincomes.iloc[-1]
                    last_ni = netincomes.iloc[0]
                    years = netincomes.shape[0] - 1
                    if first_ni and last_ni and years > 0:
                        data["EPSCAGR"] = float(((last_ni/first_ni) ** (1/years) - 1) * 100)
        # Leverage ratios from balance sheet
        equity = None
        total_debt = None
        if not bal.empty:
            # Use the latest year column in balance sheet (column 0)
            latest_col = bal.columns[0]
            if "Total Stockholder Equity" in bal.index:
                equity = bal.loc["Total Stockholder Equity", latest_col]
            if "Total Debt" in bal.index:
                total_debt = bal.loc["Total Debt", latest_col]
            else:
                # Sum long-term and short-term debt if available
                long_debt = bal.loc["Long Term Debt", latest_col] if "Long Term Debt" in bal.index else 0
                short_debt = bal.loc["Short Long Term Debt", latest_col] if "Short Long Term Debt" in bal.index else 0
                total_debt = long_debt + short_debt if (long_debt or short_debt) else None
            if equity and total_debt is not None and equity != 0:
                data["Debt/Equity"] = float(total_debt / equity)
            if "Total Current Assets" in bal.index and "Total Current Liabilities" in bal.index:
                ca = bal.loc["Total Current Assets", latest_col]
                cl = bal.loc["Total Current Liabilities", latest_col]
                if cl and cl != 0:
                    data["CurrentRatio"] = float(ca / cl)
        # Free Cash Flow from cash flow statement (Operating Cash Flow + Capital Expenditures)
        if not cf.empty:
            latest_cf_col = cf.columns[0]
            cfo = cf.loc["Total Cash From Operating Activities", latest_cf_col] if "Total Cash From Operating Activities" in cf.index else None
            capex = cf.loc["Capital Expenditures", latest_cf_col] if "Capital Expenditures" in cf.index else None
            if cfo is not None and capex is not None:
                fcf = cfo + capex  # capex is negative in cash flow
                data["FreeCashFlow"] = float(fcf)
        # Profitability: ROE and ROCE
        if netincomes is not None and equity:
            last_net_income = netincomes.iloc[0] if netincomes.shape[0] > 0 else None
            if last_net_income and equity:
                data["ROE"] = float(last_net_income / equity * 100)
        if not fin.empty:
            # ROCE = Operating Income / (Equity + Debt)
            cap_employed = (equity or 0) + (total_debt or 0)
            if "Operating Income" in fin.index and cap_employed > 0:
                op_income = fin.loc["Operating Income"].iloc[0]
                if pd.notna(op_income):
                    data["ROCE"] = float(op_income / cap_employed * 100)
            # Interest Coverage = EBIT / Interest Expense
            ebit = None
            if "EBIT" in fin.index:
                ebit = fin.loc["EBIT"].iloc[0]
            elif "Operating Income" in fin.index:
                ebit = fin.loc["Operating Income"].iloc[0]
            if "Interest Expense" in fin.index:
                interest_exp = fin.loc["Interest Expense"].iloc[0]
            else:
                interest_exp = None
            if ebit and interest_exp and interest_exp != 0:
                data["InterestCoverage"] = float(abs(ebit / interest_exp))
        # Ownership: Promoter holding and pledged shares (from Screener.in if enabled)
        if requests and BeautifulSoup:
            try:
                company_code = ticker.split('.')[0]
                url = f"https://www.screener.in/company/{company_code}/consolidated/"
                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    prom_row = soup.find("td", string="Promoters")
                    if prom_row:
                        cells = prom_row.find_parent("tr").find_all("td")
                        if len(cells) > 1:
                            prom_str = cells[-1].get_text().strip()
                            if prom_str.endswith("%"):
                                prom_str = prom_str[:-1]
                            if prom_str:
                                data["PromoterHolding"] = float(prom_str)
                    # Check for pledged percentage in text
                    text = resp.text
                    pledged_match = re.search(r'Pledged.*?(\d+\.?\d*)%?', text)
                    if pledged_match:
                        pledged_val = pledged_match.group(1)
                        try:
                            data["PledgedPercent"] = float(pledged_val)
                        except:
                            pass
            except Exception:
                pass
    except Exception as e:
        data["Error"] = str(e)
    return data

def compute_score(row):
    """Compute total and category scores for a stock based on various metrics."""
    growth_score = financial_score = valuation_score = technical_score = ownership_score = 0
    # Growth (Revenue & EPS CAGR)
    if pd.notna(row.get("RevCAGR")):
        if row["RevCAGR"] > 15: growth_score += 1
        elif row["RevCAGR"] < 5: growth_score -= 1
    if pd.notna(row.get("EPSCAGR")):
        if row["EPSCAGR"] > 15: growth_score += 1
        elif row["EPSCAGR"] < 5: growth_score -= 1
    # Profitability (ROE, ROCE)
    if pd.notna(row.get("ROE")):
        if row["ROE"] > 15: financial_score += 1
        elif row["ROE"] < 10: financial_score -= 1
    if pd.notna(row.get("ROCE")):
        if row["ROCE"] > 15: financial_score += 1
        elif row["ROCE"] < 10: financial_score -= 1
    # Leverage & Liquidity (Debt/Equity, Interest Coverage, Current Ratio)
    if pd.notna(row.get("Debt/Equity")):
        de = row["Debt/Equity"]
        if de < 0.5: financial_score += 1
        elif de > 1: financial_score -= 1
    if pd.notna(row.get("InterestCoverage")):
        ic = row["InterestCoverage"]
        if ic > 5: financial_score += 1
        elif ic < 1: financial_score -= 1
    if pd.notna(row.get("CurrentRatio")):
        cr = row["CurrentRatio"]
        if cr > 1.5: financial_score += 1
        elif cr < 1: financial_score -= 1
    # Cash flow & Dividend
    if pd.notna(row.get("FreeCashFlow")):
        if row["FreeCashFlow"] > 0: financial_score += 1
        else: financial_score -= 1
    if pd.notna(row.get("DividendYield")) and row["DividendYield"] is not None:
        # DividendYield likely in 0-1 range (e.g., 0.02 for 2%)
        dy_pct = row["DividendYield"] * 100
        if dy_pct > 3: financial_score += 1
    # Valuation (P/E, PEG, EV/EBITDA)
    if pd.notna(row.get("P/E")):
        pe = row["P/E"]
        if pe and pe < 15: valuation_score += 1
        elif pe and pe > 40: valuation_score -= 1
    if pd.notna(row.get("PEG")):
        peg = row["PEG"]
        if peg and peg < 1: valuation_score += 1
        elif peg and peg > 2: valuation_score -= 1
    if pd.notna(row.get("EV/EBITDA")):
        eve = row["EV/EBITDA"]
        if eve and eve < 10: valuation_score += 1
        elif eve and eve > 20: valuation_score -= 1
    # Technical (RSI, MACD, Moving Averages, Beta)
    if pd.notna(row.get("RSI")):
        rsi = row["RSI"]
        if rsi < 30: technical_score += 1        # oversold (potential uptrend)
        elif rsi > 70: technical_score -= 1      # overbought
    if pd.notna(row.get("MACD_signal_diff")):
        if row["MACD_signal_diff"] and row["MACD_signal_diff"] > 0: 
            technical_score += 1  # bullish MACD crossover
    if pd.notna(row.get("Price_above_200MA")):
        if row["Price_above_200MA"]: technical_score += 1
        else: technical_score -= 1
    if pd.notna(row.get("Price_above_50MA")):
        if row["Price_above_50MA"]: technical_score += 1
    if pd.notna(row.get("Beta")):
        beta = row["Beta"]
        if beta and beta < 0.8: technical_score += 1   # low volatility
        elif beta and beta > 1.2: technical_score -= 1  # high volatility
    # Ownership (Promoter holding, Pledged shares)
    if pd.notna(row.get("PromoterHolding")):
        ph = row["PromoterHolding"]
        if ph > 50: ownership_score += 1
        elif ph < 30: ownership_score -= 1
    if pd.notna(row.get("PledgedPercent")):
        pledged = row["PledgedPercent"]
        if pledged > 0: ownership_score -= 1
        else: ownership_score += 1
    total_score = growth_score + financial_score + valuation_score + technical_score + ownership_score
    return total_score, growth_score, financial_score, valuation_score, technical_score, ownership_score

# Fetch data for default tickers and compute scores
stock_data_list = [fetch_stock_data(tkr) for tkr in nifty50_tickers]
df = pd.DataFrame(stock_data_list)
# Drop any tickers that failed to fetch (Error)
if "Error" in df.columns:
    df = df[df["Error"].isna()]
# Compute scores and add as columns
scores = df.apply(lambda r: compute_score(r), axis=1)
score_cols = ["Score", "GrowthScore", "FinancialScore", "ValuationScore", "TechnicalScore", "OwnershipScore"]
scores_df = pd.DataFrame(scores.tolist(), columns=score_cols, index=df.index)
df = pd.concat([df, scores_df], axis=1)
# Sort stocks by total Score
df_ranked = df.sort_values("Score", ascending=False)

# Streamlit UI
st.title("📈 Indian Stock Screener (NSE Stocks)")
st.markdown("This app analyzes Indian stocks on fundamental and technical parameters and ranks them based on a composite score.")

st.header("Top Stocks (Nifty 50) by Overall Score")
top_stocks = df_ranked.head(10).reset_index(drop=True)
st.dataframe(top_stocks[["Ticker", "Score", "GrowthScore", "FinancialScore", "ValuationScore", "TechnicalScore", "OwnershipScore"]])

# File uploader for user portfolio
uploaded_file = st.file_uploader("Upload your portfolio CSV (with a column of stock tickers)", type=["csv"])
if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    # Determine ticker list from uploaded file
    if "Ticker" in user_df.columns:
        portfolio_tickers = user_df["Ticker"].astype(str).tolist()
    else:
        # assume first column contains tickers
        portfolio_tickers = user_df.iloc[:, 0].astype(str).tolist()
    # Clean ticker symbols and ensure NSE format
    portfolio_tickers = [t.strip().upper() for t in portfolio_tickers if str(t) != 'nan']
    portfolio_tickers = [t + ".NS" if not t.endswith(".NS") else t for t in portfolio_tickers]
    portfolio_tickers = list(dict.fromkeys(portfolio_tickers))  # unique, preserve order
    if portfolio_tickers:
        st.header("Portfolio Analysis")
    for tkr in portfolio_tickers:
        data = fetch_stock_data(tkr)
        ticker_name = tkr.rstrip(".NS")
        if data.get("Error"):
            st.write(f"**{ticker_name}:** ❌ Data not available.")
            continue
        # Compute strengths and weaknesses for this stock
        pros = []
        cons = []
        if data.get("RevCAGR") is not None:
            if data["RevCAGR"] > 15: pros.append(f"Strong revenue growth (~{data['RevCAGR']:.1f}% 5Y CAGR)")
            elif data["RevCAGR"] < 5: cons.append(f"Poor revenue growth (~{data['RevCAGR']:.1f}% 5Y CAGR)")
        if data.get("EPSCAGR") is not None:
            if data["EPSCAGR"] > 15: pros.append(f"Strong earnings growth (~{data['EPSCAGR']:.1f}% 5Y CAGR)")
            elif data["EPSCAGR"] < 5: cons.append(f"Weak earnings growth (~{data['EPSCAGR']:.1f}% 5Y CAGR)")
        if data.get("ROE") is not None:
            if data["ROE"] > 15: pros.append(f"High Return on Equity (ROE ~{data['ROE']:.1f}%)")
            elif data["ROE"] < 10: cons.append(f"Low Return on Equity (ROE ~{data['ROE']:.1f}%)")
        if data.get("ROCE") is not None:
            if data["ROCE"] > 15: pros.append(f"High Return on Capital Employed (ROCE ~{data['ROCE']:.1f}%)")
            elif data["ROCE"] < 10: cons.append(f"Low ROCE (~{data['ROCE']:.1f}%)")
        if data.get("Debt/Equity") is not None:
            de = data["Debt/Equity"]
            if de < 0.5: pros.append(f"Low debt (Debt/Equity = {de:.2f})")
            elif de > 1: cons.append(f"High debt (Debt/Equity = {de:.2f})")
        if data.get("InterestCoverage") is not None:
            if data["InterestCoverage"] < 2: cons.append("Low interest coverage (debt service ability is weak)")
        if data.get("CurrentRatio") is not None:
            cr = data["CurrentRatio"]
            if cr < 1: cons.append(f"Current ratio {cr:.1f} (potential liquidity issues)")
            elif cr > 2: pros.append(f"Healthy current ratio of {cr:.1f}")
        if data.get("FreeCashFlow") is not None:
            if data["FreeCashFlow"] < 0: cons.append("Negative free cash flow (recent year)")
            else: pros.append("Positive free cash flow generation")
        if data.get("DividendYield") is not None and data["DividendYield"] > 0:
            dy_percent = data["DividendYield"] * 100
            if dy_percent >= 3: pros.append(f"Good dividend yield (~{dy_percent:.1f}%)")
            elif dy_percent < 1: cons.append("Low dividend yield")
        if data.get("P/E") is not None:
            pe = data["P/E"]
            if pe and pe > 40: cons.append(f"Overvalued on P/E (≈{pe:.1f})")
            elif pe and pe < 10: pros.append(f"Attractive P/E valuation (≈{pe:.1f})")
        if data.get("PEG") is not None:
            peg = data["PEG"]
            if peg and peg > 2: cons.append(f"High PEG ratio (≈{peg:.1f}) suggests rich valuation")
            elif peg and peg < 1: pros.append("PEG below 1 (undervalued relative to growth)")
        if data.get("RSI") is not None:
            rsi = data["RSI"]
            if rsi > 70: cons.append(f"Stock may be overbought (RSI {rsi:.0f})")
            elif rsi < 30: pros.append(f"Stock appears oversold (RSI {rsi:.0f})")
        if data.get("Price_above_200MA") is not None:
            if not data["Price_above_200MA"]: cons.append("Below 200-day MA (weak long-term trend)")
            else: pros.append("Above 200-day MA (strong trend)")
        if data.get("PromoterHolding") is not None:
            ph = data["PromoterHolding"]
            if ph > 50: pros.append(f"High promoter holding ({ph:.1f}%)")
            elif ph < 30: cons.append(f"Low promoter holding ({ph:.1f}%)")
        if data.get("PledgedPercent") is not None:
            pledged = data["PledgedPercent"]
            if pledged > 0: cons.append(f"Promoters pledged {pledged:.1f}% of shares (red flag)")
            else: pros.append("No promoter share pledging")
        # Display the analysis for this stock
        st.subheader(f"{ticker_name} — Score: {int(data.get('Score') or 0)}")
        if pros:
            st.write("**Strengths:**")
            for p in pros:
                st.markdown(f"- {p}")
        if cons:
            st.write("**Weaknesses:**")
            for c in cons:
                st.markdown(f"- {c}")
