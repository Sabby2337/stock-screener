import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import re

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

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
    data = {"Ticker": ticker}
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if not hist.empty:
            hist.ta.rsi(length=14, append=True)
            hist.ta.macd(append=True)
            hist.ta.sma(length=50, append=True)
            hist.ta.sma(length=200, append=True)
            if "RSI_14" in hist.columns:
                data["RSI"] = float(hist["RSI_14"].iloc[-1])
            if "MACD_12_26_9" in hist.columns and "MACDs_12_26_9" in hist.columns:
                macd_val = hist["MACD_12_26_9"].iloc[-1]
                signal_val = hist["MACDs_12_26_9"].iloc[-1]
                if pd.notna(macd_val) and pd.notna(signal_val):
                    data["MACD_signal_diff"] = float(macd_val - signal_val)
            last_price = hist["Close"].iloc[-1]
            if "SMA_50" in hist.columns and pd.notna(hist["SMA_50"].iloc[-1]):
                data["Price_above_50MA"] = bool(last_price > hist["SMA_50"].iloc[-1])
            if "SMA_200" in hist.columns and pd.notna(hist["SMA_200"].iloc[-1]):
                data["Price_above_200MA"] = bool(last_price > hist["SMA_200"].iloc[-1])
        info = stock.info
        data["P/E"] = info.get("trailingPE")
        data["PEG"] = info.get("pegRatio")
        ev = info.get("enterpriseValue")
        ebitda = info.get("ebitda")
        data["EV/EBITDA"] = float(ev / ebitda) if ev and ebitda else None
        data["Beta"] = info.get("beta")
        if info.get("dividendYield") is not None:
            data["DividendYield"] = float(info["dividendYield"])
        elif info.get("trailingAnnualDividendYield") is not None:
            data["DividendYield"] = float(info["trailingAnnualDividendYield"])
        if info.get("debtToEquity") is not None:
            data["Debt/Equity"] = float(info["debtToEquity"])
        fin = stock.financials
        bal = stock.balance_sheet
        cf = stock.cashflow
        netincomes = None
        if not fin.empty:
            fin.columns = fin.columns.astype(str)
            if "Total Revenue" in fin.index:
                revenues = fin.loc["Total Revenue"]
                if revenues.shape[0] >= 2:
                    first_rev = revenues.iloc[-1]
                    last_rev = revenues.iloc[0]
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
        equity = None
        total_debt = None
        if not bal.empty:
            latest_col = bal.columns[0]
            if "Total Stockholder Equity" in bal.index:
                equity = bal.loc["Total Stockholder Equity", latest_col]
            if "Total Debt" in bal.index:
                total_debt = bal.loc["Total Debt", latest_col]
            else:
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
        if not cf.empty:
            latest_cf_col = cf.columns[0]
            cfo = cf.loc["Total Cash From Operating Activities", latest_cf_col] if "Total Cash From Operating Activities" in cf.index else None
            capex = cf.loc["Capital Expenditures", latest_cf_col] if "Capital Expenditures" in cf.index else None
            if cfo is not None and capex is not None:
                fcf = cfo + capex
                data["FreeCashFlow"] = float(fcf)
        if netincomes is not None and equity:
            last_net_income = netincomes.iloc[0] if netincomes.shape[0] > 0 else None
            if last_net_income and equity:
                data["ROE"] = float(last_net_income / equity * 100)
        if not fin.empty:
            cap_employed = (equity or 0) + (total_debt or 0)
            if "Operating Income" in fin.index and cap_employed > 0:
                op_income = fin.loc["Operating Income"].iloc[0]
                if pd.notna(op_income):
                    data["ROCE"] = float(op_income / cap_employed * 100)
            ebit = fin.loc["EBIT"].iloc[0] if "EBIT" in fin.index else fin.loc["Operating Income"].iloc[0] if "Operating Income" in fin.index else None
            interest_exp = fin.loc["Interest Expense"].iloc[0] if "Interest Expense" in fin.index else None
            if ebit and interest_exp and interest_exp != 0:
                data["InterestCoverage"] = float(abs(ebit / interest_exp))
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
    growth_score = financial_score = valuation_score = technical_score = ownership_score = 0
    if pd.notna(row.get("RevCAGR")):
        if row["RevCAGR"] > 15: growth_score += 1
        elif row["RevCAGR"] < 5: growth_score -= 1
    if pd.notna(row.get("EPSCAGR")):
        if row["EPSCAGR"] > 15: growth_score += 1
        elif row["EPSCAGR"] < 5: growth_score -= 1
    if pd.notna(row.get("ROE")):
        if row["ROE"] > 15: financial_score += 1
        elif row["ROE"] < 10: financial_score -= 1
    if pd.notna(row.get("ROCE")):
        if row["ROCE"] > 15: financial_score += 1
        elif row["ROCE"] < 10: financial_score -= 1
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
    if pd.notna(row.get("FreeCashFlow")):
        financial_score += 1 if row["FreeCashFlow"] > 0 else -1
    if pd.notna(row.get("DividendYield")):
        dy_pct = row["DividendYield"] * 100
        if dy_pct > 3: financial_score += 1
    if pd.notna(row.get("P/E")):
        pe = row["P/E"]
        if pe < 15: valuation_score += 1
        elif pe > 40: valuation_score -= 1
    if pd.notna(row.get("PEG")):
        peg = row["PEG"]
        if peg < 1: valuation_score += 1
        elif peg > 2: valuation_score -= 1
    if pd.notna(row.get("EV/EBITDA")):
        eve = row["EV/EBITDA"]
        if eve < 10: valuation_score += 1
        elif eve > 20: valuation_score -= 1
    if pd.notna(row.get("RSI")):
        rsi = row["RSI"]
        if rsi < 30: technical_score += 1
        elif rsi > 70: technical_score -= 1
    if pd.notna(row.get("MACD_signal_diff")):
        if row["MACD_signal_diff"] > 0: technical_score += 1
    if pd.notna(row.get("Price_above_200MA")):
        technical_score += 1 if row["Price_above_200MA"] else -1
    if pd.notna(row.get("Price_above_50MA")):
        if row["Price_above_50MA"]: technical_score += 1
    if pd.notna(row.get("Beta")):
        beta = row["Beta"]
        if beta < 0.8: technical_score += 1
        elif beta > 1.2: technical_score -= 1
    if pd.notna(row.get("PromoterHolding")):
        ph = row["PromoterHolding"]
        if ph > 50: ownership_score += 1
        elif ph < 30: ownership_score -= 1
    if pd.notna(row.get("PledgedPercent")):
        pledged = row["PledgedPercent"]
        ownership_score += 1 if pledged == 0 else -1
    total_score = growth_score + financial_score + valuation_score + technical_score + ownership_score
    return total_score, growth_score, financial_score, valuation_score, technical_score, ownership_score

# Streamlit UI
st.title("📈 Indian Stock Screener (NSE Stocks)")
st.markdown("This app analyzes Indian stocks on fundamental and technical parameters and ranks them based on a composite score.")

with st.spinner("Fetching and analyzing data for Nifty 50 stocks..."):
    stock_data_list = []
    for tkr in nifty50_tickers:
        result = fetch_stock_data(tkr)
        if result and not result.get("Error"):
            stock_data_list.append(result)
    df = pd.DataFrame(stock_data_list)

if df.empty:
    st.warning("No data available. Please try again later.")
else:
    scores = df.apply(lambda r: compute_score(r), axis=1)
    score_cols = ["Score", "GrowthScore", "FinancialScore", "ValuationScore", "TechnicalScore", "OwnershipScore"]
    scores_df = pd.DataFrame(scores.tolist(), columns=score_cols, index=df.index)
    df = pd.concat([df, scores_df], axis=1)
    df_ranked = df.sort_values("Score", ascending=False)

    st.header("Top Stocks (Nifty 50) by Overall Score")
    top_stocks = df_ranked.head(10).reset_index(drop=True)
    st.dataframe(top_stocks[["Ticker", "Score", "GrowthScore", "FinancialScore", "ValuationScore", "TechnicalScore", "OwnershipScore"]])
