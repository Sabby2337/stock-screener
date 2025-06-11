import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import time
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
        try:
            hist = stock.history(period="1y", timeout=10)
        except Exception as e:
            data["Error"] = f"History error: {e}"
            return data

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
        dy = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
        data["DividendYield"] = float(dy) if dy else None
        data["Debt/Equity"] = float(info.get("debtToEquity")) if info.get("debtToEquity") else None

        # Screener parsing
        if requests and BeautifulSoup:
            try:
                code = ticker.split('.')[0]
                url = f"https://www.screener.in/company/{code}/consolidated/"
                r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                if r.status_code == 200:
                    soup = BeautifulSoup(r.text, "html.parser")
                    prom_row = soup.find("td", string="Promoters")
                    if prom_row:
                        cells = prom_row.find_parent("tr").find_all("td")
                        if len(cells) > 1 and "%" in cells[-1].text:
                            data["PromoterHolding"] = float(cells[-1].text.strip('%'))
                    pledge = re.search(r'Pledged.*?(\d+\.?\d*)%?', r.text)
                    if pledge:
                        data["PledgedPercent"] = float(pledge.group(1))
            except Exception as e:
                data["Error"] = f"Screener error: {e}"

    except Exception as e:
        data["Error"] = str(e)
    return data

def compute_score(row):
    g = f = v = t = o = 0
    if pd.notna(row.get("RSI")) and row["RSI"] < 30: t += 1
    if pd.notna(row.get("MACD_signal_diff")) and row["MACD_signal_diff"] > 0: t += 1
    if row.get("Price_above_50MA"): t += 1
    if row.get("Price_above_200MA"): t += 1
    if pd.notna(row.get("P/E")) and row["P/E"] < 15: v += 1
    if pd.notna(row.get("PEG")) and row["PEG"] < 1: v += 1
    if pd.notna(row.get("EV/EBITDA")) and row["EV/EBITDA"] < 10: v += 1
    if pd.notna(row.get("DividendYield")) and row["DividendYield"] > 0.02: f += 1
    if pd.notna(row.get("Debt/Equity")) and row["Debt/Equity"] < 0.5: f += 1
    if pd.notna(row.get("PromoterHolding")) and row["PromoterHolding"] > 50: o += 1
    if row.get("PledgedPercent") == 0: o += 1
    return g + f + v + t + o, g, f, v, t, o

# Streamlit UI
st.title("📊 Indian Stock Screener – Nifty50")
st.markdown("Ranks Nifty 50 stocks using fundamentals + technicals + ownership.")

with st.spinner("Fetching and analyzing stocks..."):
    all_data = []
    for tkr in nifty50_tickers:
        for _ in range(2):  # Retry twice
            time.sleep(0.5)
            res = fetch_stock_data(tkr)
            if not res.get("Error"):
                all_data.append(res)
                break

df = pd.DataFrame(all_data)

if df.empty:
    st.warning("No data fetched.")
else:
    score_df = df.apply(lambda r: compute_score(r), axis=1, result_type='expand')
    score_df.columns = ["Score", "Growth", "Financial", "Valuation", "Technical", "Ownership"]
    final_df = pd.concat([df, score_df], axis=1).sort_values("Score", ascending=False)
    st.subheader("Top Scoring Stocks")
    st.dataframe(final_df[["Ticker", "Score", "Growth", "Financial", "Valuation", "Technical", "Ownership"]].head(10))
