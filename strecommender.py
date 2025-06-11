import streamlit as st
import pandas as pd
from nsetools import Nse

nse = Nse()

nifty_tickers = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "KOTAKBANK", "ITC", "BAJFINANCE", "LT"
]

st.set_page_config(page_title="Nifty50 Screener – NSE", layout="wide")
st.title("📊 Indian Stock Screener (NSE via `nsetools`)")

data = []
for ticker in nifty_tickers:
    try:
        info = nse.get_quote(ticker)
        data.append({
            "Symbol": ticker,
            "Company Name": info.get("companyName"),
            "Price": info.get("lastPrice"),
            "Day Change (%)": info.get("pChange"),
            "EPS": info.get("eps"),
            "Market Cap": info.get("marketCap"),
            "Industry": info.get("industry")
        })
    except Exception as e:
        data.append({"Symbol": ticker, "Error": str(e)})

df = pd.DataFrame(data)

if df.empty:
    st.error("❌ No data fetched. NSE server may be slow or blocking.")
else:
    st.success("✅ Fetched stock data successfully")
    st.dataframe(df)
