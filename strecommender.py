import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd

st.set_page_config(page_title="Ticker Test – yfinance", layout="centered")
st.title("🧪 yFinance Ticker Test (TCS.NS)")

ticker = "TCS.NS"
st.markdown(f"### Testing ticker: **{ticker}**")

try:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")

    if hist is None or hist.empty:
        st.error("❌ No historical data fetched. yFinance might be blocked or ticker is invalid.")
    else:
        st.success("✅ Historical data fetched successfully!")

        # Apply indicators
        hist.ta.rsi(length=14, append=True)
        hist.ta.macd(append=True)
        hist.ta.sma(length=50, append=True)
        hist.ta.sma(length=200, append=True)

        # Display last 5 rows
        st.subheader("📈 Last 5 Days of Data")
        st.dataframe(hist.tail())

        # Plot closing price
        st.subheader("🔹 Close Price Chart")
        st.line_chart(hist["Close"])

        # Plot RSI if available
        if "RSI_14" in hist.columns:
            st.subheader("🔸 RSI (14)")
            st.line_chart(hist["RSI_14"])
        else:
            st.warning("RSI_14 not calculated.")
except Exception as e:
    st.error(f"⚠️ Exception occurred:\n\n`{e}`")
