import streamlit as st
import pandas as pd

st.set_page_config(page_title="ALPHAFORGE UI Diagnostic")
st.title("Streamlit Rendering Diagnostic")
st.success("Basic rendering works")

st.subheader("Positions")
st.dataframe(pd.DataFrame(columns=["symbol", "qty", "market_value"])
             if False else pd.DataFrame([{"symbol": "TEST", "qty": 1}]),
             width="stretch")

st.subheader("Orders")
st.info("No orders yet.")
