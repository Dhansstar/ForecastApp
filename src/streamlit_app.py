import streamlit as st
from eda import run_eda
from prediction import run_prediction

# Konfigurasi Halaman
st.set_page_config(
    page_title="DemandSense AI - Inventory Forecasting",
    page_icon="📦",
    layout="wide"
)

# Sidebar Branding
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
st.sidebar.title("DemandSense AI")
st.sidebar.markdown("---")

# Menu Navigasi
menu = st.sidebar.radio(
    "Pilih Menu:",
    ["📊 Dashboard EDA", "🔮 30-Day Forecasting"],
    index=1
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Model Performance:**\n"
    "Total Volume Accuracy: **95.98%**"
)

# Routing Halaman
if menu == "📊 Dashboard EDA":
    run_eda()
else:
    run_prediction()