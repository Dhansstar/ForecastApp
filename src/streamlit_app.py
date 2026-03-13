import streamlit as st
import base64
import os
from streamlit_option_menu import option_menu
import eda
import prediction

st.set_page_config(page_title="DemandSense", layout="wide")

# --- UTILS: LOAD CSS & GIF ---
def set_bg_local(gif_path):
    if os.path.exists(gif_path):
        with open(gif_path, "rb") as f:
            base64_gif = base64.b64encode(f.read()).decode()
        st.markdown(f'<style>:root {{--bg-image: url("data:image/gif;base64,{base64_gif}");}}</style>', unsafe_allow_html=True)

def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Execute Styling
local_css("style.css")
set_bg_local("6.gif") # Pastikan file ini ada di folder

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown('<div id="text-split"><h2>🚀 DEMANDSENSE</h2></div>', unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None,
        options=["EDA", "Prediction"],
        icons=["bar-chart-fill", "cpu-fill"],
        default_index=0,
        styles={
            "container": {"background-color": "transparent"},
            "nav-link": {"font-weight": "bold", "color": "white"},
            "nav-link-selected": {"background-color": "rgba(255,255,255,0.1)"}
        }
    )

# --- ROUTING ---
if selected == "EDA":
    eda.run()
elif selected == "Prediction":
    prediction.run()