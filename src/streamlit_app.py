import streamlit as st
from streamlit_option_menu import option_menu
import sys
import os

# --- 1. FIX PATH & IMPORT ---
# Biar Python bisa baca folder 'src' meskipun running dari berbagai level folder
current_dir = os.path.dirname(os.path.abspath(__file__)) # folder 'src'
root_dir = os.path.dirname(current_dir) # folder 'forecastapp' (root)

if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from src import eda, prediction
except ImportError:
    import eda, prediction

# --- 2. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="DemandSense AI", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 3. INJECT CSS GLOBAL ---
def local_css(file_name):
    # Cari style.css di folder yang sama dengan script ini
    file_path = os.path.join(current_dir, file_name)
    if os.path.exists(file_path):
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown('<div id="text-split"><h1 class="animate-header" style="text-align:center; color:#f8fafc; margin-bottom:20px;">🚀 DemandSense</h1></div>', unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["EDA Analysis", "Demand Prediction"],
        icons=["bar-chart-fill", "cpu-fill"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {
                "padding": "0!important", 
                "background-color": "rgba(255, 255, 255, 0.03)!important", 
                "border-radius": "12px",
                "border": "1px solid rgba(255, 255, 255, 0.1)"
            },
            "icon": {"color": "#3b82f6", "font-size": "18px"}, 
            "nav-link": {
                "font-size": "14px", 
                "text-align": "left", 
                "margin": "8px", 
                "color": "#94a3b8",
                "border-radius": "8px",
                "--hover-color": "rgba(59, 130, 246, 0.1)"
            },
            "nav-link-selected": {
                "background": "linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%)!important",
                "color": "white!important",
                "font-weight": "600",
                "box-shadow": "0px 4px 15px rgba(59, 130, 246, 0.3)"
            }
        }
    )

# --- 5. ROUTING & CONTENT ---
st.markdown('<div class="main-container">', unsafe_allow_html=True)

if selected == "EDA Analysis":
    eda.run()
elif selected == "Demand Prediction":
    prediction.run()

st.markdown('</div>', unsafe_allow_html=True)

# --- 6. ANIMATION ENGINE (ANIME.JS) ---
st.markdown("""
<script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.1/anime.min.js"></script>
<script>
    const observer = new MutationObserver((mutations) => {
        // Animasi Header
        anime({
            targets: '.animate-header',
            translateY: [-20, 0],
            opacity: [0, 1],
            delay: anime.stagger(100),
            easing: 'easeOutExpo',
            duration: 800
        });
        
        // Animasi Card & Element
        anime({
            targets: '.glass-card, .stButton',
            scale: [0.98, 1],
            opacity: [0, 1],
            duration: 1000,
            easing: 'easeOutElastic(1, .8)'
        });
    });

    observer.observe(document.body, { childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)