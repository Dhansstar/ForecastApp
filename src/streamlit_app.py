import streamlit as st
from streamlit_option_menu import option_menu
import sys
import os

# --- FIX PATH (Biar Python gak dongo nyari folder src) ---
# Kita tambahin directory utama ke sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # Naik satu level ke root
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Sekarang baru import modulnya
try:
    from src import eda, prediction
except ImportError:
    # Backup plan kalau struktur folder lo beda
    import eda, prediction 

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="DemandSense AI", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# 2. INJECT CSS GLOBAL
def local_css(file_name):
    # Nyari file style.css di root atau di folder yang sama
    paths = [file_name, os.path.join(current_dir, file_name)]
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            return

local_css("style.css")

# 3. SIDEBAR NAVIGATION
with st.sidebar:
    st.markdown('<div id="text-split"><h1 class="animate-header" style="text-align:center;">🚀 DemandSense</h1></div>', unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["EDA Analysis", "Demand Prediction"],
        icons=["bar-chart-line-fill", "cpu-fill"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "rgba(255, 255, 255, 0.05)", "border-radius": "10px"},
            "icon": {"color": "#3b82f6", "font-size": "18px"}, 
            "nav-link": {
                "font-size": "14px", 
                "text-align": "left", 
                "margin":"5px", 
                "color": "#f8fafc",
                "--hover-color": "rgba(59, 130, 246, 0.2)"
            },
            "nav-link-selected": {"background-color": "#3b82f6", "font-weight": "600"},
        }
    )

# 4. ROUTING LOGIC
main_view = st.container()

with main_view:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    if selected == "EDA Analysis":
        eda.run()
    elif selected == "Demand Prediction":
        prediction.run()
        
    st.markdown('</div>', unsafe_allow_html=True)

# 5. GLOBAL ANIMATION SCRIPT (Anime.js)
st.markdown("""
<script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.1/anime.min.js"></script>
<script>
    function triggerAnimations() {
        anime({
            targets: '.animate-header',
            translateY: [-30, 0],
            opacity: [0, 1],
            delay: anime.stagger(150),
            easing: 'easeOutExpo',
            duration: 1000
        });

        anime({
            targets: '.square',
            rotate: '1turn',
            scale: [0.8, 1.2],
            backgroundColor: ['#3b82f6', '#ec4899'],
            duration: 3000,
            loop: true,
            direction: 'alternate',
            easing: 'easeInOutSine'
        });
    }

    const observer = new MutationObserver((mutations) => {
        if (mutations.some(m => m.addedNodes.length > 0)) triggerAnimations();
    });

    observer.observe(document.body, { childList: true, subtree: true });
    window.onload = triggerAnimations;
</script>
""", unsafe_allow_html=True)