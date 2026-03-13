import streamlit as st
from streamlit_option_menu import option_menu
from src import eda, prediction
import os

# 1. KONFIGURASI HALAMAN (Wajib paling atas, jangan ada di eda.py/prediction.py)
st.set_page_config(
    page_title="DemandSense AI", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# 2. INJECT CSS GLOBAL
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"File {file_name} tidak ditemukan.")

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

# 4. ROUTING LOGIC DENGAN ISOLASI CONTAINER
# Kita pakai st.empty() untuk mastiin setiap pindah menu, layar di-clear dulu
main_view = st.container()

with main_view:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    if selected == "EDA Analysis":
        eda.run()
    elif selected == "Demand Prediction":
        prediction.run()
        
    st.markdown('</div>', unsafe_allow_html=True)

# 5. ULTRA-PERSISTENT ANIMATION SCRIPT (Anime.js + MutationObserver)
# Script ini bakal tetep jalan & nge-scan elemen baru pas lo ganti menu
st.markdown("""
<script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.1/anime.min.js"></script>
<script>
    function triggerAnimations() {
        // Animasi Header Meluncur
        anime({
            targets: '.animate-header',
            translateY: [-30, 0],
            opacity: [0, 1],
            delay: anime.stagger(150),
            easing: 'easeOutExpo',
            duration: 1000
        });

        // Animasi Square Neon
        anime({
            targets: '.square',
            rotate: '1turn',
            scale: [0.8, 1.2],
            backgroundColor: ['#3b82f6', '#ec4899', '#10b981'],
            duration: 3000,
            loop: true,
            direction: 'alternate',
            easing: 'easeInOutSine'
        });
    }

    // Monitor perubahan DOM supaya pas Streamlit ganti menu, animasi tetep kena
    const observer = new MutationObserver((mutations) => {
        let shouldAnimate = false;
        mutations.forEach(m => {
            if (m.addedNodes.length > 0) shouldAnimate = true;
        });
        if (shouldAnimate) triggerAnimations();
    });

    observer.observe(document.body, { childList: true, subtree: true });
    
    // Jalankan pertama kali
    document.addEventListener('DOMContentLoaded', triggerAnimations);
</script>
""", unsafe_allow_html=True)