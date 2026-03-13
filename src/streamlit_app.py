import streamlit as st
import base64
import os
from streamlit_option_menu import option_menu
import eda
import prediction

# 1. SET CONFIG
st.set_page_config(page_title="DemandSense", layout="wide", initial_sidebar_state="expanded")

# --- UTILS: LOAD ASSETS ---
def apply_custom_assets():
    base_dir = os.path.dirname(__file__)
    gif_path = os.path.join(base_dir, "6.gif")
    
    # Background GIF Loader
    if os.path.exists(gif_path):
        with open(gif_path, "rb") as f:
            base64_gif = base64.b64encode(f.read()).decode()
        
        st.markdown(
            f'''
            <style>
            .stApp {{
                background: url("data:image/gif;base64,{base64_gif}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            /* Mencegah tabrakan icon & text di header */
            .animate-header {{
                display: flex;
                align-items: center;
                gap: 15px;
                background: transparent !important;
            }}
            </style>
            ''', 
            unsafe_allow_html=True
        )

    # Inject Anime.js & Dropdown Animation Logic
    st.components.v1.html(
        """
        <script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.1/anime.min.js"></script>
        <script>
            const doc = window.parent.document;
            
            const runAnimations = () => {
                // 1. Animasi Dropdown untuk Menu Sidebar
                window.parent.anime({
                    targets: doc.querySelectorAll('.nav-link'),
                    translateY: [-50, 0],
                    opacity: [0, 1],
                    delay: window.parent.anime.stagger(100),
                    easing: 'easeOutElastic(1, .6)'
                });

                // 2. Animasi Judul Page (Fade & Slide)
                window.parent.anime({
                    targets: doc.querySelectorAll('.animate-header'),
                    translateX: [-30, 0],
                    opacity: [0, 1],
                    duration: 1500,
                    easing: 'easeOutExpo'
                });
            };

            // Observer untuk re-trigger saat pindah menu
            const observer = new MutationObserver(() => runAnimations());
            observer.observe(doc.querySelector('.stApp'), { childList: true, subtree: true });
            
            setTimeout(runAnimations, 500);
        </script>
        """,
        height=0,
    )

apply_custom_assets()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div id="text-split"><h2 class="animate-header">🚀 DEMANDSENSE</h2></div>', unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["EDA", "Prediction"],
        icons=["bar-chart-line-fill", "cpu-fill"],
        default_index=0,
        styles={
            "container": {"background-color": "transparent", "padding": "5px"},
            "nav-link": {
                "font-size": "16px", "text-align": "left", "margin": "10px 0px",
                "color": "white", "border-radius": "10px", 
                "background": "rgba(255, 255, 255, 0.05)",
                "transition": "all 0.3s"
            },
            "nav-link-selected": {
                "background-color": "#3b82f6",
                "box-shadow": "0px 5px 15px rgba(59, 130, 246, 0.4)"
            }
        }
    )

# --- MAIN CONTENT WRAPPER ---
st.markdown('<div class="main-content">', unsafe_allow_html=True)
if selected == "EDA":
    eda.run()
elif selected == "Prediction":
    prediction.run()
st.markdown('</div>', unsafe_allow_html=True)