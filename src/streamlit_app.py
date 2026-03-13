import streamlit as st
import base64
import os
from streamlit_option_menu import option_menu
import eda
import prediction

# 1. SET CONFIG
st.set_page_config(page_title="DemandSense", layout="wide", initial_sidebar_state="expanded")

# --- UTILS: LOAD ASSETS ---
def get_base64_bin(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def apply_custom_assets():
    # Background GIF
    gif_path = os.path.join(os.path.dirname(__file__), "6.gif")
    if os.path.exists(gif_path):
        base64_gif = get_base64_bin(gif_path)
        st.markdown(
            f'''
            <style>
            .stApp {{
                background: url("data:image/gif;base64,{base64_gif}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            /* Efek Dropdown & Hover untuk Navigasi */
            .nav-link {{
                border-radius: 10px !important;
                margin: 5px 0px !important;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            }}
            .nav-link:hover {{
                transform: translateX(10px) scale(1.05) !important;
                background-color: rgba(59, 130, 246, 0.1) !important;
            }}
            </style>
            ''', 
            unsafe_allow_html=True
        )

    # 2. External CSS
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # 3. Inject Anime.js Logic (FULL WAAPI ANIMATION)
    st.components.v1.html(
        """
        <script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.1/anime.min.js"></script>
        <script>
            const doc = window.parent.document;
            
            const runAnimations = () => {
                // --- 1. Animasi Header (Fade In) ---
                doc.querySelectorAll('.animate-header').forEach(el => {
                    window.parent.anime({
                        targets: el,
                        translateY: [-20, 0],
                        opacity: [0, 1],
                        easing: 'easeOutExpo',
                        duration: 1200
                    });
                });

                // --- 2. Animasi Navigasi (Dropdown Effect) ---
                // Kita nangkep elemen menu navigasi biar pas ganti menu ada transisinya
                doc.querySelectorAll('.nav-link').forEach((el, i) => {
                    window.parent.anime({
                        targets: el,
                        translateX: [-50, 0],
                        opacity: [0, 1],
                        delay: i * 150,
                        easing: 'easeOutElastic(1, .8)',
                        duration: 1000
                    });
                });

                // --- 3. Animasi Square (WAAPI Style) ---
                doc.querySelectorAll('.square').forEach(el => {
                    window.parent.anime({
                        targets: el,
                        translateX: '15rem',
                        scale: 1.25,
                        skew: -45,
                        rotate: '1turn',
                        duration: 2500,
                        direction: 'alternate',
                        loop: true,
                        easing: 'easeInOutQuad'
                    });
                });
            };

            setTimeout(runAnimations, 600);
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
        icons=["bar-chart-fill", "cpu-fill"],
        default_index=0,
        styles={
            "container": {"background-color": "transparent"},
            "nav-link": {
                "font-weight": "bold", 
                "color": "white", 
                "text-align": "left", 
                "margin": "0px",
                "--hover-color": "rgba(59, 130, 246, 0.2)"
            },
            "nav-link-selected": {
                "background-color": "rgba(59, 130, 246, 0.3)", 
                "border-left": "4px solid #3b82f6",
                "font-size": "1.1rem"
            }
        }
    )

# --- ROUTING ---
st.markdown('<div class="main-content">', unsafe_allow_html=True)
if selected == "EDA":
    eda.run()
elif selected == "Prediction":
    prediction.run()
st.markdown('</div>', unsafe_allow_html=True)