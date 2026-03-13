import streamlit as st
import base64
import os
from streamlit_option_menu import option_menu
import eda
import prediction

# 1. SET CONFIG (Wajib di baris pertama)
st.set_page_config(page_title="DemandSense", layout="wide", initial_sidebar_state="expanded")

# --- UTILS: LOAD ASSETS ---
def get_base64_bin(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def apply_custom_assets():
    # 1. Background GIF (Base64) - Biar nembus ke dashboard
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
            .stApp::before {{
                content: "";
                position: absolute;
                top: 0; left: 0; width: 100%; height: 100%;
                background-color: rgba(0, 0, 0, 0.5); /* Overlay kegelapan */
                z-index: -1;
            }}
            </style>
            ''', 
            unsafe_allow_html=True
        )

    # 2. External CSS - Load dari file style.css lo
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # 3. Inject Anime.js Logic (FULL COMBINED)
    st.components.v1.html(
        """
        <script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.1/anime.min.js"></script>
        <script>
            // Fungsi untuk menjalankan animasi setelah DOM ready
            const runAnimations = () => {
                const doc = window.parent.document;
                
                // --- Animasi Judul Header ---
                doc.querySelectorAll('.animate-header').forEach(el => {
                    window.parent.anime({
                        targets: el,
                        translateX: [30, 0],
                        opacity: [0, 1],
                        easing: 'easeOutExpo',
                        duration: 1500,
                        delay: 300
                    });
                });

                // --- Animasi Kotak (Square) Lo ---
                doc.querySelectorAll('.square').forEach(el => {
                    window.parent.anime({
                        targets: el,
                        translateX: '15rem',
                        scale: 1.25,
                        skew: -45,
                        rotate: '1turn',
                        duration: 2000,
                        direction: 'alternate',
                        loop: true,
                        easing: 'easeInOutQuad'
                    });
                });
            };

            // Jalankan animasi
            setTimeout(runAnimations, 500); 
        </script>
        """,
        height=0,
    )

# Execute Styling & Animations
apply_custom_assets()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    # Judul Sidebar dengan efek text-split
    st.markdown('<div id="text-split"><h2 class="animate-header">🚀 DEMANDSENSE</h2></div>', unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["EDA", "Prediction"],
        icons=["bar-chart-fill", "cpu-fill"],
        default_index=0,
        styles={
            "container": {"background-color": "transparent"},
            "nav-link": {"font-weight": "bold", "color": "white", "transition": "0.3s"},
            "nav-link-selected": {"background-color": "rgba(59, 130, 246, 0.2)", "border": "1px solid #3b82f6"}
        }
    )

# --- MAIN ROUTING ---
# Div wrapper biar styling di style.css lo kena ke main area
st.markdown('<div class="main-content">', unsafe_allow_html=True)

if selected == "EDA":
    eda.run()
elif selected == "Prediction":
    prediction.run()

st.markdown('</div>', unsafe_allow_html=True)