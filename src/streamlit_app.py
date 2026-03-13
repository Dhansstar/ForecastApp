import streamlit as st
import base64
import os
from streamlit_option_menu import option_menu
import eda
import prediction

st.set_page_config(page_title="DemandSense", layout="wide")

# --- LOAD CSS & GIF ---
def local_css(file_name):
    base_dir = os.path.dirname(__file__)
    # Path GIF
    gif_path = os.path.join(base_dir, "6.gif")
    with open(gif_path, "rb") as f:
        base64_gif = base64.b64encode(f.read()).decode()
    
    # Path CSS
    with open(os.path.join(base_dir, file_name)) as f:
        css_content = f.read().replace("REPLACE_WITH_YOUR_BASE64_HERE", base64_gif)
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

local_css("style.css")

# --- JAVASCRIPT RADAR ---
st.components.v1.html(
    """
    <script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.1/anime.min.js"></script>
    <script>
        const doc = window.parent.document;
        const runAnime = () => {
            // Animasi Dropdown Sidebar (Stagger)
            window.parent.anime({
                targets: doc.querySelectorAll('.nav-link'),
                translateY: [-20, 0],
                opacity: [0, 1],
                delay: window.parent.anime.stagger(100),
                easing: 'easeOutQuad'
            });
            
            // Radar Headers
            const check = setInterval(() => {
                const headers = doc.querySelectorAll('.animate-header:not([data-animated])');
                if(headers.length > 0) {
                    window.parent.anime({
                        targets: headers,
                        translateY: [-30, 0],
                        opacity: [0, 1],
                        duration: 1000,
                        easing: 'easeOutExpo'
                    });
                    headers.forEach(h => h.setAttribute('data-animated', 'true'));
                }
                
                const squares = doc.querySelectorAll('.square:not([data-animated])');
                if(squares.length > 0) {
                    window.parent.anime({
                        targets: squares,
                        translateX: '10rem',
                        rotate: '1turn',
                        duration: 2000,
                        direction: 'alternate',
                        loop: true,
                        easing: 'easeInOutSine',
                        opacity: [0, 1]
                    });
                    squares.forEach(s => s.setAttribute('data-animated', 'true'));
                }
            }, 500);
        };
        setTimeout(runAnime, 500);
    </script>
    """,
    height=0,
)

# --- SIDEBAR & CONTENT ---
with st.sidebar:
    st.markdown('<div id="text-split"><h2 class="animate-header">🚀 DEMANDSENSE</h2></div>', unsafe_allow_html=True)
    selected = option_menu(None, ["EDA", "Prediction"], 
        icons=["bar-chart-line-fill", "cpu-fill"], 
        styles={"nav-link": {}, "nav-link-selected": {}}) # Style pindah ke CSS

if selected == "EDA":
    eda.run()
else:
    prediction.run()