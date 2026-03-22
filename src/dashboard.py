import streamlit as st
import streamlit.components.v1 as components
import os
import sys
import subprocess
import atexit
import socket

# ---------------- 4. FULLSCREEN LAYOUT ---------------- 
st.set_page_config(
    page_title="Intrusion Detection Command Centre",
    page_icon="🎖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Padding and Streamlit UI elements
st.markdown("""
<style>
/* 3. REMOVE DEFAULT STREAMLIT UI */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* NO PADDINGS/MARGINS */
.block-container {
    padding-top: 0rem;
    padding-bottom: 0rem;
    padding-left: 0rem;
    padding-right: 0rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Auto-Boot Backend ----------------
@st.cache_resource
def start_backend_daemon():
    # Detect absolute path of intrusion_logs.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_script = os.path.join(script_dir, "intrusion_logs.py")
    
    # Check if port 5050 is already in use
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        is_running = s.connect_ex(('localhost', 5050)) == 0
        
    if not is_running:
        print("🚀 Booting Hardware Detection Backend Process...")
        p = subprocess.Popen([sys.executable, backend_script])
        
        def kill_backend():
            print("🛑 Killing Hardware Backend...")
            p.kill()
            
        atexit.register(kill_backend)
        return p
    else:
        print("⚡ Hardware Backend is already running on port 5050.")
        return None

start_backend_daemon()

# ---------------- 1. LOAD EXISTING HTML FILE ----------------
html_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")

if os.path.exists(html_file_path):
    with open(html_file_path, "r") as f:
        html_content = f.read()
    
    # ---------------- 2. RENDER HTML INSIDE STREAMLIT ----------------
    components.html(html_content, height=1000, scrolling=True)
else:
    st.error(f"❌ LOG ERROR: {html_file_path} not found.")
