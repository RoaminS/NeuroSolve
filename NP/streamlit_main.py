"""
streamlit_main.py
Licence : Creative Commons BY-NC-SA 4.0
Auteurs : 
    - Kocupyr Romain (chef de projet) : rkocupyr@gmail.com
    - GPT multi_gpt_api (OpenAI)
"""

import streamlit as st
import os
import runpy

st.set_page_config(page_title="🧠 NeuroSolve Suite", layout="wide")
st.title("🧠 NeuroSolve - Interface Modules Dynamiques")

APPS_DIR = "apps"

# === Scanner tous les .py de apps/
modules = [f for f in os.listdir(APPS_DIR) if f.endswith(".py")]
modules.sort()

# Mapping clean
module_labels = {
    "ns014_live_predictor_streamlit.py": "🎯 Live Prédiction EEG",
    "logs_viewer.py": "📁 Visualiseur de sessions",
    "global_dashboard.py": "🌍 Dashboard Global",
    "generate_sessions_summary.py": "🧱 Résumé Cross-Session",
    "streamlit_email_config.py": "📤 Configuration Notifications"
}

# Choix UI
selected_file = st.sidebar.selectbox(
    "📂 Sélectionne un module NeuroSolve",
    modules,
    format_func=lambda name: module_labels.get(name, f"📦 {name}")
)

selected_path = os.path.join(APPS_DIR, selected_file)

# === Run
st.markdown(f"### 🚀 Module lancé : `{selected_file}`")

if selected_path and os.path.exists(selected_path):
    runpy.run_path(selected_path, run_name="__main__")
else:
    st.error("❌ Module introuvable.")
