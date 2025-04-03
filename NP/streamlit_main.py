"""
streamlit_main.py

🧠 Lancement unifié de la suite NeuroSolve (Streamlit)
Licence : Creative Commons BY-NC-SA 4.0
"""

import streamlit as st
import runpy
import os

st.set_page_config(page_title="🧠 NeuroSolve Suite", layout="wide")
st.sidebar.title("🧠 Menu Principal NeuroSolve")

# Choix utilisateur
section = st.sidebar.radio("📂 Choisir un module :", [
    "🎯 Live Prédiction EEG",
    "📁 Visualiseur de sessions",
    "🌍 Dashboard global",
    "📤 Config Email / Notif",
    "🧱 Résumé Cross-Session"
])

# Mapping clean
module_mapping = {
    "🎯 Live Prédiction EEG": "ns014_live_predictor_streamlit.py",
    "📁 Visualiseur de sessions": "logs_viewer.py",
    "🌍 Dashboard global": "global_dashboard.py",
    "📤 Config Email / Notif": "streamlit_email_config.py",
    "🧱 Résumé Cross-Session": "generate_sessions_summary.py"
}

selected_file = module_mapping.get(section)

if selected_file and os.path.exists(selected_file):
    st.markdown(f"### 🚀 Module sélectionné : `{selected_file}`")
    runpy.run_path(selected_file, run_name="__main__")
else:
    st.error("❌ Fichier introuvable ou erreur de sélection.")
