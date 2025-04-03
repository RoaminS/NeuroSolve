"""
streamlit_main.py

Licence : Creative Commons BY-NC-SA 4.0
Auteurs : 
    - Kocupyr Romain (chef de projet) : rkocupyr@gmail.com
    - GPT multi_gpt_api (OpenAI)
"""

import streamlit as st

st.set_page_config(page_title="🧠 NeuroSolve Suite", layout="wide")
st.sidebar.title("🧠 NeuroSolve UI")
section = st.sidebar.radio("Choisir une vue :", [
    "🎯 Live Prédiction EEG",
    "📁 Visualiseur de sessions",
    "🌍 Dashboard global",
    "📤 Config Email / Notif",
    "🧱 Générer Résumé Cross-Session"
])

if section == "🎯 Live Prédiction EEG":
    exec(open("ns014_live_predictor_streamlit.py").read())

elif section == "📁 Visualiseur de sessions":
    exec(open("logs_viewer.py").read())

elif section == "🌍 Dashboard global":
    exec(open("global_dashboard.py").read())

elif section == "📤 Config Email / Notif":
    exec(open("streamlit_email_config.py").read())

elif section == "🧱 Générer Résumé Cross-Session":
    exec(open("generate_sessions_summary.py").read())
