"""
streamlit_multipage_autodiscover.py

Licence : Creative Commons BY-NC-SA 4.0

UI NeuroSolve Multi-Apps (Auto)
📁 Scan dynamique de tous les scripts dans apps/

Auteurs : 
    - Kocupyr Romain (chef de projet) : rkocupyr@gmail.com
    - GPT multi_gpt_api (OpenAI)
"""""

import streamlit as st
import os

st.set_page_config(page_title="🧠 NeuroSolve Suite", layout="wide")
st.sidebar.title("🧠 Applications NeuroSolve")

APP_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps"))
apps = [f for f in os.listdir(APP_FOLDER) if f.endswith(".py")]
apps.sort()

selected_app = st.sidebar.selectbox("📂 Sélectionne un module :", apps)

if selected_app:
    st.subheader(f"🚀 Lancement de : `{selected_app}`")
    app_path = os.path.join(APP_FOLDER, selected_app)
    with open(app_path, "r") as f:
        code = f.read()
        exec(code, globals())
