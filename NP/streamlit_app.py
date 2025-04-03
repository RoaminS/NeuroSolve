'''
Ce fichier crée une UI web complète en local ou sur n'importe quel hébergement Streamlit :

Il affiche :
🎞️ Le GIF prediction_live.gif

🔍 L’image SHAP en temps réel shap_live_frame.png

📊 La table des prédictions ns020_predictions.csv

⚠️ Les alertes de alerts_detected.json (si elles existent)
pip install streamlit
streamlit run streamlit_app.py

Auteur : Kocupyr Romain
Dev    : multi_gpt_api
Licence : CC BY-NC-SA 4.0
'''

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from PIL import Image

st.set_page_config(layout="wide", page_title="NeuroSolve Streamlit UI")

# === TITRE
st.title("🧠 NeuroSolve – Streamlit Web Dashboard")
st.subheader("EEG NP Solver • XAI Live • Alert Monitoring")

# === FICHIERS REQUIS
json_path = "predictions_log.json"
gif_path = "prediction_live.gif"
shap_img = "shap_live_frame.png"
alert_path = "alerts_detected.json"
csv_path = "ns020_predictions.csv"

# === COLONNES
col1, col2 = st.columns([1, 2])

# === LIVE GIF EEG
with col1:
    st.markdown("### 🎞️ EEG Timeline (Live Prediction)")
    if os.path.exists(gif_path):
        st.image(gif_path)
    else:
        st.warning("GIF non trouvé : `prediction_live.gif`")

# === SHAP VISU
with col2:
    st.markdown("### 🔍 SHAP – Interprétation en direct")
    if os.path.exists(shap_img):
        st.image(shap_img)
    else:
        st.warning("Image SHAP non trouvée : `shap_live_frame.png`")

# === PREDICTION TABLE
st.markdown("### 📊 Prédictions EEG (log complet)")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.dataframe(df.tail(10), use_container_width=True)
else:
    st.warning("Fichier `ns020_predictions.csv` introuvable.")

# === ALERTS
st.markdown("### ⚠️ Alertes cognitives déclenchées")

if os.path.exists(alert_path):
    with open(alert_path) as f:
        alerts = json.load(f)
    for alert in alerts:
        st.error(f"⚠️ Sujet : {alert['frames']} — Proba Moyenne : {alert['mean_prob']} — Durée : {alert['duration']}s")
else:
    st.info("Aucune alerte détectée ou fichier `alerts_detected.json` manquant.")

# === FOOTER
st.markdown("---")
st.markdown("Made with ❤️ by **Kocupyr Romain** & multi_gpt_api")
st.markdown("[GitHub NeuroSolve](https://github.com/RoaminS/NeuroSolve)")
