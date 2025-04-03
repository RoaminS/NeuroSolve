"""
logs_viewer.py
Licence : Creative Commons BY-NC-SA 4.0
Auteurs : 
    - Kocupyr Romain (chef de projet) : rkocupyr@gmail.com
    - GPT multi_gpt_api 

💡 Commande pour le lancer :

streamlit run logs_viewer.py
"""

import streamlit as st
import os
import pandas as pd
import json
from PIL import Image
import shutil
import zipfile


st.set_page_config(layout="wide", page_title="🧠 NeuroSolve Logs Viewer")

st.title("🧠 NeuroSolve — Session Explorer")
st.markdown("📁 Navigation dans les sessions EEG sauvegardées par `ns014_live_predictor.py`")


# === Sélection de la session
LOGS_DIR = "logs"
sessions = sorted([d for d in os.listdir(LOGS_DIR) if d.startswith("session_")])

if not sessions:
    st.warning("Aucune session trouvée dans le dossier `logs/`")
    st.stop()

selected_session = st.selectbox("🧠 Sélectionne une session :", sessions)
session_path = os.path.join(LOGS_DIR, selected_session)

def push_session_to_api(zip_path, api_url="http://localhost:5000/upload_session"):
    try:
        with open(zip_path, 'rb') as f:
            files = {'file': (os.path.basename(zip_path), f)}
            r = requests.post(api_url, files=files)
        if r.status_code == 200:
            st.success("📡 Session envoyée à l’API !")
            st.json(r.json())
        else:
            st.error(f"❌ Échec de l’envoi : {r.status_code}")
    except Exception as e:
        st.error(f"⚠️ Erreur d’envoi : {e}")

# === Chargement des fichiers
csv_file = [f for f in os.listdir(session_path) if f.endswith(".csv")]
json_file = os.path.join(session_path, "predictions_log.json")
gif_file = os.path.join(session_path, "prediction_live.gif")
shap_img = os.path.join(session_path, "shap_live_frame.png")

# === PREDICTIONS CSV
if csv_file:
    st.markdown("### 📊 Prédictions EEG")
    df = pd.read_csv(os.path.join(session_path, csv_file[0]))
    st.dataframe(df.tail(20), use_container_width=True)
else:
    st.info("Aucune prédiction CSV trouvée.")

# === ALERTES
st.markdown("### ⚠️ Alertes cognitives")

if os.path.exists(json_file):
    with open(json_file) as f:
        data = json.load(f)
        alerts = [p for p in data if p.get("alert") is True]
    if alerts:
        for alert in alerts:
            st.error(f"⚠️ [itération {alert['iteration']}] prob1={alert['prob_class_1']} @ {alert['time_sec']}s")
    else:
        st.success("✅ Aucune alerte détectée")
else:
    st.warning("Log JSON non trouvé.")

# === VISU GIF
st.markdown("### 🎞️ EEG Timeline (GIF)")

if os.path.exists(gif_file):
    st.image(gif_file)
else:
    st.info("GIF non trouvé.")

# === SHAP
st.markdown("### 🔍 SHAP (Image XAI)")

if os.path.exists(shap_img):
    st.image(shap_img)
else:
    st.info("Image SHAP non disponible.")

# == Zip Session
def zip_session(session_path):
    zip_name = session_path + ".zip"
    shutil.make_archive(session_path, 'zip', session_path)
    st.success(f"📦 Session compressée : {zip_name}")
    return zip_name

# === FOOTER
st.markdown("---")
st.markdown(f"📂 Session : `{selected_session}`")
st.markdown("## 📦 Export & API")

# Bouton ZIP
if st.button("📁 Créer une archive ZIP de cette session"):
    zip_path = zip_session(session_path)

# Bouton PUSH
if st.button("📤 Envoyer la session à l’API Flask"):
    zip_path = session_path + ".zip"
    if os.path.exists(zip_path):
        push_session_to_api(zip_path)
    else:
        st.warning("💡 Zipper la session avant de l’envoyer.")

st.markdown("Made with ❤️ by **Kocupyr Romain** & `multi_gpt_api`")
