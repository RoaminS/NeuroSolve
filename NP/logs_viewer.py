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
import requests
import qrcode
from io import BytesIO


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


# === Fonction QR
def generate_qr_for_session(session_zip_path):
    url_placeholder = f"https://neurosolve.local/sessions/{os.path.basename(session_zip_path)}"
    qr = qrcode.make(url_placeholder)
    buffer = BytesIO()
    qr.save(buffer, format="PNG")
    return buffer.getvalue()

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

# === Générer automatiquement un fichier summary.json
def generate_session_summary(predictions, output_path):
    df = pd.DataFrame(predictions)
    total = len(df)
    alerts = df["alert"].sum() if "alert" in df else 0
    duration = round(df["time_sec"].max() - df["time_sec"].min(), 2)
    avg_prob = round(df["prob_class_1"].mean(), 3)

    summary = {
        "session": os.path.basename(output_path).replace("summary.json", ""),
        "nb_frames": total,
        "nb_alerts": int(alerts),
        "alert_rate": round(alerts / total, 3) if total else 0,
        "duration_sec": duration,
        "avg_prob_class_1": avg_prob,
        "timestamp_generated": datetime.now().isoformat()
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary

# === SUMMARY JSON
summary_path = os.path.join(session_path, "summary.json")

if os.path.exists(json_file):
    with open(json_file) as f:
        predictions = json.load(f)
    summary = generate_session_summary(predictions, summary_path)
    st.markdown("### 🧠 Synthèse de la session")
    st.json(summary)

# === Graphe résumé: Synthèse de la session
import plotly.express as px
if summary:
    st.markdown("### 📈 Graphique de synthèse")

    fig = px.bar(
        x=["Alert Rate (%)", "Durée (s)", "Frames"],
        y=[summary["alert_rate"] * 100, summary["duration_sec"], summary["nb_frames"]],
        text=[f"{summary['alert_rate']*100:.1f}%", f"{summary['duration_sec']}s", f"{summary['nb_frames']}"],
        labels={"x": "Indicateur", "y": "Valeur"},
        title="📊 Indicateurs clés de la session EEG"
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)


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
st.markdown("### 🔗 QR Code de partage (placeholder URL)")
if os.path.exists(zip_path):
    qr_img = generate_qr_for_session(zip_path)
    st.image(qr_img, width=200, caption="Scanne pour accéder à la session")


# === Bouton ZIP
if st.button("📁 Créer une archive ZIP de cette session"):
    zip_path = zip_session(session_path)

# === Bouton DOWNLOAD ZIP
if os.path.exists(zip_path):
    with open(zip_path, "rb") as f:
        st.download_button(
            label="⬇️ Télécharger la session zippée",
            data=f,
            file_name=os.path.basename(zip_path),
            mime="application/zip"
        )


# === Bouton PUSH
if st.button("📤 Envoyer la session à l’API Flask"):
    zip_path = session_path + ".zip"
    if os.path.exists(zip_path):
        push_session_to_api(zip_path)
    else:
        st.warning("💡 Zipper la session avant de l’envoyer.")

st.markdown("Made with ❤️ by **Kocupyr Romain** & `multi_gpt_api`")
