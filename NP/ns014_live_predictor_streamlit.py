#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ns014_live_predictor_streamlit.py

Licence : CC BY-NC-SA 4.0

✅ EEG prédiction live (LSL) avec XAI (SHAP)
✅ Export .csv / .json / .gif / résumé + QR
✅ Envoi email si alerte détectée

Auteur : Kocupyr Romain  
Dev    : multi_gpt_api

le lancer: run streamlite ns014_live_predictor_streamlit.py
"""

import os, json, pickle, smtplib, shutil, time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
import torch
import tensorflow as tf
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

import qrcode
from playsound import playsound
from pylsl import StreamInlet, resolve_stream
from ripser import ripser
from ns015_shap_live import shap_explain_live


# === CONFIGURATION
MODEL_PATH = "ns013_results/model.pkl"
USE_ADFORMER = os.path.exists("ns013_results/model_adformer.pth")
ALERT_SOUND = "assets/alert_sound.mp3"
WINDOW_SIZE = 512
FS = 128
THRESHOLD = 0.85

# === Dossier logs auto
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_DIR = os.path.join("logs", f"session_{timestamp_str}")
os.makedirs(LOG_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(LOG_DIR, f"ns014_predictions_{timestamp_str}.csv")
JSON_LOG = os.path.join(LOG_DIR, "predictions_log.json")
GIF_FILE = os.path.join(LOG_DIR, "prediction_live.gif")


# === QR Code
def generate_qr_for_zip(zip_path):
    placeholder_url = f"https://neurosolve.local/sessions/{os.path.basename(zip_path)}"
    qr = qrcode.make(placeholder_url)
    buf = BytesIO()
    qr.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

# === Email config
def load_notifier_config(path="notifier_config.json"):
    if not os.path.exists(path):
        st.warning("⚠️ Fichier notifier_config.json manquant.")
        return None
    with open(path) as f:
        return json.load(f)

def send_email_alert(summary, config, zip_path=None):
    msg = MIMEMultipart()
    msg['From'] = config["sender_email"]
    msg['To'] = ", ".join(config["recipients"])
    msg['Subject'] = f"[NeuroSolve] Alerte cognitive - {summary['session_folder']}"
    body = f"""
🧠 Session : {summary['session_folder']}
Frames : {summary['nb_frames']}
Alertes : {summary['nb_alerts']} ({summary['alert_rate']*100:.1f}%)
Durée : {summary['duration_sec']}s
Prob moy classe 1 : {summary['avg_prob_class_1']}
    """
    msg.attach(MIMEText(body, "plain"))

    if zip_path:
        qr = qrcode.make(f"https://neurosolve.local/sessions/{os.path.basename(zip_path)}")
        buf = BytesIO()
        qr.save(buf, format="PNG")
        buf.seek(0)
        msg.attach(MIMEImage(buf.read(), name="qr_session.png"))

    try:
        with smtplib.SMTP(config["smtp_server"], config["smtp_port"]) as server:
            server.starttls()
            server.login(config["sender_email"], config["password"])
            server.send_message(msg)
        print("📧 Email envoyé avec succès.")
    except Exception as e:
        print(f"❌ Email failed: {e}")

# === EEG
def get_lsl_segment(n=512, channels=1):
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    buffer = []
    while len(buffer) < n:
        sample, _ = inlet.pull_sample()
        buffer.append(sample[:channels])
    return np.array(buffer).T[0]

# === Features
def extract_tda_features(x):
    traj = np.array([x[i:i+3] for i in range(len(x)-3)])
    b1 = ripser(traj)['dgms'][1]
    if len(b1):
        persistence = max(d - b for b, d in b1)
        return [persistence, b1[0][0], b1[0][1]]
    return [0, 0, 0]

def extract_wavelet_features(data, wavelet='db4', level=4):
    import pywt
    coeffs = pywt.wavedec(data, wavelet, level=level)
    arr = pywt.coeffs_to_array(coeffs)[0]
    return arr.flatten()

# === Prédiction
def predict_segment(model, scaler, segment, use_adformer=False):
    tda_feat = extract_tda_features(segment)
    wavelet_feat = extract_wavelet_features(segment)
    full_feat = np.concatenate([tda_feat, wavelet_feat])
    full_feat = scaler.transform([full_feat])

    if use_adformer:
        proba = model.predict(full_feat)[0]  # Tensorflow retourne array
        pred = int(np.argmax(proba))
    else:
        pred = model.predict(full_feat)[0]
        proba = model.predict_proba(full_feat)[0]

    return pred, proba


# === Envoi automatique du ZIP à une API Flask
def push_zip_to_api(zip_path, endpoint="http://localhost:6000/upload_session"):
    try:
        with open(zip_path, 'rb') as f:
            files = {'file': (os.path.basename(zip_path), f)}
            r = requests.post(endpoint, files=files)
        if r.status_code == 200:
            st.success("📡 Session envoyée à l’API Flask !")
            st.json(r.json())
        else:
            st.error(f"❌ Échec de l’envoi ({r.status_code})")
    except Exception as e:
        st.error(f"⚠️ Erreur API : {e}")


# === LIVE LOOP
def live_loop(config=None):

    if USE_ADFORMER:
        model = torch.load("ns013_results/model_adformer.pth", map_location=torch.device('cpu'))
        model.eval()
        scaler = np.load("ns013_results/model_scaler_adformer.npz", allow_pickle=True)["scaler"][()]
    else:
        model = pickle.load(open("ns013_results/model.pkl", "rb"))
        scaler = np.load("ns013_results/model_scaler.npz", allow_pickle=True)["scaler"][()]
            
    predictions = []
    gif_frames = []

    for i in range(20):
        t = round(i * 2, 2)
        segment = get_lsl_segment()
        pred, prob = predict_segment(model, scaler, segment, use_adformer)


        predictions.append({
            "time_sec": t,
            "iteration": i,
            "prediction": int(pred),
            "prob_class_0": float(prob[0]),
            "prob_class_1": float(prob[1]),
            "timestamp": time.time(),
            "subject": "subject_01",
            "alert": prob[1] > THRESHOLD
        })

        if prob[1] > THRESHOLD:
            playsound(ALERT_SOUND)

        if i % 5 == 0:
            shap_explain_live(segment, model, scaler)

        # Graph
        plt.figure(figsize=(6,3))
        plt.plot(segment)
        plt.title(f"{t}s - Pred: {pred}")
        fname = os.path.join(LOG_DIR, f"frame_{i:03}.png")
        plt.savefig(fname)
        gif_frames.append(fname)
        plt.close()

        time.sleep(2)

    # Export
    df = pd.DataFrame(predictions)
    df.to_csv(RESULTS_CSV, index=False)
    with open(JSON_LOG, "w") as f:
        json.dump(predictions, f, indent=2)

    # GIF
    import imageio
    imageio.mimsave(GIF_FILE, [imageio.imread(f) for f in gif_frames], duration=0.6)
    for f in gif_frames: os.remove(f)

    # Résumé
    summary = {
        "session_folder": os.path.basename(LOG_DIR),
        "nb_frames": len(predictions),
        "nb_alerts": sum(p["alert"] for p in predictions),
        "alert_rate": round(sum(p["alert"] for p in predictions) / len(predictions), 3),
        "duration_sec": round(predictions[-1]["time_sec"] - predictions[0]["time_sec"], 2),
        "avg_prob_class_1": round(np.mean([p["prob_class_1"] for p in predictions]), 3),
        "timestamp_generated": datetime.now().isoformat()
    }

    # === Sauvegarde summary.json (pour logs_viewer)
    summary_json_path = os.path.join(LOG_DIR, "summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"🧠 Résumé sauvegardé : {summary_json_path}")


    # Résumé global
    csv_sum = "sessions_summary.csv"
    df_sum = pd.read_csv(csv_sum) if os.path.exists(csv_sum) else pd.DataFrame()
    pd.concat([df_sum, pd.DataFrame([summary])]).to_csv(csv_sum, index=False)

    # ZIP
    zip_path = shutil.make_archive(LOG_DIR, 'zip', LOG_DIR)

    # Email
    if summary["nb_alerts"] > 0 and config:
        send_email_alert(summary, config, zip_path)
        
    # API
    if config.get("push_to_api", False):
        push_zip_to_api(zip_path)

    return zip_path
    
# === UI STREAMLIT
st.set_page_config(page_title="EEG Live Predictor")
st.title("🧠 NeuroSolve – Prédiction EEG Temps Réel")

# ✅ Choix du modèle par utilisateur
model_type = st.selectbox("🧠 Choisis le modèle :", ["RandomForest (.pkl)", "AdFormer (.pth)"])
use_adformer = model_type == "AdFormer (.pth)"

config = load_notifier_config()

if st.button("🧠 Lancer la prédiction EEG (LSL)"):
    with st.spinner("Analyse en cours..."):
        zip_path = live_loop(config=config, use_adformer=use_adformer)
    st.success("✅ Session terminée")
    st.image(generate_qr_for_zip(zip_path), width=220)
    st.markdown("### 🔗 QR Code session")
    st.image(generate_qr_for_zip(zip_path), width=220, caption="Scanne pour voir la session EEG 🧠")

    with open(zip_path, "rb") as f:
        st.download_button(
            label="⬇️ Télécharger le ZIP de la session",
            data=f,
            file_name=os.path.basename(zip_path),
            mime="application/zip"
        )
