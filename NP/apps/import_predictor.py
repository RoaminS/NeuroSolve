import os
import json
import pickle
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import mne
import tensorflow as tf
from datetime import datetime
from io import BytesIO
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import qrcode
from playsound import playsound
from ripser import ripser
from ns015_shap_live import shap_explain_live

# === CONFIGURATION
MODEL_PATH = "ns013_results/model.pkl"
USE_ADFORMER = os.path.exists("ns013_results/model_adformer.h5")
ALERT_SOUND = "assets/alert_sound.mp3"
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
        print(f"❌ Échec de l'envoi de l'email : {e}")

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

# === Traitement des fichiers EEG
def process_eeg_file(file_path, file_type, model, scaler, use_adformer=False):
    if file_type == 'set':
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
    elif file_type == 'edf':
        raw = mne.io.read_raw_edf(file_path, preload=True)
    else:
        st.error("Format de fichier non supporté.")
        return

    data, times = raw[:]
    predictions = []
    gif_frames = []

    for i in range(0, data.shape[1], 512):  # Supposons des segments de 512 échantillons
        segment = data[:, i:i+512]
        if segment.shape[1] < 512:
            break
        pred, prob = predict_segment(model, scaler, segment.flatten(), use_adformer)

        predictions.append({
            "time_sec": times[i],
            "iteration": i // 512,
            "prediction": int(pred),
            "prob_class_0": float(prob[0]),
            "prob_class_1": float(prob[1]),
            "timestamp": time.time(),
            "subject": "subject_01",
            "alert": prob[1] > THRESHOLD
        })

        if prob[1] > THRESHOLD:
            playsound(ALERT_SOUND)

        if (i // 512) % 5 == 0:
            shap_explain_live(segment.flatten(), model, scaler)

        # Graph
        plt.figure(figsize=(6,3))
        plt.plot(segment.T)
        plt.title(f"{times[i]:.2f}s - Pred: {pred}")
        fname = os.path.join(LOG_DIR, f"frame_{i//512:03}.png")
        plt.savefig(fname)
        gif_frames.append(fname)
        plt.close()

    # Export
    df = pd.DataFrame(predictions)
    df.to_csv(RESULTS_CSV, index=False)
    with open(JSON_LOG, "w") as f:
        json.dump(predictions, f, indent=2)

    # GIF
    import imageio
    imageio.mimsave(G
::contentReference[oaicite:2]{index=2}
 
