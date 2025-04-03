#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ns014_live_predictor_streamlit.py

✅ Prédiction temps réel EEG avec modèle NS013
✅ Extraction TDA + ondelettes sur fenêtre glissante
✅ Chargement d’un modèle ML (RandomForest / XGBoost)
✅ Affichage live des prédictions + export CSV

Auteur : Kocupyr Romain
Dev    : multi_gpt_api, Grok3
Licence : CC BY-NC-SA 4.0
"""

import os
import pickle
import numpy as np
import pandas as pd
import datetime
import mne
import time
import shutil
import smtplib
import qrcode
import streamlit as st
from io import BytesIO
from email.mime.text import MIMEText
from playsound import playsound
from pylsl import StreamInlet, resolve_stream
from ns015_shap_live import shap_explain_live
import matplotlib.pyplot as plt
from scipy.signal import welch
from ripser import ripser
from datetime import datetime


# === TIME-BASED SESSION DIRECTORY
timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_DIR = os.path.join("logs", f"session_{timestamp_str}")
os.makedirs(LOG_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(LOG_DIR, f"ns014_predictions_{timestamp_str}.csv")
JSON_LOG = os.path.join(LOG_DIR, "predictions_log.json")
GIF_FILE = os.path.join(LOG_DIR, "prediction_live.gif")
SHAP_IMG = os.path.join(LOG_DIR, "shap_live_frame.png")  # si utilisé
ALERT_SOUND = "assets/alert_sound.mp3"  # reste en dur pour le moment

# === CONFIGURATION
MODEL_PATH = "ns013_results/model.pkl"
EEG_PATH = "eeg_samples/subject_01.set"
RESULTS_CSV = "ns014_predictions.csv"
WINDOW_SIZE = 512
FS = 128
CHANNEL = "Cz"
PREDICTION_INTERVAL = 2  # sec
THRESHOLD = 0.85  # Seuil d’alerte cognitive (proba classe 1)

# === Chargement configuration email
def load_notifier_config(path="notifier_config.json"):
    if not os.path.exists(path):
        st.error("❌ notifier_config.json introuvable.")
        return None
    with open(path) as f:
        return json.load(f)

# === QR code
def generate_qr_for_zip(zip_path):
    placeholder_url = f"https://neurosolve.local/sessions/{os.path.basename(zip_path)}"
    qr = qrcode.make(placeholder_url)
    buffer = BytesIO()
    qr.save(buffer, format="PNG")
    return buffer.getvalue()

# === Envoi Email ==
def send_email_alert(summary, config, zip_path=None):
    from email.mime.multipart import MIMEMultipart
    from email.mime.image import MIMEImage

    body = f"""
    Une session EEG a déclenché une alerte cognitive.  
    Session : {summary['session_folder']}  
    Frames : {summary['nb_frames']}  
    Alertes : {summary['nb_alerts']} ({summary['alert_rate']*100:.1f}%)  
    Durée : {summary['duration_sec']}s  
    Prob. moyenne classe 1 : {summary['avg_prob_class_1']}  
    """

    msg = MIMEMultipart()
    msg['From'] = config["sender_email"]
    msg['To'] = ", ".join(config["recipients"])
    msg['Subject'] = f"[NeuroSolve] Alerte cognitive - {summary['session_folder']}"
    msg.attach(MIMEText(body, "plain"))

    # QR si dispo
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
        print(f"❌ Erreur email : {e}")

    if summary_data["nb_alerts"] > 0:
        send_email_alert(summary_data, config, zip_path)

# === Fonctions de features (ondelettes + TDA)
def extract_wavelet_features(data, wavelet='db4', level=4):
    import pywt
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeff_arr = pywt.coeffs_to_array(coeffs)[0]
    return coeff_arr.flatten()

def extract_tda_features(data):
    traj = np.array([data[i:i+3] for i in range(len(data) - 3)])
    dgms = ripser(traj)['dgms']
    b1 = dgms[1]
    if len(b1) > 0:
        persistence = max([death - birth for birth, death in b1])
        birth = b1[0][0]
        death = b1[0][1]
    else:
        persistence, birth, death = 0, 0, 0
    return np.array([persistence, birth, death])

# === EEG Temps réel via LSL
def get_lsl_segment(window_size=512, channels=1):
    print("🔍 Recherche d'un flux LSL EEG...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    print("✅ Flux EEG connecté !")

    eeg_buffer = []
    while len(eeg_buffer) < window_size:
        sample, _ = inlet.pull_sample()
        eeg_buffer.append(sample[:channels])
    return np.array(eeg_buffer).T[0]  # (512,)


# === Chargement du modèle
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# === Prédiction sur segment EEG
def predict_segment(model, scaler, segment):
    tda_feat = extract_tda_features(segment)
    wavelet_feat = extract_wavelet_features(segment)
    full_feat = np.concatenate([tda_feat, wavelet_feat])
    full_feat = scaler.transform([full_feat])
    pred = model.predict(full_feat)[0]
    prob = model.predict_proba(full_feat)[0]
    return pred, prob

# === Lecture EEG en boucle
def live_loop(mode="lsl", gui=False, save_gif=False):
    print("🧠 Lancement prédiction EEG live | Mode :", mode)

    # Chargement modèle et scaler
    model_data = np.load("ns013_results/model_scaler.npz", allow_pickle=True)
    scaler = model_data["scaler"][()]
    model = load_model(MODEL_PATH)

    predictions = []
    gif_frames = []

    for i in range(20):  # Nombre d'itérations de prédiction
        if mode == "lsl":
            segment = get_lsl_segment(WINDOW_SIZE, channels=1)
        else:
            segment = np.random.normal(0, 1, size=WINDOW_SIZE)  # fallback

        t = round(i * PREDICTION_INTERVAL, 2)
        pred, prob = predict_segment(model, scaler, segment)

        print(f"[{t}s] ▶️ Classe prédite : {pred} | Proba : {np.round(prob, 3)}")
        timestamp = time.time()
        subject_id = "subject_01"  # ou dynamiquement depuis EEG si possible
        
        predictions.append({
            "subject": subject_id,
            "iteration": i,
            "time_sec": t,
            "timestamp": timestamp,
            "prediction": int(pred),
            "prob_class_0": float(prob[0]),
            "prob_class_1": float(prob[1]) if len(prob) > 1 else 0
        })
        

        if i % 5 == 0:
            shap_explain_live(segment, model, scaler)

        
        if prob[1] > THRESHOLD:
            print("🔔⚠️ Alerte cognitive détectée ! Probabilité classe 1 :", round(prob[1], 3))
            predictions[-1]["alert"] = True
            playsound("assets/alert_sound.mp3")
        else:
            predictions[-1]["alert"] = False


        # Streamlit ou terminal
        if gui:
            import streamlit as st
            st.line_chart(predictions)

        if predictions[-1]["alert"]:
            st.warning("⚠️ Alerte cognitive déclenchée !")
        else:
            st.success("✅ Activité normale")


        # GIF generation
        if save_gif:
            plt.figure(figsize=(6, 3))
            plt.plot(segment)
            plt.title(f"{t}s - Prédiction: {pred}")
            plt.tight_layout()
            fname = f"frame_{i:03}.png"
            plt.savefig(fname)
            gif_frames.append(fname)
            plt.close()

        time.sleep(PREDICTION_INTERVAL)

    # Export .csv
    df = pd.DataFrame(predictions)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"📁 Prédictions sauvegardées : {RESULTS_CSV}")

    # Export JSON LOG
    import json
    with open(JSON_LOG, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"🧾 Log JSON sauvegardé : {JSON_LOG}")

    # Export GIF output
    if save_gif:
        import imageio
        images = [imageio.imread(f) for f in gif_frames]
        imageio.mimsave(GIF_FILE, images, duration=0.6)
        print(f"🎞️ GIF prédictif généré : {GIF_FILE}")
        for f in gif_frames: os.remove(f)

    # === ARCHIVAGE SUMMARY GLOBAL
    summary_data = {
        "session_folder": os.path.basename(LOG_DIR),
        "nb_frames": len(predictions),
        "nb_alerts": sum(1 for p in predictions if p.get("alert")),
        "alert_rate": round(sum(1 for p in predictions if p.get("alert")) / len(predictions), 3),
        "duration_sec": round(predictions[-1]["time_sec"] - predictions[0]["time_sec"], 2) if len(predictions) > 1 else 0,
        "avg_prob_class_1": round(np.mean([p["prob_class_1"] for p in predictions]), 3),
        "timestamp_generated": datetime.datetime.now().isoformat()
    }
    

    # ➕ QR Code joint
    if zip_path:
        import qrcode
        from io import BytesIO
        qr = qrcode.make(f"https://neurosolve.local/sessions/{os.path.basename(zip_path)}")
        qr_bytes = BytesIO()
        qr.save(qr_bytes, format="PNG")
        qr_bytes.seek(0)

        image = MIMEImage(qr_bytes.read(), name="qr_session.png")
        image.add_header("Content-ID", "<qr>")
        msg.attach(image)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDERS["from"], EMAIL_SENDERS["password"])
            server.sendmail(EMAIL_SENDERS["from"], EMAIL_RECIPIENTS, msg.as_string())
        print("📧 Email(s) envoyé(s) avec succès.")
    except Exception as e:
        print(f"❌ Erreur envoi email : {e}")

    if summary_data["nb_alerts"] > 0:
        send_email_alert(summary_data, config, zip_path)

    
    summary_df = pd.DataFrame([summary_data])
    summary_csv_path = "sessions_summary.csv"
    if os.path.exists(summary_csv_path):
        prev = pd.read_csv(summary_csv_path)
        summary_df = pd.concat([prev, summary_df], ignore_index=True)
    summary_df.to_csv(summary_csv_path, index=False)
    print("📦 Résumé global mis à jour automatiquement : sessions_summary.csv")

    # === ZIP AUTOMATIQUE DE LA SESSION
    zip_path = LOG_DIR + ".zip"
    shutil.make_archive(LOG_DIR, 'zip', LOG_DIR)
    print(f"📦 Session zippée automatiquement : {zip_path}")
    return zip_path
        zip_path = live_loop(mode="lsl", gui=True, save_gif=True)



def generate_qr_for_zip(zip_path):
    placeholder_url = f"https://neurosolve.local/sessions/{os.path.basename(zip_path)}"
    qr = qrcode.make(placeholder_url)
    buffer = BytesIO()
    qr.save(buffer, format="PNG")
    return buffer.getvalue()


config = load_notifier_config()

if st.button("🧠 Lancer la Prédiction EEG Live"):
    with st.spinner("Acquisition en cours..."):
        zip_path = live_loop(mode="lsl", gui=True, save_gif=True)  # tu dois retourner zip_path

    if zip_path:
        st.success("✅ Session terminée avec succès.")
        st.markdown("### 🔗 QR Code de la session")
        st.image(generate_qr_for_zip(zip_path), width=220)

