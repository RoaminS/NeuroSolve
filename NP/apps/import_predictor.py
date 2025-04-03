#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
import_predictor.py

✅ Import EEG (.set, .edf, .bdf, .h5, .json)
✅ Extraction TDA + ondelettes
✅ Prédiction avec modèle NS013 (pkl ou AdFormer .h5)
✅ Export CSV / Résumé / Alertes
✅ QR code + SHAP

Auteur : Kocupyr Romain
Dev    : multi_gpt_api
"""

import os
import json
import pickle
import shutil
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from io import BytesIO
import qrcode
import mne
import pywt
from ripser import ripser
from ns015_shap_live import shap_explain_live
from sklearn.preprocessing import StandardScaler

# === CONFIG
THRESHOLD = 0.85
LOG_DIR = os.path.join("logs", f"import_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
os.makedirs(LOG_DIR, exist_ok=True)
MODEL_DIR = "ns013_results"

# === QR Code
def generate_qr_for_zip(path):
    url = f"https://neurosolve.local/sessions/{os.path.basename(path)}"
    qr = qrcode.make(url)
    buf = BytesIO()
    qr.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

# === Charge EEG depuis tous les formats
def load_eeg_any(file, topn=20):
    ext = os.path.splitext(file.name)[1].lower()
    X, subj = [], []
    if ext == ".h5":
        import h5py
        with h5py.File(file, 'r') as f:
            X = f["X"][:topn]
            subj = [str(s) for s in f["subj"][:topn]]
    elif ext == ".json":
        content = json.load(file)
        for entry in content[:topn]:
            X.append(np.array(entry["vector"]))
            subj.append(entry.get("subject", "unknown"))
    elif ext in [".set", ".edf", ".bdf"]:
        raw = None
        if ext == ".set":
            raw = mne.io.read_raw_eeglab(file, preload=True)
        elif ext == ".edf":
            raw = mne.io.read_raw_edf(file, preload=True)
        elif ext == ".bdf":
            raw = mne.io.read_raw_bdf(file, preload=True)
        raw.pick_types(eeg=True).filter(0.5, 45).resample(128)
        data = raw.get_data()
        for i in range(0, data.shape[1] - 512, 512):
            vec = data[:, i:i+512].T.flatten()
            X.append(vec)
            subj.append(file.name)
            if len(X) >= topn: break
    else:
        st.error("❌ Format non supporté.")
        return [], []

    return np.array(X), subj

# === Features
def extract_features(x):
    traj = np.array([x[i:i+3] for i in range(len(x)-3)])
    b1 = ripser(traj)['dgms'][1]
    if len(b1):
        persistence = max(d-b for b,d in b1)
        tda = [persistence, b1[0][0], b1[0][1]]
    else:
        tda = [0, 0, 0]

    coeffs = pywt.wavedec(x, 'db4', level=4)
    wav = pywt.coeffs_to_array(coeffs)[0].flatten()
    return np.concatenate([tda, wav])

# === Modèle
def load_model(use_adformer=False):
    if use_adformer:
        model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "model_adformer.h5"))
        scaler = np.load(os.path.join(MODEL_DIR, "model_scaler_adformer.npz"), allow_pickle=True)["scaler"][()]
    else:
        model = pickle.load(open(os.path.join(MODEL_DIR, "model.pkl"), "rb"))
        scaler = np.load(os.path.join(MODEL_DIR, "model_scaler.npz"), allow_pickle=True)["scaler"][()]
    return model, scaler

# === Interface Streamlit
st.set_page_config(page_title="🧠 Importateur EEG & Prédicteur")
st.title("🧠 NeuroSolve – Prédictions depuis EEG importé")

uploaded = st.file_uploader("📂 Importer un fichier EEG (.set, .edf, .bdf, .h5, .json)", type=["set", "edf", "bdf", "h5", "json"])
model_type = st.selectbox("🧠 Choix du modèle :", ["RandomForest (.pkl)", "AdFormer (.h5)"])
use_adformer = model_type == "AdFormer (.h5)"

if uploaded and st.button("🚀 Lancer les prédictions"):
    X, subjects = load_eeg_any(uploaded)
    if len(X) == 0:
        st.warning("Aucune donnée valide.")
        st.stop()

    model, scaler = load_model(use_adformer)
    predictions = []

    for i, x in enumerate(X):
        features = extract_features(x)
        features_scaled = scaler.transform([features])
        if use_adformer:
            proba = model.predict(features_scaled)[0]
            pred = int(np.argmax(proba))
        else:
            pred = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0]

        alert = proba[1] > THRESHOLD
        predictions.append({
            "i": i,
            "subject": subjects[i],
            "prediction": int(pred),
            "prob_class_0": float(proba[0]),
            "prob_class_1": float(proba[1]),
            "alert": alert
        })

        if i % 5 == 0:
            shap_explain_live(x, model, scaler)

    df = pd.DataFrame(predictions)
    st.dataframe(df)

    df.to_csv(os.path.join(LOG_DIR, "import_predictions.csv"), index=False)
    json.dump(predictions, open(os.path.join(LOG_DIR, "import_predictions.json"), "w"), indent=2)

    st.success("✅ Prédictions terminées.")
    st.metric("Total vecteurs", len(predictions))
    st.metric("Nb alertes", df["alert"].sum())

    zip_path = shutil.make_archive(LOG_DIR, "zip", LOG_DIR)
    st.download_button("⬇️ Télécharger session ZIP", open(zip_path, "rb"), file_name=os.path.basename(zip_path), mime="application/zip")
    st.image(generate_qr_for_zip(zip_path), width=200, caption="🔗 QR session EEG importée")
