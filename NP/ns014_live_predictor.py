#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ns014_live_predictor.py

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
import mne
import time
from pylsl import StreamInlet, resolve_stream
from ns032_shap_live import shap_explain_live
import matplotlib.pyplot as plt
from scipy.signal import welch
from ripser import ripser

# === CONFIGURATION
MODEL_PATH = "ns013_results/model.pkl"
EEG_PATH = "eeg_samples/subject_01.set"
RESULTS_CSV = "ns020_predictions.csv"
WINDOW_SIZE = 512
FS = 128
CHANNEL = "Cz"
PREDICTION_INTERVAL = 2  # sec

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
        predictions.append([t, pred, *prob])

        
        shap_explain_live(segment, model, scaler)


        # Streamlit ou terminal
        if gui:
            import streamlit as st
            st.line_chart(predictions)

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
    df = pd.DataFrame(predictions, columns=["time_sec", "prediction", "prob0", "prob1"])
    df.to_csv(RESULTS_CSV, index=False)
    print(f"📁 Prédictions sauvegardées : {RESULTS_CSV}")

    # GIF output
    if save_gif:
        import imageio
        images = [imageio.imread(f) for f in gif_frames]
        imageio.mimsave("prediction_live.gif", images, duration=0.6)
        print("🎞️ GIF prédictif généré : prediction_live.gif")
        for f in gif_frames: os.remove(f)

# === MAIN

if __name__ == "__main__":
    live_loop(mode="lsl", gui=False, save_gif=True)  # ← ou Streamlit avec gui=True

