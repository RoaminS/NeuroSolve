#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ns015_shap_live.py

✅ Explication XAI temps réel des prédictions EEG
✅ Utilisation de SHAP (Shapley values)
✅ Visualisation dynamique des features contributifs
✅ Intégrable à ns014_live_predictor.py

Auteur  : Kocupyr Romain
Dev     : multi_gpt_api, NeuroSolv Crew
Licence : CC BY-NC-SA 4.0
"""

import os
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
from ripser import ripser
from scipy.signal import welch
import pywt

# === CONFIG
MODEL_PATH = "ns013_results/model.pkl"
SCALER_PATH = "ns013_results/model_scaler.npz"
WINDOW_SIZE = 512
CHANNELS = 1

# === Feature extraction
def extract_wavelet_features(data, wavelet='db4', level=4):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return pywt.coeffs_to_array(coeffs)[0].flatten()

def extract_tda_features(data):
    traj = np.array([data[i:i+3] for i in range(len(data)-3)])
    dgms = ripser(traj)['dgms']
    b1 = dgms[1]
    if len(b1) > 0:
        persistence = max([d - b for b, d in b1])
        birth, death = b1[0][0], b1[0][1]
    else:
        persistence, birth, death = 0, 0, 0
    return np.array([persistence, birth, death])

# === Chargement modèle + scaler
def load_model_and_scaler():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    scaler = np.load(SCALER_PATH, allow_pickle=True)["scaler"].item()
    return model, scaler

# === SHAP LIVE EXPLANATION
def shap_explain_live(segment, model, scaler):
    tda_feat = extract_tda_features(segment)
    wavelet_feat = extract_wavelet_features(segment)
    full_feat = np.concatenate([tda_feat, wavelet_feat])
    normed = scaler.transform([full_feat])

    explainer = shap.Explainer(model, masker=shap.maskers.Independent(normed))
    shap_values = explainer(normed)

    # Plot live barplot
    plt.figure(figsize=(10, 4))
    shap.plots.bar(shap_values[0], show=False)
    plt.title("🔍 SHAP Explication - Prédiction EEG Live")
    plt.tight_layout()
    plt.savefig("shap_live_frame.png")
    plt.show()
    print("✅ SHAP visuel sauvegardé : shap_live_frame.png")

# === Démo
def main():
    print("🧠 Lancement explication SHAP temps réel")

    model, scaler = load_model_and_scaler()
    # Simule un segment EEG (à remplacer par EEG réel dans ns014)
    segment = np.random.normal(0, 1, size=WINDOW_SIZE)

    shap_explain_live(segment, model, scaler)

if __name__ == "__main__":
    main()
