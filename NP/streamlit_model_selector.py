"""
streamlit_model_selector.py

Licence : Creative Commons BY-NC-SA 4.0

Auteurs : 
    - Kocupyr Romain (créateur)
Dev:
    - Multi_gpt_api 
"""

import os
import torch
import pickle
import streamlit as st
import numpy as np
import joblib

def select_and_load_model():
    st.markdown("### 🎯 Sélection du modèle NeuroSolve")

    base_path = "ns013_results"
    perso_path = os.path.join(base_path, "model_perso")

    # Regroupe tous les modèles .pkl et .pth
    all_models = []

    for path in [base_path, perso_path]:
        for f in os.listdir(path):
            if f.endswith((".pkl", ".pth")):
                all_models.append((f, os.path.join(path, f)))

    if not all_models:
        st.warning("⚠️ Aucun modèle trouvé dans ns013_results/")
        st.stop()

    # Sélection utilisateur
    model_names = [f for f, _ in all_models]
    selected_name = st.selectbox("📂 Modèles disponibles :", model_names)
    selected_path = [p for f, p in all_models if f == selected_name][0]

    # Détection du type
    is_adformer = selected_path.endswith(".pth")

    # Chargement du modèle 
    if is_adformer:
        model = torch.load(selected_path, map_location=torch.device("cpu"))
        model.eval()
    else:
        with open(selected_path, "rb") as f:
            model = pickle.load(f)

    st.success(f"✅ Modèle chargé : `{selected_name}`")

    # Scaler associé
    scaler_path = selected_path.replace(".pth", "_scaler.pkl").replace(".pkl", "_scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        st.info(f"🔧 Scaler trouvé : {os.path.basename(scaler_path)}")
    else:
        scaler = None
        st.warning("⚠️ Aucun scaler associé trouvé.")

    return model, scaler, selected_name
