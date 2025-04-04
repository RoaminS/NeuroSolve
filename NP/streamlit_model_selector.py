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

def select_and_load_model(base_path="ns013_results"):
    st.markdown("### 🧠 Sélection d’un modèle NeuroSolve")

    # Recherche récursive
    model_paths = []
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.endswith((".pth", ".pkl")):
                model_paths.append(os.path.join(root, f))

    if not model_paths:
        st.error("❌ Aucun modèle trouvé dans ns013_results/")
        st.stop()

    model_labels = [os.path.relpath(p, base_path) for p in model_paths]
    selected_label = st.selectbox("📂 Modèle disponible :", model_labels)
    model_path = os.path.join(base_path, selected_label)

    is_adformer = model_path.endswith(".pth")

    # Chargement du modèle
    if is_adformer:
        model = torch.load(model_path, map_location=torch.device("cpu"))
        model.eval()
        model_type = "AdFormer"
    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        model_type = "RandomForest"

    st.success(f"✅ Modèle chargé : {selected_label}")

    # === Chargement du scaler associé
    scaler_path = model_path.replace(".pth", "_scaler.pkl").replace(".pkl", "_scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        st.info(f"🔧 Scaler trouvé : {os.path.basename(scaler_path)}")
    else:
        scaler = None
        st.warning("⚠️ Aucun scaler trouvé associé à ce modèle.")

    return model, scaler, model_type, model_path, selected_label
