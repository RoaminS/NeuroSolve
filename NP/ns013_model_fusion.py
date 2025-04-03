#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ns013_model_fusion.py

✅ Fusion TDA (topologie) + ondelettes pour classification EEG
✅ Chargement des features depuis NS012
✅ Entraînement de modèle ML (RandomForest / XGBoost)
✅ Évaluation complète (acc, f1, AUC, matrice confusion)
✅ Export des résultats & visualisation

Auteur : Kocupyr Romain
Dev    : multi_gpt_api, Grok3
Licence : CC BY-NC-SA 4.0
"""

import os
import json
import shap
import pickle
import torch
import webbrowser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler

# === CONFIG
CYCLES_CSV = "ns012_results/cycles_detected.csv"
WAVELETS_NPY = "ns012_results/wavelet_features.npy"
LABELS_CSV = "labels.csv"  # Doit contenir : subject_id, class
OUTPUT_DIR = "ns013_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Chargement des données
def load_and_merge():
    cycles = pd.read_csv(CYCLES_CSV)
    labels = pd.read_csv(LABELS_CSV)

    # Fusion sur subject_id
    df = pd.merge(cycles, labels, how="inner", on="subject_id")

    # Ajout des ondelettes si dispo
    if os.path.exists(WAVELETS_NPY):
        wavelets = np.load(WAVELETS_NPY, allow_pickle=True).item()
        df["wavelet_feat"] = df["subject_id"].map(wavelets)

        # Supprime les lignes sans ondelettes
        df = df[df["wavelet_feat"].notnull()]
        wavelet_matrix = np.vstack(df["wavelet_feat"].values)
    else:
        wavelet_matrix = None

    return df, wavelet_matrix

# === Modèle ML
def run_model(df, wavelets=None):
    print("✅ Fusion des features TDA + ondelettes...")

    X_tda = df[["persistence", "birth", "death"]].values
    y = df["class"].values

    if wavelets is not None:
        X = np.hstack([X_tda, wavelets])
    else:
        X = X_tda

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=150, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None
 
    # === Export modèle pkl
    with open(os.path.join(OUTPUT_DIR, "model.pkl"), "wb") as f:
        pickle.dump(clf, f)
    print(f"💾 Modèle sauvegardé : model.pkl")
    np.savez(os.path.join(OUTPUT_DIR, "model_scaler.npz"), scaler=scaler)
    print("💾 Scaler sauvegardé : model_scaler.npz")

    # === PyTorch MLP
    class SimpleMLP(nn.Module):
        def __init__(self, input_size, hidden=64, output_size=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden),
                nn.ReLU(),
                nn.Linear(hidden, output_size)
            )

        def forward(self, x):
            return self.net(x)

    model = SimpleMLP(input_size=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = loss_fn(output, y_tensor)
        loss.backward()
        optimizer.step()

    # === Sauvegarde versionnée du modèle AdFormer (PyTorch)
    model_dir = os.path.join(OUTPUT_DIR, f"model_adformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(model, os.path.join(model_dir, "model_adformer.pth"))
    np.savez(os.path.join(model_dir, "model_scaler_adformer.npz"), scaler=scaler)
    print("💾 Scaler sauvegardé : model_scaler_adformer.npz")
    print(f"💾 Modèle AdFormer sauvegardé dans : {model_dir}")

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=150, random_state=0),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier()
    }


    scores = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append((name, score))
        print(f"✅ {name} Accuracy: {score:.3f}")

    # Plot benchmark
    model_names, accs = zip(*scores)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(model_names), y=list(accs), palette="Set2")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Benchmark modèles ML (fusion EEG)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_benchmark.png"))
    print("📊 Benchmark modèles ML sauvegardé")


    print("\n🧠 Résultats du modèle :")
    print(classification_report(y_test, y_pred))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matrice de confusion")
    plt.xlabel("Prédiction")
    plt.ylabel("Vérité terrain")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    print("📊 Matrice de confusion sauvegardée")

    # ROC
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], '--')
        plt.title("Courbe ROC")
        plt.xlabel("Faux positifs")
        plt.ylabel("Vrais positifs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
        print("📈 ROC Curve sauvegardée")

    # === XAI avec SHAP
    try:
        explainer = shap.Explainer(clf, X_train)
        shap_values = explainer(X_test)
        plt.figure()
        shap.plots.beeswarm(shap_values, show=False)
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "shap_beeswarm.png"))
        print("🔍 SHAP beeswarm plot sauvegardé")
    except Exception as e:
        print(f"⚠️ SHAP non exécuté : {e}")


# === MAIN
def main():
    print("🧬 NS013 — Fusion TDA + ondelettes pour classification EEG")
    df, wavelets = load_and_merge()
    run_model(df, wavelets)

    # === HTML DASHBOARD
    html_path = os.path.join(OUTPUT_DIR, "ns013_dashboard.html")
    with open(html_path, "w") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html><head>
        <meta charset="UTF-8">
        <title>NS013 - EEG Model Fusion</title>
        <style>
            body {{ font-family: Arial; padding: 30px; background: #fafafa; }}
            img {{ max-width: 600px; margin: 20px 0; }}
        </style>
        </head><body>
        <h1>🧠 NS013 - Modèle EEG Fusion (TDA + Ondelette)</h1>
        <p>Comparaison de modèles + interprétabilité XAI + visualisation</p>

        <h2>📊 Matrice de confusion</h2>
        <img src="confusion_matrix.png" alt="Confusion Matrix">

        <h2>📈 Courbe ROC</h2>
        <img src="roc_curve.png" alt="ROC">

        <h2>🤖 Benchmark modèles ML</h2>
        <img src="model_benchmark.png" alt="Benchmark">

        <h2>🔍 SHAP (Interprétabilité)</h2>
        <img src="shap_beeswarm.png" alt="SHAP">

        <p style="margin-top:40px;">Fichier généré automatiquement par <strong>ns013_model_fusion.py</strong></p>
        </body></html>
        """)
    
    webbrowser.open(os.path.abspath(html_path))
    print(f"🌐 Dashboard ouvert automatiquement : {html_path}")


if __name__ == "__main__":
    main()
