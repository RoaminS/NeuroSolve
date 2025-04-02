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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# === MAIN
def main():
    print("🧬 NS013 — Fusion TDA + ondelettes pour classification EEG")
    df, wavelets = load_and_merge()
    run_model(df, wavelets)

if __name__ == "__main__":
    main()
