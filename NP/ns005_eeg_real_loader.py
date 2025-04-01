#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ns005_eeg_real_loader.py

🧠 Chargement de vecteurs EEG réels pour intégration dans NeuroSolv
Auteur : Kocupyr Romain
Dev    : multi_gpt_api, Grok3
Licence : Creative Commons BY-NC-SA 4.0
Commande: python ns005_eeg_real_loader.py --h5 <eeg_data_....h5> --topn 10


"""

import os
import numpy as np
import h5py
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from antropy import perm_entropy, sample_entropy

# === PARAMÈTRES
parser = argparse.ArgumentParser()
parser.add_argument("--h5", required=True, help="Fichier .h5 EEG (ADFormer)")
parser.add_argument("--output", default="eeg_vectors_real.csv", help="Fichier CSV de sortie")
parser.add_argument("--topn", type=int, default=5, help="Nombre de vecteurs max à exporter")
args = parser.parse_args()

# === CONFIG
RAW_SIZE = 9728
FEATURE_SIZE = 267
TOTAL_SIZE = RAW_SIZE + FEATURE_SIZE
SAMPLES = 512
NUM_ELECTRODES = 19

# === LECTURE HDF5
print(f"📂 Lecture : {args.h5}")
with h5py.File(args.h5, "r") as f:
    X = f["X"][:]
    subj = f["subj"][:]
    y = f["y"][:] if "y" in f else np.array([-1]*len(X))

print(f"✅ {len(X)} segments EEG chargés")

# === SCALER (normalisation EEG brute)
scaler = StandardScaler().fit(X)

# === EXTRACTION
vectors = []
for i in range(min(len(X), args.topn)):
    vec = scaler.transform([X[i]])[0]
    raw = vec[:RAW_SIZE].reshape(SAMPLES, NUM_ELECTRODES)

    # === SCORE EEG simple (entropy + variance)
    pe = perm_entropy(raw.ravel(), order=3, normalize=True)
    se = sample_entropy(raw.ravel(), order=2)
    var = np.var(raw)
    score = pe * 0.6 + se * 0.3 + var * 0.1

    vectors.append({
        "id": f"sample_{i}",
        "subject": subj[i].decode() if isinstance(subj[i], bytes) else str(subj[i]),
        "score": round(score, 4),
        "label": int(y[i]) if y[i] >= 0 else "N/A",
        "vector": vec.tolist()
    })

# === SAUVEGARDE CSV
df = pd.DataFrame([{
    "id": v["id"],
    "subject": v["subject"],
    "score": v["score"],
    "label": v["label"]
} for v in vectors])

vec_df = pd.DataFrame([v["vector"] for v in vectors])
full_df = pd.concat([df, vec_df], axis=1)
full_df.to_csv(args.output, index=False)

print(f"📁 EEG vectors exportés dans : {args.output}")
