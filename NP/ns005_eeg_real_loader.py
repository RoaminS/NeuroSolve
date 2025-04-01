#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ns005_eeg_real_loader.py

— EEG ↔ NP Solver Bridge

🧠 Charge des vecteurs EEG réels, les transforme, et :
    - Exporte en JSON pour NeuroNP Solver (ns001)
    - Génère un graphe EEG pour Visualizer (ns003)
    - Peut alimenter un solveur NP en temps réel (simulation EEG-loop)

Auteur : Kocupyr Romain
Dev    : multi_gpt_api, Grok3
Licence : Creative Commons BY-NC-SA 4.0

Commande: python ns005_eeg_real_loader.py --h5 eeg_data_alzheimer_99_id.h5 \
  --topn 10 --json --graph --live

"""

import os
import argparse
import numpy as np
import pandas as pd
import h5py
import json
from antropy import perm_entropy, sample_entropy
from sklearn.preprocessing import StandardScaler
import networkx as nx

# === ARGS
parser = argparse.ArgumentParser()
parser.add_argument("--h5", required=True, help="Fichier EEG HDF5")
parser.add_argument("--outdir", default="ns005_output", help="Répertoire de sortie")
parser.add_argument("--topn", type=int, default=5, help="Nb de vecteurs EEG à exporter")
parser.add_argument("--json", action="store_true", help="Exporter JSON compatible NS001")
parser.add_argument("--graph", action="store_true", help="Générer graphe EEG pour ns003")
parser.add_argument("--live", action="store_true", help="Simuler feed live vers NS001 solver")
args = parser.parse_args()

# === CONFIG
SAMPLES = 512
NUM_ELECTRODES = 19
RAW_SIZE = SAMPLES * NUM_ELECTRODES
FEATURE_SIZE = 267
TOTAL_SIZE = RAW_SIZE + FEATURE_SIZE
os.makedirs(args.outdir, exist_ok=True)

# === LECTURE EEG
print(f"📂 Lecture : {args.h5}")
with h5py.File(args.h5, 'r') as f:
    X = f["X"][:]
    subj = f["subj"][:]
    y = f["y"][:] if "y" in f else np.array([-1]*len(X))
print(f"✅ {len(X)} EEG segments chargés")

scaler = StandardScaler().fit(X)

# === STRUCTURE EEG 10-20 (simple)
electrode_labels = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6", "O1", "O2"
]

# === EXPORT DATA
json_ready = []
graphs = []

for i in range(min(args.topn, len(X))):
    vec = scaler.transform([X[i]])[0]
    raw = vec[:RAW_SIZE].reshape(SAMPLES, NUM_ELECTRODES)
    subj_id = subj[i].decode() if isinstance(subj[i], bytes) else str(subj[i])

    pe = perm_entropy(raw.ravel(), order=3, normalize=True)
    se = sample_entropy(raw.ravel(), order=2)
    var = np.var(raw)
    score = round(pe * 0.6 + se * 0.3 + var * 0.1, 4)

    eeg_entry = {
        "id": f"eeg_{i}",
        "subject": subj_id,
        "score": score,
        "label": int(y[i]) if y[i] >= 0 else None,
        "vector": vec.tolist()
    }

    # === JSON export
    if args.json:
        json_ready.append(eeg_entry)

    # === GRAPHE EEG
    if args.graph:
        corr = np.corrcoef(raw.T)
        G = nx.Graph()
        for e_idx, label in enumerate(electrode_labels):
            G.add_node(label)
        for i1 in range(NUM_ELECTRODES):
            for i2 in range(i1 + 1, NUM_ELECTRODES):
                if corr[i1, i2] > 0.5:
                    G.add_edge(electrode_labels[i1], electrode_labels[i2], weight=round(corr[i1, i2], 2))

        gml_path = os.path.join(args.outdir, f"eeg_graph_{i:03d}.graphml")
        nx.write_graphml(G, gml_path)
        graphs.append(gml_path)
        print(f"🌐 Graphe EEG #{i} exporté : {gml_path}")

    # === LIVE SIMULATION ?
    if args.live:
        print(f"⚡ EEG LIVE → Feed vers ns001 (simulé)")
        print(f"🧠 EEG[{i}] | Subject={subj_id} | Score={score}")
        print(f"➡️ vector[:5] = {vec[:5].tolist()}")
        print("🧩 Envoi vers solveur : [simulé ↪️ NeuroNP]")

# === SAUVEGARDE JSON ?
if args.json:
    json_path = os.path.join(args.outdir, "eeg_vectors_ns001.json")
    with open(json_path, "w") as f:
        json.dump(json_ready, f, indent=2)
    print(f"📁 JSON export NS001 : {json_path}")

print("✅ Fin du loader EEG → NeuroSolv Bridge")
