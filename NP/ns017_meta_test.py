'''
Ce qu’il te faut pour le lancer :

1. 📂 Place tes fichiers .h5 dans un dossier :

real_eeg_data_h5/
├── patient_01.h5
├── patient_02.h5
├── ...

2. ▶️ Lance :

python ns099_meta_test.py
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ns099_meta_test.py

✅ Teste toutes les stratégies de résolution NP sur des EEG réels (.h5)
✅ Mesure le taux de réussite, le temps, le nombre de steps
✅ Comparaison brute vs guided vs random
✅ Exports : .csv, .json, heatmap.png, dashboard.html

Auteur : Kocupyr Romain
Dev    : multi_gpt_api
Licence : CC BY-NC-SA 4.0
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from NP.ns001_neuro_np_solver import subset_sum_solver, SolverConfig
import h5py

# === CONFIG
DATA_DIR = "real_eeg_data_h5"
OUTPUT_DIR = "ns099_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STRATEGIES = ["exhaustive", "guided", "random"]
MAX_DEPTH = 5000
TARGET_OFFSET = 10

# === Chargement EEG .h5
def load_eeg_h5(file_path, topn=5):
    with h5py.File(file_path, 'r') as f:
        X = f["X"][:]
        subj = f["subj"][:]
        y = f["y"][:] if "y" in f else np.array([-1] * len(X))
    return X[:topn], subj[:topn], y[:topn]

# === Génère un ensemble d'entiers à partir du vecteur EEG
def eeg_vector_to_problem(vec, n=19):
    base_set = np.abs(vec[:n]).astype(int).tolist()
    target = sum(base_set[:3]) + TARGET_OFFSET
    return base_set, target

# === Benchmark
def test_strategies_on_eeg(X, subj_ids, y):
    logs = []

    for i, vec in enumerate(X):
        base_set, target = eeg_vector_to_problem(vec)
        subj = subj_ids[i].decode() if isinstance(subj_ids[i], bytes) else str(subj_ids[i])

        print(f"\n🧠 EEG segment #{i+1} — Sujet : {subj} | Target = {target}")

        for strat in STRATEGIES:
            config = SolverConfig(strategy=strat, log_steps=False, max_depth=MAX_DEPTH)
            t0 = time.time()
            result = subset_sum_solver(base_set, target, config=config)
            duration = time.time() - t0

            log = {
                "subject": subj,
                "segment": i,
                "strategy": strat,
                "target": target,
                "found": len(result["solutions"]) > 0,
                "steps": result["steps"],
                "duration_sec": round(duration, 4),
            }
            logs.append(log)
            print(f"✅ {strat:<10} | Found: {log['found']} | Steps: {log['steps']} | Time: {log['duration_sec']}s")
    return logs

# === Visualisation heatmap
def plot_heatmap(df, output_path):
    pivot = df.pivot_table(index="subject", columns="strategy", values="duration_sec", aggfunc="mean")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("⏱️ Temps moyen par stratégie et sujet")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"📈 Heatmap sauvegardée : {output_path}")

# === Dashboard HTML
def generate_dashboard():
    with open(os.path.join(OUTPUT_DIR, "meta_dashboard.html"), "w") as f:
        f.write(f"""
        <html><head>
        <meta charset="utf-8">
        <title>NS099 — Meta Benchmark EEG</title>
        <style>
            body {{ font-family: sans-serif; padding: 20px; }}
            img {{ max-width: 600px; margin-bottom: 20px; }}
        </style>
        </head><body>
        <h1>🧠 NS099 — Benchmark EEG Réels sur Problèmes NP</h1>
        <p>Comparaison brute / guided / random</p>
        <h2>⏱️ Temps moyen par sujet & stratégie</h2>
        <img src="heatmap_duration.png">
        <p><a href="meta_results.csv">📁 Résultats CSV</a> — <a href="meta_results.json">📄 Résultats JSON</a></p>
        </body></html>
        """)
    print("🌐 Dashboard généré : meta_dashboard.html")

# === MAIN
def main():
    all_logs = []

    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".h5"):
            print(f"📂 Traitement du fichier : {fname}")
            X, subj, y = load_eeg_h5(os.path.join(DATA_DIR, fname))
            logs = test_strategies_on_eeg(X, subj, y)
            all_logs.extend(logs)

    df = pd.DataFrame(all_logs)
    df.to_csv(os.path.join(OUTPUT_DIR, "meta_results.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "meta_results.json"), "w") as f:
        json.dump(all_logs, f, indent=2)

    print("\n✅ Résultats sauvegardés dans :", OUTPUT_DIR)

    plot_heatmap(df, os.path.join(OUTPUT_DIR, "heatmap_duration.png"))
    generate_dashboard()

if __name__ == "__main__":
    main()

