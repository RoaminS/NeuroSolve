#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ns999_benchmark.py

✅ Comparaison brute vs guided vs random pour EEG NP Solver
✅ Mesure du temps, des solutions trouvées, du taux de succès
✅ Export .csv pour analyse comparative

Auteur : Kocupyr Romain
Dev    : multi_gpt_api, Grok3
Licence : CC BY-NC-SA 4.0
"""

import os
import time
import json
import numpy as np
import pandas as pd
from math import pi
import seaborn as sns
import shutil, webbrowser
import matplotlib.pyplot as plt
from ns001_neuro_np_solver import subset_sum_solver, SolverConfig

# === CONFIG
EEG_JSON = "ns005_output/eeg_vectors_ns001.json"
RESULTS_DIR = "benchmark_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(RESULTS_DIR, "benchmark_results.csv")

TARGET_OFFSET = 10
MAX_DEPTH = 5000
STRATEGIES = ["brute", "guided", "random"]

# === Charger les vecteurs EEG
def load_vectors(path, topn=10):
    with open(path) as f:
        data = json.load(f)
    return data[:topn]

# === Guided : génère des poids EEG simulés
def generate_guided_weights(vec):
    n = len(vec)
    weights = np.array(vec)
    weights = np.abs(weights) + 1e-3
    return weights / weights.sum()

# === Benchmark d'une stratégie
def run_benchmark(numbers, target, strategy):
    if strategy == "brute":
        config = SolverConfig(strategy="exhaustive", log_steps=False, max_depth=MAX_DEPTH)
    elif strategy == "random":
        config = SolverConfig(strategy="random", log_steps=False, max_depth=MAX_DEPTH)
    elif strategy == "guided":
        config = SolverConfig(strategy="guided", log_steps=False, max_depth=MAX_DEPTH)
    else:
        raise ValueError("Stratégie inconnue")

    t0 = time.time()
    result = subset_sum_solver(numbers, target, config=config)
    duration = time.time() - t0

    return {
        "found": len(result["solutions"]) > 0,
        "steps": result["steps"],
        "duration": round(duration, 4),
        "solutions": result["solutions"][:1]  # juste la première
    }

# === MAIN LOOP
def main():
    vectors = load_vectors(EEG_JSON, topn=10)
    logs = []

    for idx, entry in enumerate(vectors):
        vec = np.array(entry["vector"])
        base_set = np.abs(vec[:19]).astype(int).tolist()
        target = sum(base_set[:3]) + TARGET_OFFSET

        print(f"\n🧠 Vecteur EEG {idx+1}/{len(vectors)} | Target = {target}")

        for strat in STRATEGIES:
            log = {
                "id": entry["id"],
                "subject": entry.get("subject", "unknown"),
                "strategy": strat
            }

            result = run_benchmark(base_set, target, strat)
            log.update(result)
            logs.append(log)
            print(f"✅ {strat:<7} | found: {result['found']} | steps: {result['steps']} | time: {result['duration']}s")

    df = pd.DataFrame(logs)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n📁 Résultats sauvegardés : {OUTPUT_CSV}")

    # === VISUALISATION TEMPS vs SUCCÈS
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="strategy", y="duration", hue="found")
    plt.title("⏱️ Temps de résolution (par stratégie)")
    plt.ylabel("Durée (secondes)")
    plt.xlabel("Stratégie")
    plt.legend(title="Succès")
    plt.tight_layout()
    plt.savefig("benchmark_time_vs_success.png")
    print("📊 Graphique temps vs succès : benchmark_time_vs_success.png")

    # === MATRICE DE CONFUSION STRATÉGIE × VECTEUR
    pivot = df.pivot_table(index="id", columns="strategy", values="found", aggfunc="sum").fillna(0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", cbar=True, fmt=".0f")
    plt.title("📈 Succès par Vecteur EEG × Stratégie")
    plt.xlabel("Stratégie")
    plt.ylabel("Vecteur EEG")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "benchmark_matrix.png"))
    print("📊 Matrice stratégie × EEG : benchmark_matrix.png")


    # === RADAR PLOT PAR STRATÉGIE
    radar_metrics = df.groupby("strategy").agg({
        "found": "mean",
        "duration": "mean",
        "steps": "mean"
    }).reset_index()

    # Normalisation entre 0 et 1
    normed = radar_metrics.copy()
    for col in ["found", "duration", "steps"]:
        normed[col] = (normed[col] - normed[col].min()) / (normed[col].max() - normed[col].min() + 1e-8)

    categories = ["Success Rate", "Duration", "Steps"]
    N = len(categories)

    plt.figure(figsize=(6, 6))
    for i, row in normed.iterrows():
        values = row[["found", "duration", "steps"]].tolist()
        values += values[:1]
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        plt.polar(angles, values, label=row["strategy"], linewidth=2)

    plt.xticks([n / float(N) * 2 * pi for n in range(N)], categories)
    plt.title("Radar plot des stratégies")
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig("benchmark_radar_plot.png")
    print("📊 Radar plot sauvegardé : benchmark_radar_plot.png")

    # === HTML DASHBOARD
    html_path = os.path.join(RESULTS_DIR, "benchmark_summary.html")
    with open(html_path, "w") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>🧠 NeuroSolv Benchmark Résumé</title>
            <style>
                body {{ font-family: sans-serif; padding: 20px; }}
                h1 {{ color: #333; }}
                img {{ max-width: 600px; display: block; margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <h1>🧠 Résultats du Benchmark NeuroSolv</h1>
            <p><a href="benchmark_results.csv" download>📥 Télécharger les résultats (.csv)</a></p>

            <h2>1. Temps de résolution par stratégie (avec succès)</h2>
            <img src="benchmark_time_vs_success.png" alt="Barplot temps vs succès">

            <h2>2. Matrice de succès EEG × Stratégie</h2>
            <img src="benchmark_matrix.png" alt="Matrice EEG/Strat">

            <h2>3. Radar Plot de performance</h2>
            <img src="benchmark_radar_plot.png" alt="Radar stratégie">

            <hr>
            <p><em>Généré automatiquement par ns999_benchmark.py</em></p>
        </body>
        </html>
        """)

    # Copier le CSV dans le dossier courant pour le HTML
    shutil.copy(OUTPUT_CSV, os.path.join(os.getcwd(), "benchmark_results.csv"))
    webbrowser.open(os.path.abspath(html_path))
    print(f"📄 Dashboard ouvert : {html_path}")

if __name__ == "__main__":
    main()

