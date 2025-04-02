#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ns012_cycle_detector.py

✅ Détection de cycles cognitifs persistants dans les signaux EEG
✅ Sliding window + homologie persistante (Betti-1)
✅ Visualisation temporelle et export des cycles détectés

Auteur  : Kocupyr Romain
Dev     : multi_gpt_api, Grok3
Licence : CC BY-NC-SA 4.0
"""

import os
import numpy as np
import mne
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ripser import ripser
from persim import plot_diagrams

# === CONFIG
EEG_PATH = "eeg_samples/subject_01.set"
RESULTS_DIR = "ns012_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

WINDOW_SIZE = 256
STEP_SIZE = 64
FS = 128

# === Chargement EEG
def load_eeg_segment(path, channel="Cz"):
    raw = mne.io.read_raw_eeglab(path, preload=True)
    raw.pick_channels([channel])
    raw.filter(0.5, 45)
    data = raw.get_data()[0]
    return data

# === Analyse TDA sur fenêtre temporelle
def detect_cycles(data, fs=FS, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    windows = []
    cycles = []

    for start in range(0, len(data) - window_size, step_size):
        segment = data[start:start + window_size]
        traj = np.array([segment[i:i+3] for i in range(len(segment) - 3)])
        diagrams = ripser(traj)['dgms']
        b1 = diagrams[1]  # Betti-1 → cycles
        for birth, death in b1:
            cycles.append({
                "start_sec": start / fs,
                "end_sec": (start + window_size) / fs,
                "persistence": round(death - birth, 4),
                "birth": round(birth, 4),
                "death": round(death, 4)
            })
        windows.append((start, b1))

    return cycles, windows

# === Visualisation persistences dans le temps
def plot_cycles(cycles, out_path):
    df = pd.DataFrame(cycles)
    if df.empty:
        print("❌ Aucun cycle détecté.")
        return
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="start_sec", y="persistence", hue="persistence", palette="coolwarm", size="persistence")
    plt.title("🔁 Cycles persistants détectés (Betti-1) sur le temps")
    plt.xlabel("Temps (s)")
    plt.ylabel("Persistance")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"📈 Visualisation sauvegardée : {out_path}")

# === Sauvegarde JSON + CSV
def export_cycles(cycles, results_dir):
    json_path = os.path.join(results_dir, "cycles_detected.json")
    csv_path = os.path.join(results_dir, "cycles_detected.csv")

    with open(json_path, "w") as f:
        json.dump(cycles, f, indent=2)
    pd.DataFrame(cycles).to_csv(csv_path, index=False)
    print(f"💾 Export JSON : {json_path}")
    print(f"💾 Export CSV  : {csv_path}")

# === MAIN
def main():
    print("🧠 Détection des cycles EEG persistants (Betti-1)")
    data = load_eeg_segment(EEG_PATH, channel="Cz")
    cycles, windows = detect_cycles(data)
    export_cycles(cycles, RESULTS_DIR)
    plot_cycles(cycles, os.path.join(RESULTS_DIR, "cycles_persistence_timeline.png"))

if __name__ == "__main__":
    main()
