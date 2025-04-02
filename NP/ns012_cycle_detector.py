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
import imageio
from matplotlib.animation import FuncAnimation
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

# Fusion avec topo_wavelet → pré-version NS013
def extract_wavelet_features(data, wavelet='db4', level=4):
    import pywt
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeff_arr = pywt.coeffs_to_array(coeffs)[0]
    return coeff_arr.flatten()

# === MAIN
def main():
    print("🧠 Détection des cycles EEG persistants (Betti-1)")
    data = load_eeg_segment(EEG_PATH, channel="Cz")
    cycles, windows = detect_cycles(data)
    export_cycles(cycles, RESULTS_DIR)
    plot_cycles(cycles, os.path.join(RESULTS_DIR, "cycles_persistence_timeline.png"))
    generate_cycle_gif(data, FS, windows)
    wavelet_feat = extract_wavelet_features(data)
    # Tu peux maintenant combiner wavelet_feat + persistence pour un modèle ML ou une visualisation


def generate_cycle_gif(data, fs, windows, out_path="ns012_results/cycle_animation.gif"):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_ylim(np.min(data), np.max(data))
    ax.set_xlim(0, len(data) / fs)
    ax.set_title("🎞️ Animation des cycles détectés")
    line, = ax.plot([], [], lw=2)

    def update(i):
        start, _ = windows[i]
        x = np.linspace(start / fs, (start + WINDOW_SIZE) / fs, WINDOW_SIZE)
        y = data[start:start + WINDOW_SIZE]
        line.set_data(x, y)
        return line,

    ani = FuncAnimation(fig, update, frames=len(windows), blit=True)
    ani.save(out_path, writer='pillow', fps=2)
    print(f"🎞️ Animation sauvegardée : {out_path}")

def correlate_with_labels(cycles_csv, labels_csv):
    cycles_df = pd.read_csv(cycles_csv)
    labels_df = pd.read_csv(labels_csv)  # Doit contenir colonne: subject_id, mmse_score ou class

    # Exemple : fusion par ID (à adapter selon format EEG/label)
    merged = pd.merge(cycles_df, labels_df, how="left", on="subject_id")

    # Corrélation simple persistance ↔ MMSE
    if "mmse_score" in merged.columns:
        corr = merged["persistence"].corr(merged["mmse_score"])
        print(f"📊 Corrélation cycles / MMSE : {corr:.4f}")
    elif "class" in merged.columns:
        sns.boxplot(data=merged, x="class", y="persistence")
        plt.title("Persistance topologique par classe cognitive")
        plt.tight_layout()
        plt.savefig("ns012_results/cycle_vs_class.png")
        print("📈 Boxplot cycle/class : ns012_results/cycle_vs_class.png")


if __name__ == "__main__":
    main()
