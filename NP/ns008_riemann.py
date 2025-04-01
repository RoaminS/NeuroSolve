# ns008_riemann_explorer.py
"""
🧠 NeuroSolv Module 008 (Boosted) : EEG ↔ Zéros de Riemann + Fourier + Export JSON

- Mapping EEG vers les zéros de ζ(s)
- Visualisation superposée avec spectre de Fourier
- Densité spectrale alignée avec les zéros
- Export JSON mappé

Auteur : Kocupyr Romain
Dev    : multi_gpt_api, Grok3
Licence : CC BY-NC-SA 4.0
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.fft import fft, fftfreq
from mpmath import zetazero
from datetime import datetime

# === CONFIG
EEG_VECTOR_PATH = "neurosolv_sessions/session_001.json"
EXPORT_JSON_PATH = "neurosolv_sessions/riemann_mapped.json"
IMG_OUT = "neurosolv_sessions/riemann_mapping_superposed.png"
FS = 128
N_ZEROS = 30

# === UTILS
def load_eeg_vector(path):
    with open(path) as f:
        session = json.load(f)
    return np.array(session["eeg_vector"])

def get_riemann_zeros(n=N_ZEROS):
    return [zetazero(i+1) for i in range(n)]

def map_eeg_to_zeros(eeg_vec, n_zeros=N_ZEROS):
    eeg_norm = (eeg_vec - np.min(eeg_vec)) / (np.max(eeg_vec) - np.min(eeg_vec))
    eeg_idx = (eeg_norm * (n_zeros - 1)).astype(int)
    zeros = get_riemann_zeros(n_zeros)
    return [zeros[i].imag for i in eeg_idx], eeg_idx.tolist()

def compute_fourier(eeg_vec):
    yf = np.abs(fft(eeg_vec))
    xf = fftfreq(len(eeg_vec), 1 / FS)
    return xf[:len(xf)//2], yf[:len(yf)//2]

def export_json(eeg, zeros_img, indices):
    export = {
        "timestamp": datetime.now().isoformat(),
        "eeg_vector": eeg.tolist(),
        "riemann_indices": indices,
        "riemann_zeros_imag": zeros_img
    }
    with open(EXPORT_JSON_PATH, "w") as f:
        json.dump(export, f, indent=2)
    print(f"📁 Export JSON : {EXPORT_JSON_PATH}")

# === PLOT
def plot_combined(eeg, zeros_img, fft_x, fft_y):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    axs[0].bar(range(len(eeg)), eeg, color="skyblue")
    axs[0].set_title("EEG Vector (amplitude)")

    axs[1].stem(zeros_img, linefmt='r-', markerfmt='ro', basefmt=" ")
    axs[1].set_title("Zéros de Riemann mappés depuis EEG")

    axs[2].plot(fft_x, fft_y, color="green", linewidth=1.5)
    axs[2].set_xlim(0, 64)
    axs[2].set_title("Spectre de Fourier EEG")

    plt.tight_layout()
    plt.savefig(IMG_OUT)
    plt.show()
    print(f"🖼️ Image sauvegardée : {IMG_OUT}")

# === MAIN
if __name__ == "__main__":
    print("🧠 Mapping EEG ↔ ζ(s) + Fourier boost lancé...")

    eeg = load_eeg_vector(EEG_VECTOR_PATH)
    zeros_img, indices = map_eeg_to_zeros(eeg, n_zeros=N_ZEROS)
    fft_x, fft_y = compute_fourier(eeg)

    export_json(eeg, zeros_img, indices)
    plot_combined(eeg, zeros_img, fft_x, fft_y)

    print("✅ Riemann EEG Explorer terminé.")
