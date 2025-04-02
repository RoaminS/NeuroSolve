"""
ns009_topo_wavelet.py

✅ Analyse topologique des signaux EEG
✅ Transformation en ondelettes pour extraction des caractéristiques
✅ Visualisation des résultats

Auteur : Kocupyr Romain
Dev    : multi_gpt_api, Grok3
Licence : Creative Commons BY-NC-SA 4.0
"""

import os
import numpy as np
import mne
import pywt
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams

# === CONFIGURATION
DATA_DIR = "eeg_data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Chargement des données EEG
def load_eeg_data(file_path):
    raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
    raw.pick_types(eeg=True)
    raw.filter(0.5, 45)
    return raw

# === Transformation en ondelettes
def wavelet_transform(data, wavelet='db4', level=5):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return coeffs

# === Analyse topologique
def topological_analysis(coeffs):
    diagrams = ripser(coeffs)['dgms']
    return diagrams

# === Visualisation
def plot_results(diagrams, save_path):
    plt.figure(figsize=(10, 5))
    plot_diagrams(diagrams, show=False)
    plt.title("Diagrammes de persistance")
    plt.savefig(save_path)
    plt.close()

# === Boucle principale
def main():
    eeg_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.fif')]
    for eeg_file in eeg_files:
        file_path = os.path.join(DATA_DIR, eeg_file)
        raw = load_eeg_data(file_path)
        data = raw.get_data()
        for i, channel_data in enumerate(data):
            coeffs = wavelet_transform(channel_data)
            diagrams = topological_analysis(coeffs)
            save_path = os.path.join(RESULTS_DIR, f"{eeg_file}_channel_{i}.png")
            plot_results(diagrams, save_path)
            print(f"Résultats sauvegardés pour {eeg_file}, canal {i}")

if __name__ == "__main__":
    main()
