"""
ns008_riemann_explorer.py

Licence : Creative Commons BY-NC-SA 4.0

Auteurs : 
    - Kocupyr Romain (chef de projet) : rkocupyr@gmail.com
    - Multi_gpt_api 
    - Grok3
"""

import numpy as np
import mne
from mpmath import zetazero
import matplotlib.pyplot as plt

# === CONFIGURATION
EEG_FILE = "chemin/vers/votre_fichier_eeg.set"  # Remplacez par le chemin de votre fichier EEG
CHANNEL = "Cz"  # Nom du canal EEG à analyser
SAMPLES = 512  # Nombre d'échantillons à extraire
FS = 256  # Fréquence d'échantillonnage en Hz

# === CHARGEMENT DES DONNÉES EEG
def charger_donnees_eeg(fichier, canal, echantillons):
    raw = mne.io.read_raw_eeglab(fichier, preload=True, verbose=False)
    raw.pick_channels([canal])
    data, times = raw[:]
    return data[0, :echantillons], times[:echantillons]

# === MISE EN CORRESPONDANCE EEG ↔ ZÉROS DE RIEMANN
def mapper_eeg_riemann(eeg_data):
    puissances = np.abs(np.fft.fft(eeg_data))[:len(eeg_data) // 2]
    puissances = (puissances - np.min(puissances)) / (np.max(puissances) - np.min(puissances))
    indices_zeros = (puissances * 10).astype(int)
    zeros_riemann = [zetazero(n+1).imag for n in indices_zeros]
    return zeros_riemann

# === VISUALISATION
def visualiser_mapping(eeg_data, zeros_riemann, temps):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(temps, eeg_data)
    plt.title("Signal EEG")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.stem(zeros_riemann, use_line_collection=True)
    plt.title("Zéros de la fonction zêta de Riemann correspondants")
    plt.xlabel("Index")
    plt.ylabel("Partie Imaginaire")

    plt.tight_layout()
    plt.show()

# === MAIN
if __name__ == "__main__":
    eeg_data, temps = charger_donnees_eeg(EEG_FILE, CHANNEL, SAMPLES)
    zeros_riemann = mapper_eeg_riemann(eeg_data)
    visualiser_mapping(eeg_data, zeros_riemann, temps)
