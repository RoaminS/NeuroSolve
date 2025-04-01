"""
ns006_live_solver.py
Auteur : Kocupyr Romain
Dev    : multi_gpt_api, Grok3
Licence : Creative Commons BY-NC-SA 4.0
"""

import mne
import numpy as np
from time import sleep
from ns001_neuro_np_solver import solve_np_problem
from ns003_visualizer import plot_solution_2d
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# === Config
CHANNELS = 19
SAMPLES = 512
FS = 128
NP_TASK = "subset_sum"
THRESHOLD = 50  # Valeur EEG pour activer un noeud

print("🧠 [NeuroSolv Live] Acquisition EEG via MNE en cours...")

# === Flux MNE en direct (remplace ça selon ton matériel)
# Pour test : fichier EEG simulé ou en live
raw = mne.io.read_raw_eeglab("demo_eeg.set", preload=True)
raw.pick_types(eeg=True)
raw.filter(1., 45., fir_design='firwin')
raw.resample(FS)

print("✅ Flux EEG prêt. Début de l’analyse...")

# === Boucle de traitement par fenêtre
start = 0
step = SAMPLES  # Slide de 512 échantillons
duration = raw.times[-1] / FS
np.random.seed(42)

while start + SAMPLES < raw.n_times:
    window = raw.get_data(start=start, stop=start + SAMPLES)  # (n_channels, SAMPLES)
    window = window.T  # (SAMPLES, n_channels)
    
    print(f"\n🔁 Fenêtre EEG {start}-{start+SAMPLES} | shape={window.shape}")

    # === Normalisation
    normed = (window - np.mean(window, axis=0)) / (np.std(window, axis=0) + 1e-6)

    # === Mapping vers un problème NP (ex: subset sum)
    # On simule une correspondance EEG → ensemble de nombres
    signal_vector = np.mean(np.abs(normed), axis=0) * 100  # (19,)
    eeg_set = signal_vector.astype(int).tolist()

    # Ajout d'un peu d'aléa pour la démonstration
    eeg_set = [min(max(val + np.random.randint(-5, 5), 1), 100) for val in eeg_set]
    target = THRESHOLD + np.random.randint(10, 30)

    print(f"🎯 Ensemble EEG : {eeg_set} | Target = {target}")

    # === Résolution via solveur NS001
    solution, meta = solve_np_problem(eeg_set, target, algo="brute", return_meta=True)

    if solution:
        print(f"✅ Solution trouvée : {solution}")
    else:
        print("❌ Aucune solution NP trouvée.")

    # === Visualisation live
    plot_solution_2d(eeg_set, solution, title=f"EEG Window {start}-{start+SAMPLES}")

    # Pause pour la boucle (réglable)
    sleep(1.5)
    start += step

print("🎉 Analyse EEG en temps réel terminée.")
