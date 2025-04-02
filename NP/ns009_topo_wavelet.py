"""
ns009_topo_wavelet_pro_gold.py

✅ Analyse topologique avancée des signaux EEG
✅ Transformation en ondelettes pour extraction des caractéristiques
✅ Visualisation des résultats avec heatmaps
✅ Exportation des caractéristiques en JSON
✅ Traitement multi-sujets avec gestion des erreurs
✅ Génération automatique d'un dashboard HTML

Auteur : Kocupyr Romain
Dev    : multi_gpt_api, Grok3
Licence : Creative Commons BY-NC-SA 4.0
"""

import os
import json
import numpy as np
import mne
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from ripser import ripser
from persim import plot_diagrams
from jinja2 import Template

# === CONFIGURATION
DATA_DIR = "eeg_data"
RESULTS_DIR = "results"
DASHBOARD_FILE = os.path.join(RESULTS_DIR, "dashboard.html")
WAVELET = 'db4'
LEVEL = 5
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Chargement des données EEG
def load_eeg_data(file_path):
    try:
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        raw.pick_types(eeg=True)
        raw.filter(0.5, 45)
        return raw
    except Exception as e:
        print(f"Erreur lors du chargement de {file_path} : {e}")
        return None

# === Transformation en ondelettes
def wavelet_transform(data, wavelet=WAVELET, level=LEVEL):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return coeffs

# === Analyse topologique
def topological_analysis(coeffs):
    diagrams = ripser(coeffs)['dgms']
    return diagrams

# === Visualisation des heatmaps des coefficients d'ondelettes
def plot_wavelet_heatmap(coeffs, save_path):
    plt.figure(figsize=(12, 6))
    coeff_arr = pywt.coeffs_to_array(coeffs)[0]
    sns.heatmap(coeff_arr, cmap='coolwarm', cbar=True)
    plt.title("Heatmap des coefficients d'ondelettes")
    plt.savefig(save_path)
    plt.close()

# === Visualisation des diagrammes de persistance
def plot_persistence_diagrams(diagrams, save_path):
    plt.figure(figsize=(10, 5))
    plot_diagrams(diagrams, show=False)
    plt.title("Diagrammes de persistance")
    plt.savefig(save_path)
    plt.close()

# === Sauvegarde des caractéristiques en JSON
def save_features_to_json(features, save_path):
    with open(save_path, 'w') as f:
        json.dump(features, f, indent=4)

# === Génération du dashboard HTML
def generate_dashboard(results):
    template = Template("""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>Dashboard des Résultats EEG</title>
        <style>
            body { font-family: Arial, sans-serif; }
            .container { display: flex; flex-wrap: wrap; }
            .result { margin: 20px; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <h1>Dashboard des Résultats EEG</h1>
        <div class="container">
            {% for result in results %}
            <div class="result">
                <h2>{{ result.subject }} - Canal {{ result.channel }}</h2>
                <img src="{{ result.heatmap }}" alt="Heatmap">
                <img src="{{ result.persistence }}" alt="Diagramme de persistance">
                <pre>{{ result.features }}</pre>
            </div>
            {% endfor %}
        </div>
    </body>
    </html>
    """)
    with open(DASHBOARD_FILE, 'w') as f:
        f.write(template.render(results=results))

# === Boucle principale
def main():
    subjects = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    all_results = []

    for subject in subjects:
        subject_dir = os.path.join(DATA_DIR, subject)
        eeg_files = [f for f in os.listdir(subject_dir) if f.endswith('.fif')]

        for eeg_file in eeg_files:
            file_path = os.path.join(subject_dir, eeg_file)
            raw = load_eeg_data(file_path)
            if raw is None:
                continue

            data = raw.get_data()
            for i, channel_data in enumerate(data):
                coeffs = wavelet_transform(channel_data)
                diagrams = topological_analysis(coeffs)

                heatmap_path = os.path.join(RESULTS_DIR, f"{subject}_{eeg_file}_channel_{i}_heatmap.png")
                persistence_path = os.path.join(RESULTS_DIR, f"{subject}_{eeg_file}_channel_{i}_persistence.png")
                features_path = os.path.join(RESULTS_DIR, f"{subject}_{eeg_file}_channel_{i}_features.json")

                plot_wavelet_heatmap(coeffs, heatmap_path)
                plot_persistence_diagrams(diagrams, persistence_path)

                features = {
                    'subject': subject,
                    'eeg_file': eeg_file,
                    'channel': i,
                    'coefficients': [c.tolist() for c in coeffs]
                }
                save_features_to_json(features, features_path)

                all_results.append({
                    'subject': subject,
                    'channel': i,
                    'heatmap': heatmap_path,
                    'persistence': persistence_path,
                    'features': json.dumps(features, indent=4)
                })

                print(f"Résultats sauvegardés pour {subject}, fichier {eeg_file}, canal {i}")

    generate_dashboard(all_results)
    print(f"Dashboard généré : {DASHBOARD_FILE}")

if __name__ == "__main__":
    main()
