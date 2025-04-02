"""
ns010_topo_mapper.py

✅ Chargement et prétraitement des données EEG
✅ Calcul des coefficients d'ondelettes et des diagrammes de persistance
✅ Extraction des caractéristiques topologiques
✅ Visualisation et sauvegarde des résultats
✅ Gestion des arguments en ligne de commande (--input, --outdir)
✅ Intégration des noms des canaux dans les exports
✅ Réduction de dimension avec UMAP/t-SNE pour interprétation XAI
✅ Logs détaillés pour suivi de l'exécution
✅ Comparaison entre sujets/groupes et génération d'un rapport global

Auteur : Kocupyr Romain
Dev    : multi_gpt_api, Grok3
Licence : Creative Commons BY-NC-SA 4.0
"""

import os
import json
import argparse
import logging
import numpy as np
import mne
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import pandas as pd
from ripser import ripser
from persim import plot_diagrams
from sklearn.manifold import TSNE
from jinja2 import Template

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Chargement des données EEG
def load_eeg_data(file_path):
    try:
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        raw.pick_types(eeg=True)
        logging.info(f"Données chargées depuis {file_path}")
        return raw
    except Exception as e:
        logging.error(f"Erreur lors du chargement de {file_path} : {e}")
        return None

# === Prétraitement des données EEG
def preprocess(raw):
    raw.filter(0.5, 45, fir_design='firwin')
    raw.set_eeg_reference('average', projection=True)
    logging.info("Prétraitement effectué : filtrage et re-référencement")
    return raw

# === Calcul des coefficients d'ondelettes
def compute_wavelet_coeffs(data, wavelet='db4', level=5):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    logging.info("Coefficients d'ondelettes calculés")
    return coeffs

# === Calcul des diagrammes de persistance
def compute_persistence_diagrams(coeffs):
    diagrams = ripser(coeffs)['dgms']
    logging.info("Diagrammes de persistance calculés")
    return diagrams

# === Extraction des caractéristiques topologiques
def extract_topo_features(diagrams):
    features = []
    for dim, diagram in enumerate(diagrams):
        for birth, death in diagram:
            features.append({
                'dimension': dim,
                'birth': birth,
                'death': death,
                'persistence': death - birth
            })
    logging.info("Caractéristiques topologiques extraites")
    return features

# === Réduction de dimension pour interprétation XAI
def reduce_dimensions(features, method='umap'):
    df = pd.DataFrame(features)
    if method == 'umap':
        reducer = umap.UMAP()
    elif method == 'tsne':
        reducer = TSNE()
    else:
        raise ValueError("Méthode de réduction de dimension non reconnue. Utiliser 'umap' ou 'tsne'.")
    embedding = reducer.fit_transform(df[['birth', 'death', 'persistence']])
    df['x'] = embedding[:, 0]
    df['y'] = embedding[:, 1]
    logging.info(f"Réduction de dimension effectuée avec {method}")
    return df

# === Visualisation des diagrammes de persistance
def plot_diagram(diagrams, save_path):
    plt.figure(figsize=(10, 5))
    plot_diagrams(diagrams, show=False)
    plt.title("Diagrammes de persistance")
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Diagramme de persistance sauvegardé : {save_path}")

# === Sauvegarde des caractéristiques en JSON
def save_json_features(features, save_path):
    with open(save_path, 'w') as f:
        json.dump(features, f, indent=4)
    logging.info(f"Caractéristiques sauvegardées en JSON : {save_path}")

# === Exportation d'un résumé HTML
def export_summary(results, output_dir):
    template = Template("""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>Résumé des Résultats EEG</title>
        <style>
            body { font-family: Arial, sans-serif; }
            .container { display: flex; flex-wrap: wrap; }
            .result { margin: 20px; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <h1>Résumé des Résultats EEG</h1>
        <div class="container">
            {% for result in results %}
            <div class="result">
                <h2>{{ result.subject }} - Canal {{ result.channel }}</h2>
                <img src="{{ result.diagram }}" alt="Diagramme de persistance">
                <pre>{{ result.features }}</pre>
            </div>
            {% endfor %}
        </div>
    </body>
    </html>
    """)
    summary_path = os.path.join(output_dir, "topo_mapping.html")
    with open(summary_path, 'w') as f:
        f.write(template.render(results=results))
    logging.info(f"Résumé HTML généré : {summary_path}")

# === Fonction principale
def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    subjects = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    all_results = []

    for subject in subjects:
        subject_dir = os.path.join(input_dir, subject)
        eeg_files = [f for f in os.listdir(subject_dir) if f.endswith('.fif')]

        for eeg_file in eeg_files:
            file_path = os.path.join(subject_dir, eeg_file)
            raw = load_eeg_data(file_path)
            if raw is None:
                continue

            raw = preprocess(raw)
            data = raw.get_data()
            channel_names = raw.info['ch_names']

            for i, channel_data in enumerate(data):
                coeffs = compute_wavelet_coeffs(channel_data)
                diagrams = compute_persistence_diagrams(coeffs)
                features = extract_topo_features(diagrams)
                reduced_features = reduce_dimensions(features)

                diagram_path = os.path.join(output_dir, f"{subject}_{eeg_file}_channel_{i}_diagram.png")
                features_path = os.path.join(output_dir, f"{subject}_{eeg_file}_channel_{i}_features.json")

                plot_diagram(diagrams, diagram_path)
                save_json_features(features, features_path)

                all_results.append({
                    'subject': subject,
                    'channel': channel_names[i],
                    'diagram': diagram_path,
                    'features': json.dumps(features, indent=4)
                })

                logging.info(f"Résultats sauvegardés
::contentReference[oaicite:0]{index=0}
 
