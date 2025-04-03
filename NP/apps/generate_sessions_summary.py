#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_sessions_summary.py

Licence : Creative Commons BY-NC-SA 4.0

✅ Analyse transversale des logs EEG (sessions/)
✅ Génère un fichier sessions_summary.csv
✅ Basé sur les summary.json de chaque session

Auteur : Kocupyr Romain
Dev    : multi_gpt_api
"""

import os
import json
import pandas as pd
import plotly.express as px

LOGS_DIR = "logs"
SUMMARY_FILE = "sessions_summary.csv"
summaries = []

print("🧠 Génération du résumé global des sessions EEG...")

for folder in os.listdir(LOGS_DIR):
    folder_path = os.path.join(LOGS_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    summary_path = os.path.join(folder_path, "summary.json")
    if not os.path.exists(summary_path):
        print(f"❌ summary.json manquant dans : {folder}")
        continue

    with open(summary_path) as f:
        summary = json.load(f)
        summary["session_folder"] = folder
        summaries.append(summary)

if not summaries:
    print("❌ Aucun résumé de session trouvé.")
    exit()

df = pd.DataFrame(summaries)

# === Envoyer à l'API
def push_summary_to_api(csv_path, api_url="http://localhost:6000/upload_summary"):
    try:
        with open(csv_path, 'rb') as f:
            files = {'file': (os.path.basename(csv_path), f)}
            r = requests.post(api_url, files=files)
        if r.status_code == 200:
            print("📡 Résumé envoyé à l’API.")
        else:
            print(f"❌ Échec envoi : {r.status_code}")
    except Exception as e:
        print(f"⚠️ Erreur d’envoi : {e}")

# ✅ Tri chronologique
df = df.sort_values("timestamp_generated")

# ✅ Sauvegarde CSV
df.to_csv(SUMMARY_FILE, index=False)
print(f"✅ Résumé global sauvegardé : {SUMMARY_FILE}")
push_summary_to_api(SUMMARY_FILE)

# ✅ Aperçu terminal
print(df[["session_folder", "nb_frames", "nb_alerts", "alert_rate", "duration_sec", "avg_prob_class_1"]].tail())


# === VISU GLOBALE
plot_path = "sessions_summary_plot.html"
fig = px.bar(
    df,
    x="session",
    y="alert_rate",
    title="📊 Taux d’alertes par session EEG",
    labels={"alert_rate": "Taux d’alerte"},
    hover_data=["nb_alerts", "nb_frames", "duration_sec"]
)
fig.write_html(plot_path)
print(f"✅ Graphe global sauvegardé : {plot_path}")
