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

# ✅ Tri chronologique
df = df.sort_values("timestamp_generated")

# ✅ Sauvegarde CSV
df.to_csv(SUMMARY_FILE, index=False)
print(f"✅ Résumé global sauvegardé : {SUMMARY_FILE}")

# ✅ Aperçu terminal
print(df[["session_folder", "nb_frames", "nb_alerts", "alert_rate", "duration_sec", "avg_prob_class_1"]].tail())
