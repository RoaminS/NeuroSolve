#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ns016_alert_system.py

✅ Système d’alerte basé sur les prédictions EEG
✅ Analyse des logs JSON produits par ns014
✅ Détection de seuils critiques (ex. : proba > 0.85)
✅ Export des événements d’alerte + option de copie EEG

Auteur : Kocupyr Romain
Dev    : multi_gpt_api, NeuroSolve
Licence : CC BY-NC-SA 4.0
"""

import os
import json
import pandas as pd
from datetime import datetime

# === CONFIGURATION
PREDICTIONS_JSON = "predictions_log.json"
ALERTS_JSON = "alerts_detected.json"
THRESHOLD = 0.85
MIN_CONSECUTIVE = 2  # nombre d'alertes consécutives avant déclenchement officiel

def detect_alerts(predictions, threshold=THRESHOLD, consecutive=MIN_CONSECUTIVE):
    alerts = []
    buffer = []

    for i, p in enumerate(predictions):
        proba = p.get("prob_class_1", 0)
        if proba >= threshold:
            buffer.append(p)
            if len(buffer) >= consecutive:
                event = {
                    "type": "HIGH_PROBA_ALERT",
                    "start_time": buffer[0]["time_sec"],
                    "end_time": buffer[-1]["time_sec"],
                    "duration": round(buffer[-1]["time_sec"] - buffer[0]["time_sec"], 2),
                    "mean_prob": round(sum(b["prob_class_1"] for b in buffer) / len(buffer), 3),
                    "frames": [b["iteration"] for b in buffer],
                    "timestamp_triggered": datetime.now().isoformat()
                }
                alerts.append(event)
                buffer = []
        else:
            buffer = []

    return alerts

def save_alerts(alerts, path=ALERTS_JSON):
    with open(path, "w") as f:
        json.dump(alerts, f, indent=2)
    print(f"⚠️  {len(alerts)} alertes détectées et sauvegardées dans : {path}")

def main():
    print("🧠 NS033 — Système d’alerte cognitive EEG en action...")

    if not os.path.exists(PREDICTIONS_JSON):
        print(f"❌ Fichier introuvable : {PREDICTIONS_JSON}")
        return

    with open(PREDICTIONS_JSON) as f:
        predictions = json.load(f)

    alerts = detect_alerts(predictions)
    save_alerts(alerts)

    if alerts:
        print("📢 ALERTES DÉTECTÉES :")
        for alert in alerts:
            print(f"→ De {alert['start_time']}s à {alert['end_time']}s | proba moyenne : {alert['mean_prob']}")
    else:
        print("✅ Aucun seuil critique détecté.")

if __name__ == "__main__":
    main()
