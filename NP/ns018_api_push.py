#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ns018_api_push.py

✅ Envoi des prédictions EEG et alertes vers une API Flask distante
✅ Lecture de predictions_log.json
✅ Requête POST sécurisée + affichage réponse serveur

Auteur : Kocupyr Romain
"""

import json
import requests

PREDICTIONS_FILE = "predictions_log.json"
ALERTS_FILE = "alerts_detected.json"

API_ENDPOINT_PRED = "http://localhost:5000/receive_prediction"
API_ENDPOINT_ALERT = "http://localhost:5000/receive_alert"

def push_data():
    print("📡 NS050 — Push vers API Flask en cours...")

    if not (os.path.exists(PREDICTIONS_FILE) and os.path.exists(ALERTS_FILE)):
        print("❌ Fichiers predictions ou alerts introuvables.")
        return

    with open(PREDICTIONS_FILE) as f:
        predictions = json.load(f)

    with open(ALERTS_FILE) as f:
        alerts = json.load(f)

    # Envoi prédictions
    r_pred = requests.post(API_ENDPOINT_PRED, json={"data": predictions})
    print("✅ Prédictions envoyées :", r_pred.status_code, r_pred.text)

    # Envoi alertes
    r_alert = requests.post(API_ENDPOINT_ALERT, json={"data": alerts})
    print("✅ Alertes envoyées :", r_alert.status_code, r_alert.text)

if __name__ == "__main__":
    push_data()
