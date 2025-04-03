"""
backfill_summaries.py

Licence : Creative Commons BY-NC-SA 4.0

📂 Générer summary.json manquant pour les anciennes sessions

Auteurs : 
    - Kocupyr Romain (chef de projet) : rkocupyr@gmail.com
    - GPT multi_gpt_api (OpenAI)
"""

import os
import json
import pandas as pd
from datetime import datetime

LOGS_DIR = "logs"

for folder in os.listdir(LOGS_DIR):
    session_path = os.path.join(LOGS_DIR, folder)
    if not os.path.isdir(session_path):
        continue

    summary_path = os.path.join(session_path, "summary.json")
    json_path = os.path.join(session_path, "predictions_log.json")

    if not os.path.exists(summary_path) and os.path.exists(json_path):
        with open(json_path) as f:
            preds = json.load(f)
        df = pd.DataFrame(preds)
        if len(df) == 0:
            continue
        summary = {
            "session": folder,
            "nb_frames": len(df),
            "nb_alerts": df["alert"].sum() if "alert" in df else 0,
            "alert_rate": round(df["alert"].mean(), 3) if "alert" in df else 0,
            "duration_sec": round(df["time_sec"].max() - df["time_sec"].min(), 2),
            "avg_prob_class_1": round(df["prob_class_1"].mean(), 3),
            "timestamp_generated": datetime.now().isoformat()
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Résumé généré pour : {folder}")
