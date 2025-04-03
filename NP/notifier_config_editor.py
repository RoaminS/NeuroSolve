"""
notifier_config_editor.py

Licence : Creative Commons BY-NC-SA 4.0

✅ Interface Streamlit pour éditer notifier_config.json

Auteur : Kocupyr Romain
Dev    : multi_gpt_api
"""

import os
import json
import streamlit as st

CONFIG_PATH = "notifier_config.json"

st.set_page_config(page_title="🛠️ Éditeur Notification NeuroSolve")
st.title("🛠️ Config Notifications • NeuroSolve")

# === Charger ou créer config
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
else:
    st.warning("⚠️ Fichier `notifier_config.json` non trouvé. Création automatique...")
    config = {
        "sender_email": "",
        "password": "",
        "recipients": [],
        "smtp_server": "smtp.example.com",
        "smtp_port": 587,
        "push_to_api": False
    }

# === UI édition
st.markdown("### ✉️ Paramètres Email")

config["sender_email"] = st.text_input("Expéditeur", config.get("sender_email", ""))
config["password"] = st.text_input("Mot de passe", config.get("password", ""), type="password")

recipients = st.text_area("Destinataires (séparés par virgules)", ", ".join(config.get("recipients", [])))
config["recipients"] = [email.strip() for email in recipients.split(",") if email.strip()]

config["smtp_server"] = st.text_input("Serveur SMTP", config.get("smtp_server", "smtp.example.com"))
config["smtp_port"] = st.number_input("Port SMTP", min_value=1, value=config.get("smtp_port", 587), step=1)

config["push_to_api"] = st.checkbox("📡 Activer envoi vers API Flask", value=config.get("push_to_api", False))

# === Sauvegarde
if st.button("💾 Enregistrer la configuration"):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    st.success("✅ Configuration sauvegardée avec succès.")

# === Aperçu brut
st.markdown("### 🔍 Aperçu brut du fichier")
st.code(json.dumps(config, indent=2), language="json")
