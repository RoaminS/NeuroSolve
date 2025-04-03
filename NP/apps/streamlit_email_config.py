"""
streamlit_email_config.py

Licence : Creative Commons BY-NC-SA 4.0

Auteurs : 
    - Kocupyr Romain (chef de projet) : rkocupyr@gmail.com
    - GPT multi_gpt_api (OpenAI)
"""

import streamlit as st
import json

CONFIG_PATH = "notifier_config.json"

st.title("📬 Configuration Notifications NeuroSolve")

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        config = json.load(f)
else:
    config = {
        "emails": [],
        "phones": [],
        "from": "",
        "password": ""
    }

emails = st.text_area("📧 Emails de réception (1 par ligne)", "\n".join(config["emails"])).splitlines()
phones = st.text_area("📱 Numéros de téléphone (optionnel)", "\n".join(config["phones"])).splitlines()
sender = st.text_input("📤 Adresse email expéditrice", config["from"])
password = st.text_input("🔑 Mot de passe (application)", config["password"], type="password")

if st.button("💾 Sauvegarder"):
    config = {
        "emails": emails,
        "phones": phones,
        "from": sender,
        "password": password
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    st.success("✅ Configuration sauvegardée.")
