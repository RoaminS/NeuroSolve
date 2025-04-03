'''
Auteur : Kocupyr Romain
Dev    : multi_gpt_api
Licence : CC BY-NC-SA 4.0
'''

# 🧠 NeuroSolve — Streamlit Web Dashboard

Bienvenue dans l’interface **Streamlit** de NeuroSolve :  
Un système visuel complet pour **analyser, interpréter et surveiller** les prédictions EEG en live via IA, XAI (SHAP) et alertes cognitives.

---

## 🚀 Fonctionnalités

✅ Affichage temps réel :
- 🎞️ GIF EEG (`prediction_live.gif`)
- 🔍 Interprétation SHAP (`shap_live_frame.png`)
- 📊 Table des prédictions EEG (`ns014_predictions.csv`)
- ⚠️ Alertes déclenchées (`alerts_detected.json`)

✅ Interaction :
- 📤 **Envoi automatique** des données vers une API Flask
- 📬 Feedback API (statut + réponse JSON)

✅ Publication :
- 💻 Local
- ☁️ [Streamlit.io](https://streamlit.io/cloud)  
- 🌐 [Hugging Face Spaces](https://huggingface.co/spaces)

---

## 🧰 Installation locale

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

📁 Fichiers attendus dans le dossier :

Fichier	Description

prediction_live.gif	=> Animation EEG des prédictions

shap_live_frame.png	=> Explication SHAP en image

ns020_predictions.csv	=> Table des prédictions EEG

alerts_detected.json => Liste des alertes cognitives

predictions_log.json =>	Log brut des prédictions


🌐 Déploiement Cloud:

Crée un Space → type Streamlit

Ajoute :

streamlit_app.py

requirements.txt

Tous les fichiers .csv / .json démo

Tu peux créer un bouton dans ton README :


🔹 Streamlit.io Cloud:

Va sur streamlit.io/cloud

Connecte ton repo GitHub

Lance le fichier streamlit_app.py


🧠 Capture d’écran

✍️ Auteur
Kocupyr Romain

CoDev : [multi_gpt_api]

Projet : NeuroSolve

📄 Licence
Creative Commons BY-NC-SA 4.0
Utilisation libre à but non commercial, attribution obligatoire.

