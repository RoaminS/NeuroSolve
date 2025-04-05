from pathlib import Path

# Création du squelette de l'application homepage_3d.py avec fond futuriste + boutons interactifs
homepage_code = """
import streamlit as st
from streamlit.components.v1 import html

st.set_page_config(page_title="NeuroSolve OS", layout="wide")

st.markdown(\"""
    <style>
        body, .stApp {
            background-color: #020c1b;
            color: white;
        }
        .neuro-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding-top: 20px;
        }
        .neuro-image {
            width: 480px;
            margin-bottom: 30px;
            animation: pulse 5s infinite;
        }
        .neuro-buttons {
            display: flex;
            gap: 2rem;
        }
        .neuro-buttons button {
            background-color: #00ffff11;
            border: 1px solid #00ffff;
            color: white;
            padding: 0.8rem 1.8rem;
            font-size: 1.2rem;
            border-radius: 10px;
            cursor: pointer;
        }
        .neuro-buttons button:hover {
            background-color: #00ffff33;
        }
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.03); opacity: 0.95; }
            100% { transform: scale(1); opacity: 1; }
        }
    </style>
\""", unsafe_allow_html=True)

st.markdown('<div class="neuro-container">', unsafe_allow_html=True)
st.image("apps/assets/brain_interface.png", use_column_width=False, width=450)

col1, col2 = st.columns(2)
with col1:
    if st.button("📥 Importer un EEG"):
        st.switch_page("apps/import_predictor.py")
with col2:
    if st.button("⚡ Lancer EEG en direct"):
        st.switch_page("apps/ns014_live_predictor_streamlit.py")

st.markdown('</div>', unsafe_allow_html=True)
"""

# Sauvegarde dans le dossier apps/
apps_dir = Path("apps")
apps_dir.mkdir(exist_ok=True)
(apps_dir / "homepage_3d.py").write_text(homepage_code)

"✅ Fichier homepage_3d.py créé dans apps/. Tu peux maintenant le sélectionner dans streamlit_multipage_autodiscover.py"
