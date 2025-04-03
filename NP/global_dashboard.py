"""
global_dashboard.py

Licence : Creative Commons BY-NC-SA 4.0
Auteurs : 
    - Kocupyr Romain (chef de projet) : rkocupyr@gmail.com
    - GPT multi_gpt_api (OpenAI)
"""

import plotly.express as px
import plotly.graph_objects as go
import pdfkit
import streamlit as st
import pandas as pd
import numpy as np
import os


st.markdown("---")
st.markdown("### 📊 Corrélations inter-sessions")

GLOBAL_SUMMARY = "sessions_summary.csv"
if os.path.exists(GLOBAL_SUMMARY):
    df_global = pd.read_csv(GLOBAL_SUMMARY)

    # === Annoter les sessions "critiques"
    df_global["status"] = df_global["alert_rate"].apply(
        lambda x: "🟥 Anormale" if x > 0.3 else "🟩 Normale"
    )
    st.markdown("### 🧠 Sessions annotées")
    st.dataframe(df_global[["session_folder", "alert_rate", "status"]])


    # === Heatmap Duration vs Alert Rate
    st.markdown("#### 🔥 Heatmap Durée (s) vs Taux d’alerte (%)")
    df_heat = df_global.copy()
    df_heat["alert_rate_%"] = df_heat["alert_rate"] * 100
    fig_heat = px.density_heatmap(
        df_heat,
        x="duration_sec",
        y="alert_rate_%",
        nbinsx=20,
        nbinsy=20,
        color_continuous_scale="Inferno",
        labels={"duration_sec": "Durée session (s)", "alert_rate_%": "Taux d'alerte (%)"},
        title="🧠 Heatmap durée vs taux d'alerte"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # === Scatterplot + regression
    st.markdown("#### 📈 Corrélation Durée ↔️ Taux d’alerte")
    fig_scatter = px.scatter(
        df_global,
        x="duration_sec",
        y="alert_rate",
        size="nb_frames",
        color="avg_prob_class_1",
        hover_name="session_folder",
        labels={"duration_sec": "Durée (s)", "alert_rate": "Taux d’alerte"},
        title="🧠 Scatter des sessions EEG"
    )
    fig_scatter.add_traces(go.Scatter(
        x=df_global["duration_sec"],
        y=np.poly1d(np.polyfit(df_global["duration_sec"], df_global["alert_rate"], 1))(df_global["duration_sec"]),
        mode='lines',
        name='Trend linéaire',
        line=dict(dash='dash', color='blue')
    ))
    st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.warning("Le fichier `sessions_summary.csv` n’existe pas encore.")


pdfkit.from_file("sessions_summary_plot.html", "summary_report.pdf")
print("📥 Rapport PDF généré : summary_report.pdf")


# === Génération PDF (optionnel)
html_report = "sessions_summary_plot.html"
pdf_report = "summary_report.pdf"

if os.path.exists(html_report):
    try:
        pdfkit.from_file(html_report, pdf_report)
        st.success(f"📥 Rapport PDF généré : {pdf_report}")

        with open(pdf_report, "rb") as f:
            st.download_button("⬇️ Télécharger le rapport PDF", f, file_name=pdf_report)
    except Exception as e:
        st.error(f"❌ Erreur génération PDF : {e}")
else:
    st.warning("📄 sessions_summary_plot.html introuvable.")

