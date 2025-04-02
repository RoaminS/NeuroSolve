# 🧠 NeuroSolve — NP Modules

> Modules de résolution, visualisation, simulation et interprétation des problèmes NP via EEG + IA + Topologie

---

## 📂 Structure des modules

| Fichier | Fonction principale |
|--------|---------------------|
| `ns001_neuro_np_solver.py` | Solveur NP de base (subset sum / vertex cover) |
| `ns002_eeg_guided_solver.py` | Résolution guidée par heuristique EEG |
| `ns003_visualizer.py` | Visualisation 2D/3D/graphml des explorations |
| `ns004_solver_xai.py` | Colorisation XAI des nœuds du graphe EEG |
| `ns005_eeg_real_loader.py` | Extraction et vectorisation EEG depuis HDF5 |
| `ns006_live_solver.py` | Prototype d'exécution NP sur EEG live |
| `ns007_logger.py` | Logging EEG + solutions + frames (GIF, JSON) |
| `ns008_riemann.py` | Analyse EEG ↔ plan complexe (hypothèse Riemann) |
| `ns008_riemann_explorer.py` | Mapping spectral EEG ↔ zéros ζ(s) |
| `ns009_topo_wavelet.py` | Décomposition ondelettes + TDA EEG |
| `ns010_topo_mapper.py` | Mapping avancé topologique des signaux EEG |
| `ns011_benchmark.py` | Comparaison brute vs guided vs random |
| `ns012_cycle_detector.py` | Détection de cycles cognitifs (Betti-1) |
| `ns013_model_fusion.py` | Fusion TDA + ondelettes → Modèle ML |
| `ns014_live_predictor.py` | Prédiction temps réel EEG + SHAP live |
| `ns015_shap_live.py` | Explication XAI frame-by-frame (SHAP) |

---

## 🚀 Exécution rapide

# Résolution NP de base
```
python ns001_neuro_np_solver.py --json path/to/eeg_vectors.json
```

# Prédiction guidée par EEG
```
python ns002_eeg_guided_solver.py
```

# Visualisation 2D/3D des chemins
```
python ns003_visualizer.py --path eeg_guided_path.json
```

# Entraînement en live EEG
```
python ns006_live_solver.py
```

🧠 Objectif du dossier NP/

```
Ce sous-module vise à :

Modéliser la résolution de problèmes NP via signaux EEG

Visualiser les stratégies d'exploration heuristique

Analyser topologiquement le comportement cognitif

Fusionner données biologiques et structures NP-complexes

Offrir une base de réflexion IA + cerveau sur des problèmes fondamentaux
```

📊 Fichiers générés typiques
```
Extension	Contenu
.json	Vecteurs EEG, logs, SHAP
.png / .html	Visualisations topologiques / graphe
.graphml	Export de graphes EEG (Gephi, Cytoscape)
.csv	Résultats de prédictions, logs
.gif	Visualisation animée des sessions EEG-NP
```

👥 Contributions
```
🧠 Kocupyr Romain

🤖 multi_gpt_api

🧬 Collaborateurs bienvenus (neuro, math, IA, dev)
```


📄 Licence
```
Ce projet est sous licence Creative Commons BY-NC-SA 4.0
Usage libre non commercial — partage & attribution requis.
```
