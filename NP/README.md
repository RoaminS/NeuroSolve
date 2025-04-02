# 🧠 NeuroSolve — NP Modules

> Modules de résolution, visualisation, simulation et interprétation des problèmes NP via EEG + IA + Topologie

---

## 📂 Structure des modules

| Fichier | Fonction principale |
|--------|---------------------|
| `ns001_neuro_np_solver.py` | Solveur NP de base (subset sum / vertex cover) |
| `ns002_eeg_guided_solver.py` | Stratégie guidée par vecteurs EEG simulés ou réels |
| `ns003_visualizer.py` | Visualisation des explorations : graphe 2D/3D/GraphML |
| `ns004_solver_xai.py` | Interprétabilité XAI des chemins de résolution |
| `ns005_eeg_real_loader.py` | Chargement EEG réel depuis HDF5 pour NP Solver |
| `ns006_live_solver.py` | Solveur NP en temps réel depuis un flux EEG |
| `ns007_logger.py` | Enregistrement de sessions EEG + log + GIF |
| `ns008_riemann_explorer.py` | Mapping EEG ↔ zéros de la fonction zêta de Riemann |
| `ns009_topo_wavelet.py` | Analyse topologique + ondelettes par canal EEG |
| `ns00X_topo_mapper.py` | Fusion TDA multi-échelles & cartographie EEG |
| `ns999_benchmark.py` | Comparaison brute vs guided vs random avec stats & dashboard |

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

Ce sous-module vise à :

Modéliser la résolution de problèmes NP via signaux EEG

Visualiser les stratégies d'exploration heuristique

Analyser topologiquement le comportement cognitif

Fusionner données biologiques et structures NP-complexes

Offrir une base de réflexion IA + cerveau sur des problèmes fondamentaux

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
