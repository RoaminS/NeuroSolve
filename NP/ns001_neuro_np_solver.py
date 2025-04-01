#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ns001_neuro_np_solver.py — NP Solver EEG-driven

📥 Lit des vecteurs EEG (via JSON export ns005)
🧩 Génère un graphe de contraintes (clique/couverture)
🧠 Résout une tâche NP (Vertex Cover) avec heuristique gloutonne
🧠 Affiche les performances par EEG

Auteur  : Kocupyr Romain
Dev     : multi_gpt_api, Grok3
Licence : Creative Commons BY-NC-SA 4.0
Commande: python ns001_neuro_np_solver.py --json ns005_output/eeg_vectors_ns001.json --show_graph

"""

import json
import time
import argparse
import numpy as np
import networkx as nx

# === PARAMÈTRES
parser = argparse.ArgumentParser()
parser.add_argument("--json", required=True, help="Fichier EEG JSON (depuis ns005)")
parser.add_argument("--show_graph", action="store_true", help="Afficher le graphe généré")
args = parser.parse_args()

# === CHARGEMENT JSON
with open(args.json, "r") as f:
    eeg_data = json.load(f)

print(f"📂 EEGs chargés : {len(eeg_data)}")

# === SOLVEUR NP : Couverture de sommets gloutonne
def solve_vertex_cover(G):
    cover = set()
    edges = set(G.edges())
    while edges:
        (u, v) = edges.pop()
        cover.add(u)
        cover.add(v)
        edges = {(a, b) for (a, b) in edges if a != u and b != u and a != v and b != v}
    return cover

# === MAPPING EEG → GRAPHE
def eeg_to_graph(vec, threshold=0.6, max_nodes=20):
    """
    Simplification : EEG → graphe non dirigé
    - Les 267 features sont utilisées comme noeuds pondérés
    - Les arêtes sont créées entre les top features les plus actives
    """
    G = nx.Graph()
    vec = np.array(vec[-267:])  # On prend uniquement la partie "features"

    # Top-n indices
    top_idx = np.argsort(-np.abs(vec))[:max_nodes]
    for i in top_idx:
        G.add_node(i, value=round(vec[i], 4))

    for i in range(len(top_idx)):
        for j in range(i+1, len(top_idx)):
            vi, vj = vec[top_idx[i]], vec[top_idx[j]]
            sim = 1.0 - abs(vi - vj) / (abs(vi) + abs(vj) + 1e-5)
            if sim > threshold:
                G.add_edge(top_idx[i], top_idx[j], weight=round(sim, 3))
    return G

# === BOUCLE EEGs
for entry in eeg_data:
    subj = entry.get("subject", "unknown")
    score = entry.get("score", 0)
    vec = entry.get("vector", [])
    label = entry.get("label", None)

    G = eeg_to_graph(vec)

    if args.show_graph:
        print(f"🔍 Graphe EEG généré : {len(G.nodes)} nœuds, {len(G.edges)} arêtes")

    t0 = time.time()
    solution = solve_vertex_cover(G)
    t1 = time.time()

    print(f"\n🧠 Sujet : {subj}")
    print(f"📊 Score EEG : {score:.4f} | Label = {label}")
    print(f"🧩 Solution (vertex cover) : {sorted(list(solution))}")
    print(f"⏱️ Temps de résolution : {(t1 - t0)*1000:.2f} ms")

print("\n✅ Traitement EEG → NP terminé.")
