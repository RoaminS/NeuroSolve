#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ns003_visualizer.py
====================

🎥 Visualisation 3D (Plotly) et 2D (Matplotlib) du chemin exploré par EEG dans le problème NP (subset sum)
Auteur : Kocupyr Romain
Dev    : multi_gpt_api, Grok3
Licence : Creative Commons BY-NC-SA 4.0
"""

import os
import json
import numpy as np
import plotly.graph_objs as go
import networkx as nx
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# === PARAMS
parser = ArgumentParser()
parser.add_argument("--path", default="eeg_guided_path.json", help="Fichier JSON du chemin EEG guidé")
parser.add_argument("--outdir", default="ns003_outputs", help="Dossier de sortie")
parser.add_argument("--mode", choices=["3d", "2d", "both"], default="both", help="Mode de visualisation")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# === LOAD PATH
with open(args.path, "r") as f:
    path = json.load(f)

print(f"📂 Fichier chargé : {args.path}")
print(f"🧠 Étapes totales : {len(path)}")

# === CONSTRUIRE GRAPHE
G = nx.DiGraph()
last_seen = {}

for i, node in enumerate(path):
    subset = tuple(sorted(node["subset"]))
    score = node["score"]
    current_sum = node["sum"]

    G.add_node(i, label=str(subset), score=score, sum=current_sum)

    # Lien vers l’état parent
    parent_key = tuple(sorted(node["subset"][:-1])) if node["subset"] else None
    if parent_key in last_seen:
        G.add_edge(last_seen[parent_key], i)
    last_seen[subset] = i

# === 2D MATPLOTLIB
if args.mode in ["2d", "both"]:
    x_vals = [G.nodes[n]["sum"] for n in G.nodes]
    y_vals = [G.nodes[n]["score"] for n in G.nodes]

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(x_vals, y_vals, c=y_vals, cmap='viridis', s=20)
    plt.title("Exploration guidée EEG - Vue 2D")
    plt.xlabel("Somme partielle")
    plt.ylabel("Score EEG")
    plt.colorbar(sc, label="Score EEG")
    plt.grid(True)
    plt.tight_layout()
    out_2d = os.path.join(args.outdir, "eeg_guided_solver_2d.png")
    plt.savefig(out_2d)
    print(f"✅ Graphique 2D sauvegardé : {out_2d}")

# === 3D PLOTLY
if args.mode in ["3d", "both"]:
    x = [G.nodes[n]["sum"] for n in G.nodes]
    y = [G.nodes[n]["score"] for n in G.nodes]
    z = list(G.nodes)

    node_trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers+text",
        marker=dict(size=4, color=y, colorscale="Viridis", colorbar=dict(title="Score EEG")),
        text=[G.nodes[n]["label"] for n in G.nodes],
        hoverinfo="text"
    )

    edge_trace = []
    for u, v in G.edges:
        x0, y0, z0 = G.nodes[u]["sum"], G.nodes[u]["score"], u
        x1, y1, z1 = G.nodes[v]["sum"], G.nodes[v]["score"], v
        edge_trace.append(go.Scatter3d(
            x=[x0, x1, None],
            y=[y0, y1, None],
            z=[z0, z1, None],
            mode='lines',
            line=dict(color='gray', width=1),
            hoverinfo='none'
        ))

    fig = go.Figure(data=edge_trace + [node_trace])
    fig.update_layout(
        title="🧠 Exploration NP guidée EEG - Vue 3D",
        scene=dict(
            xaxis_title="Somme partielle",
            yaxis_title="Score EEG",
            zaxis_title="Étape"),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    out_html = os.path.join(args.outdir, "eeg_guided_solver_3d.html")
    fig.write_html(out_html)
    print(f"✅ Visualisation 3D HTML : {out_html}")
