#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ns003_visualizer.py
====================

🎥 Visualisation 3D et graphe de l'exploration du solveur NP guidé par EEG
Auteur : Kocupyr Romain
Dev    : multi_gpt_api, Grok3
Licence : Creative Commons BY-NC-SA 4.0
"""

import os
import json
import numpy as np
import plotly.graph_objs as go
import networkx as nx
from argparse import ArgumentParser

# === PARAMS
parser = ArgumentParser()
parser.add_argument("--path", default="eeg_guided_path.json", help="Chemin vers le fichier JSON du solveur")
parser.add_argument("--outdir", default="ns003_outputs", help="Dossier de sortie")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# === Load
with open(args.path, "r") as f:
    path = json.load(f)

print(f"📂 Chemin chargé : {args.path}")
print(f"🔢 Total étapes : {len(path)}")

# === Créer graphe
G = nx.DiGraph()
last_seen = {}

for i, node in enumerate(path):
    subset = tuple(sorted(node["subset"]))
    score = node["score"]
    current_sum = node["sum"]

    G.add_node(i, label=str(subset), score=score, sum=current_sum)

    # Ajoute arête depuis un état précédent si connu
    parent_key = tuple(sorted(node["subset"][:-1])) if node["subset"] else None
    if parent_key in last_seen:
        G.add_edge(last_seen[parent_key], i)
    last_seen[subset] = i

# === 3D Plot
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
    title="Exploration guidée EEG (Subset Sum)",
    scene=dict(
        xaxis_title="Somme Partielle",
        yaxis_title="Score EEG",
        zaxis_title="Étape",
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

out_path = os.path.join(args.outdir, "eeg_guided_solver_3d.html")
fig.write_html(out_path)
print(f"✅ Visualisation HTML générée : {out_path}")
