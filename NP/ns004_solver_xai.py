#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ns004_solver_xai.py

🧠 Interprétabilité et visualisation XAI du graphe EEG-guidé (NP Solver)
Auteur : Kocupyr Romain
Dev    : multi_gpt_api, Grok3
Licence : Creative Commons BY-NC-SA 4.0
"""

import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
import plotly.graph_objs as go

# === ARGS
parser = ArgumentParser()
parser.add_argument("--path", default="eeg_guided_path.json", help="Fichier JSON EEG-guided")
parser.add_argument("--outdir", default="ns004_outputs", help="Dossier de sortie")
parser.add_argument("--threshold", type=float, default=0.5, help="Seuil de mise en évidence des EEG importance")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

# === LOAD PATH
with open(args.path, "r") as f:
    path = json.load(f)

print(f"📂 Fichier EEG-guided : {args.path} | Nœuds : {len(path)}")

# === BUILD GRAPH
G = nx.DiGraph()
last_seen = {}

for i, node in enumerate(path):
    subset = tuple(sorted(node["subset"]))
    score = node["score"]
    s = node["sum"]
    G.add_node(i, label=str(subset), score=score, sum=s)
    parent = tuple(sorted(node["subset"][:-1])) if node["subset"] else None
    if parent in last_seen:
        G.add_edge(last_seen[parent], i)
    last_seen[subset] = i

# === XAI: Color mapping par score
scores = np.array([G.nodes[n]["score"] for n in G.nodes])
min_score, max_score = scores.min(), scores.max()
norm_scores = (scores - min_score) / (max_score - min_score + 1e-8)

# === Matplotlib Static Color Map
plt.figure(figsize=(12, 6))
cmap = plt.cm.get_cmap("viridis")
colors = [cmap(val) for val in norm_scores]
nx.draw_spring(G, node_color=colors, with_labels=False, node_size=50, edge_color='gray')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_score, vmax=max_score))
sm.set_array([])
plt.colorbar(sm, label="EEG Score")
plt.title("Graphe EEG-guidé coloré par importance")
plt.tight_layout()
plt.savefig(os.path.join(args.outdir, "graph_xai_matplotlib.png"))
print("✅ Graph statique XAI sauvegardé.")

# === Export CSV
df = pd.DataFrame([
    {
        "ID": n,
        "Subset": G.nodes[n]["label"],
        "Score": G.nodes[n]["score"],
        "Sum": G.nodes[n]["sum"],
        "ImportanceNorm": norm_scores[n]
    } for n in G.nodes
])
df.to_csv(os.path.join(args.outdir, "xai_scores_table.csv"), index=False)
print("📊 Export CSV XAI terminé.")

# === Interactive Plotly HTML
edge_trace = []
for u, v in G.edges:
    edge_trace.append(go.Scatter(
        x=[G.nodes[u]["sum"], G.nodes[v]["sum"], None],
        y=[G.nodes[u]["score"], G.nodes[v]["score"], None],
        mode="lines",
        line=dict(width=1, color="gray"),
        hoverinfo="none"
    ))

node_trace = go.Scatter(
    x=[G.nodes[n]["sum"] for n in G.nodes],
    y=[G.nodes[n]["score"] for n in G.nodes],
    mode="markers+text",
    marker=dict(
        size=6,
        color=norm_scores,
        colorscale="Viridis",
        colorbar=dict(title="EEG Score"),
        showscale=True
    ),
    text=[G.nodes[n]["label"] for n in G.nodes],
    hoverinfo="text"
)

layout = go.Layout(
    title="EEG XAI Graph - Interactive",
    xaxis=dict(title="Somme partielle"),
    yaxis=dict(title="Score EEG"),
    showlegend=False,
    margin=dict(l=20, r=20, t=40, b=20)
)

fig = go.Figure(data=edge_trace + [node_trace], layout=layout)
html_path = os.path.join(args.outdir, "graph_xai_interactive.html")
fig.write_html(html_path)
print(f"✅ Graphe HTML interactif exporté : {html_path}")

# === Seuil visuel
highlight_nodes = [n for n in G.nodes if norm_scores[n] >= args.threshold]
print(f"🔍 Nœuds surpassant le seuil {args.threshold} : {len(highlight_nodes)}")
with open(os.path.join(args.outdir, "highlighted_nodes.json"), "w") as f:
    json.dump(highlight_nodes, f, indent=2)

print("🎯 Analyse XAI EEG-guidée terminée.")
