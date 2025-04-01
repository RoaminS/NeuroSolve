#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ns003_visualizer.py
====================

🎥 Visualisation 3D (Plotly) et 2D (Matplotlib) du chemin exploré par EEG dans le problème NP (subset sum) + animation .gif + export .graphml
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
import imageio
from tqdm import tqdm
from argparse import ArgumentParser

# === PARAMS
parser = ArgumentParser()
parser.add_argument("--path", default="eeg_guided_path.json", help="Fichier JSON EEG-guidé")
parser.add_argument("--outdir", default="ns003_outputs", help="Dossier de sortie")
parser.add_argument("--mode", choices=["2d", "3d", "both"], default="both", help="Type de visualisation")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

# === LOAD EEG-PATH
with open(args.path, "r") as f:
    path = json.load(f)
print(f"📂 Fichier : {args.path} | Étapes : {len(path)}")

# === GRAPH CONSTRUCTION
G = nx.DiGraph()
last_seen = {}

for i, node in enumerate(path):
    subset = tuple(sorted(node["subset"]))
    score = node["score"]
    total = node["sum"]
    G.add_node(i, label=str(subset), score=score, sum=total)

    parent = tuple(sorted(node["subset"][:-1])) if node["subset"] else None
    if parent in last_seen:
        G.add_edge(last_seen[parent], i)
    last_seen[subset] = i

# === 2D Matplotlib
if args.mode in ["2d", "both"]:
    x_vals = [G.nodes[n]["sum"] for n in G.nodes]
    y_vals = [G.nodes[n]["score"] for n in G.nodes]

    # STATIC PLOT
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(x_vals, y_vals, c=y_vals, cmap='plasma', s=30)
    plt.title("Exploration EEG-guidée - Vue 2D")
    plt.xlabel("Somme partielle")
    plt.ylabel("Score EEG")
    plt.colorbar(sc)
    plt.grid(True)
    plt.tight_layout()
    path_2d = os.path.join(args.outdir, "eeg_solver_2d.png")
    plt.savefig(path_2d)
    print(f"✅ Image 2D sauvegardée : {path_2d}")

    # GIF ANIMATION (2D)
    print("🎥 Création du GIF 2D en cours...")
    gif_frames = []
    for step in tqdm(range(1, len(x_vals))):
        plt.figure(figsize=(8, 5))
        plt.scatter(x_vals[:step], y_vals[:step], c=y_vals[:step], cmap='plasma', s=20)
        plt.xlabel("Somme")
        plt.ylabel("Score EEG")
        plt.title("EEG-guided solving (2D)")
        plt.xlim(min(x_vals), max(x_vals))
        plt.ylim(min(y_vals), max(y_vals))
        plt.grid(True)
        frame_path = os.path.join(args.outdir, f"frame2d_{step:03}.png")
        plt.savefig(frame_path)
        plt.close()
        gif_frames.append(imageio.imread(frame_path))
        os.remove(frame_path)
    gif_path_2d = os.path.join(args.outdir, "eeg_solver_2d.gif")
    imageio.mimsave(gif_path_2d, gif_frames, fps=10)
    print(f"🎬 GIF 2D sauvegardé : {gif_path_2d}")

# === 3D Plotly
if args.mode in ["3d", "both"]:
    x = [G.nodes[n]["sum"] for n in G.nodes]
    y = [G.nodes[n]["score"] for n in G.nodes]
    z = list(G.nodes)

    node_trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers+text",
        marker=dict(size=4, color=y, colorscale="Viridis", colorbar=dict(title="EEG Score")),
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
            line=dict(color='gray', width=2),
            hoverinfo='none'
        ))

    fig = go.Figure(data=edge_trace + [node_trace])
    fig.update_layout(
        title="Exploration 3D - EEG guided path",
        scene=dict(xaxis_title="Somme", yaxis_title="Score EEG", zaxis_title="Étapes"),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    html_3d = os.path.join(args.outdir, "eeg_solver_3d.html")
    fig.write_html(html_3d)
    print(f"✅ HTML 3D sauvegardé : {html_3d}")

    # === GIF 3D ROTATION
    print("🎥 Création du GIF 3D en rotation...")
    frames = []
    for angle in tqdm(range(0, 360, 6)):
        camera = dict(eye=dict(x=np.cos(np.radians(angle))*2, y=np.sin(np.radians(angle))*2, z=1.2))
        fig.update_layout(scene_camera=camera)
        tmp_img = os.path.join(args.outdir, f"frame3d_{angle:03}.png")
        fig.write_image(tmp_img, engine="kaleido")
        frames.append(imageio.imread(tmp_img))
        os.remove(tmp_img)
    gif_3d = os.path.join(args.outdir, "eeg_solver_3d.gif")
    imageio.mimsave(gif_3d, frames, fps=12)
    print(f"🎬 GIF 3D sauvegardé : {gif_3d}")

# === EXPORT GRAPHML
graphml_path = os.path.join(args.outdir, "eeg_solver.graphml")
for n in G.nodes:
    G.nodes[n]["id"] = n
for u, v in G.edges:
    G.edges[u, v]["weight"] = round(abs(G.nodes[u]["score"] - G.nodes[v]["score"]), 3)
nx.write_graphml(G, graphml_path)
print(f"🧠 Export GraphML : {graphml_path}")
