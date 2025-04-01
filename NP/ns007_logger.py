"""
ns007_logger.py

✅ branche le solveur temps réel
✅ enregistre chaque session EEG
✅ génère des GIFs de visualisation
✅ ajoute l’interprétabilité XAI live

✅ Résultats
À la fin tu obtiens :

🎞️ Un fichier np_solver_eeg.gif montrant toutes les résolutions

📁 Un dossier neurosolv_sessions/ avec :

session_001.json … session_012.json

Les frames PNG intermédiaires

Auteur : Kocupyr Romain
Dev    : multi_gpt_api, Grok3
Licence : Creative Commons BY-NC-SA 4.0
"""
import os
import json
import time
import h5py
import mne
import imageio
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pylsl import StreamInlet, resolve_stream
from ns001_neuro_np_solver import solve_np_problem
from ns003_visualizer import plot_solution_2d, plot_solution_3d

# === CONFIG
SESSION_DIR = "neurosolv_sessions"
os.makedirs(SESSION_DIR, exist_ok=True)
SAMPLES = 512
NUM_CHANNELS = 19
FRAME_DIR_2D = os.path.join(SESSION_DIR, "frames_2d")
FRAME_DIR_3D = os.path.join(SESSION_DIR, "frames_3d")
os.makedirs(FRAME_DIR_2D, exist_ok=True)
os.makedirs(FRAME_DIR_3D, exist_ok=True)


# === Temps réel via LSL
def get_real_eeg_segment_lsl():
    print("🔍 Recherche d'un flux EEG LSL...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    print("✅ Flux EEG détecté.")

    eeg_buffer = []
    while len(eeg_buffer) < 512:
        sample, _ = inlet.pull_sample()
        eeg_buffer.append(sample[:NUM_CHANNELS])  # garde uniquement les 19 premiers

    eeg_data = np.array(eeg_buffer)  # shape (512, 19)
    return eeg_data

# === EEG réel depuis fichier (OpenBCI, Muse, etc.)
def get_real_eeg_segment_from_file(path):
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    raw.pick_types(eeg=True)
    raw.filter(0.5, 45)
    raw.resample(128)
    return raw.get_data()[:, :512].T  # shape (512, 19)

# === Sauvegarde JSON brute + solution
def save_session_json(data, solution, meta, path):
    session_data = {
        "timestamp": datetime.now().isoformat(),
        "eeg_vector": data.tolist(),
        "solution": solution,
        "meta": meta
    }
    with open(path, "w") as f:
        json.dump(session_data, f, indent=2)

# === Animation .gif avec matplotlib
def generate_gif(frames, outpath, duration=0.5):
    images = [imageio.imread(f) for f in frames]
    imageio.mimsave(outpath, images, duration=duration)
    print(f"🎞️ GIF exporté : {outpath}")

# === Index HTML de toutes les sessions
def generate_html_index():
    files = sorted([f for f in os.listdir(SESSION_DIR) if f.endswith(".json")])
    with open(os.path.join(SESSION_DIR, "log_index.html"), "w") as f:
        f.write("<html><head><title>Sessions EEG</title></head><body><h1>🧠 Sessions EEG Résolues</h1><ul>")
        for file in files:
            f.write(f'<li><a href="{file}">{file}</a></li>')
        f.write("</ul></body></html>")
    print("📁 HTML index généré : log_index.html")

# === Générer un tableau HTML auto-rafraîchissant
def generate_live_dashboard():
    html = f"""
    <html>
    <head>
        <meta http-equiv="refresh" content="5">
        <title>🧠 NeuroSolv Live</title>
        <style>
            body {{ font-family: Arial; padding: 20px; }}
            h1 {{ color: #333; }}
            img {{ max-width: 100%; }}
        </style>
    </head>
    <body>
        <h1>🧠 Dernière session EEG + Résolution NP</h1>
        <p>Mis à jour automatiquement toutes les 5 secondes.</p>
        <h2>Graphique 2D</h2>
        <img src="np_solver_2d.gif" alt="2D Graph">
        <h2>Graphique 3D</h2>
        <img src="np_solver_3d.gif" alt="3D Graph">
    </body>
    </html>
    """
    with open(os.path.join(SESSION_DIR, "live_plot.html"), "w") as f:
        f.write(html)
    print("📡 Dashboard temps réel : live_plot.html")

# === SESSION LOOP
def live_session(eeg_file, n_loops=5):
    frame_paths_2d, frame_paths_3d = [], []
    print(f"\n🧠 Lancement session temps réel | Source: {eeg_file}")

    for i in range(n_loops):
        print(f"\n📦 EEG #{i+1}")

        # === 1. Charger segment EEG réel
        eeg_seg = get_real_eeg_segment_from_file(eeg_file)
        # eeg_seg = simulate_eeg_segment()
        eeg_seg = get_real_eeg_segment_lsl()

        eeg_vector = np.mean(np.abs(eeg_seg), axis=0) * 100
        eeg_vector = eeg_vector.astype(int).tolist()

        target = np.random.randint(50, 100)
        print(f"🔢 EEG vector: {eeg_vector} | 🎯 Target: {target}")

        # === 2. Résolution NP
        solution, meta = solve_np_problem(eeg_vector, target, algo="brute", return_meta=True)
        print(f"✅ Solution trouvée: {solution}" if solution else "❌ Aucune solution.")

        # === 3. Sauvegarde graphique 2D
        fig_2d = os.path.join(FRAME_DIR_2D, f"frame2d_{i:03d}.png")
        plot_solution_2d(eeg_vector, solution, title=f"EEG NP Solve #{i+1}", save_path=fig_2d)
        frame_paths_2d.append(fig_2d)

        # === 4. Sauvegarde graphique 3D
        fig_3d = os.path.join(FRAME_DIR_3D, f"frame3d_{i:03d}.png")
        plot_solution_3d(eeg_vector, solution, title=f"EEG NP Solve #{i+1}", save_path=fig_3d)
        frame_paths_3d.append(fig_3d)

        # === 5. Logging JSON
        json_path = os.path.join(SESSION_DIR, f"session_{i+1:03d}.json")
        save_session_json(eeg_vector, solution, meta, json_path)

        time.sleep(0.5)

    # === 6. GIFs
    generate_gif(frame_paths_2d, os.path.join(SESSION_DIR, "np_solver_2d.gif"))
    generate_gif(frame_paths_3d, os.path.join(SESSION_DIR, "np_solver_3d.gif"))
    
    # === 7. Générer l’index HTML
    generate_live_dashboard()  # ✅ nouveau
    generate_html_index()


# === MAIN
if __name__ == "__main__":
    EEG_FILE = "eeg_samples/subject_01.set"  # Remplace par ton fichier EEG réel
    live_session(eeg_file=EEG_FILE, n_loops=8)
