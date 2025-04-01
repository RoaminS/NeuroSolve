# ns007_logger.py


import os
import json
import time
import h5py
import imageio
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from ns001_neuro_np_solver import solve_np_problem
from ns003_visualizer import plot_solution_2d, plot_solution_3d

# === CONFIG
SESSION_DIR = "neurosolv_sessions"
os.makedirs(SESSION_DIR, exist_ok=True)
SAMPLES = 512
NUM_CHANNELS = 19
GIF_FRAME_DIR = os.path.join(SESSION_DIR, "frames")
os.makedirs(GIF_FRAME_DIR, exist_ok=True)

# === Simulated EEG loader or real-time plugin (replace with real in ns006)
def simulate_eeg_segment():
    np.random.seed()
    eeg = np.random.normal(0, 1, size=(SAMPLES, NUM_CHANNELS))
    eeg += np.sin(np.linspace(0, 10*np.pi, SAMPLES)).reshape(-1, 1)  # bruit + pattern
    return eeg

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
def generate_gif_from_frames(frames, outpath, duration=0.5):
    images = [imageio.imread(f) for f in frames]
    imageio.mimsave(outpath, images, duration=duration)
    print(f"🎞️ GIF exporté : {outpath}")

# === LIVE LOGGER LOOP
def live_session(n_loops=5):
    print(f"\n🧠 Lancement session EEG x NP | frames={n_loops}")

    frame_paths = []
    all_logs = []

    for i in range(n_loops):
        print(f"\n📦 Fenêtre EEG #{i+1}")

        # === 1. Simuler un EEG ou charger un vrai
        eeg_seg = simulate_eeg_segment()
        eeg_vector = np.mean(np.abs(eeg_seg), axis=0) * 100
        eeg_vector = eeg_vector.astype(int).tolist()

        target = np.random.randint(50, 100)
        print(f"🔢 EEG vector: {eeg_vector} | 🎯 Target: {target}")

        # === 2. Résolution NP
        solution, meta = solve_np_problem(eeg_vector, target, algo="brute", return_meta=True)
        print(f"✅ Solution: {solution}" if solution else "❌ No solution found.")

        # === 3. Visualisation 2D + sauvegarde
        fig_path = os.path.join(GIF_FRAME_DIR, f"frame_{i:03d}.png")
        plot_solution_2d(eeg_vector, solution, title=f"EEG NP Solve #{i+1}", save_path=fig_path)
        frame_paths.append(fig_path)

        # === 4. Logging JSON
        json_path = os.path.join(SESSION_DIR, f"session_{i+1:03d}.json")
        save_session_json(eeg_vector, solution, meta, json_path)
        all_logs.append(json_path)

        time.sleep(0.5)

    # === 5. Création du GIF final
    gif_path = os.path.join(SESSION_DIR, "np_solver_eeg.gif")
    generate_gif_from_frames(frame_paths, gif_path)

    print(f"\n📁 Session complète enregistrée ({len(all_logs)} fenêtres)")
    print(f"📚 JSON logs : {all_logs[-1]}")
    print(f"🎬 GIF : {gif_path}")

# === MAIN
if __name__ == "__main__":
    live_session(n_loops=12)
