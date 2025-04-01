#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ns002_eeg_guided_solver.py
===========================

🧠 NeuroSolv Module 002 — Subset Sum solver guidé par une heuristique EEG-like
Auteur : Kocupyr Romain
Dev    : multi_gpt_api, Grok3
Licence : Creative Commons BY-NC-SA 4.0
"""

import numpy as np
import json
import time
import random

# ============================================================================
# === Simulation EEG / Heuristique
# ============================================================================
def simulate_eeg_weights(n):
    """
    Simule un vecteur de poids EEG-like pour chaque élément.
    Ex : l’attention ou la confiance cérébrale sur une décision.
    """
    weights = np.random.rand(n)
    weights /= weights.sum()  # Normalisation
    return weights


# ============================================================================
# === Subset Sum Solver avec guidage
# ============================================================================
def eeg_guided_subset_sum(numbers, target, eeg_weights, max_steps=5000, verbose=False):
    """
    Résout le subset sum avec une stratégie guidée (EEG ou heuristique).
    """
    steps = 0
    solutions = []
    path_log = []

    def explore(partial, idx, score):
        nonlocal steps
        if steps >= max_steps:
            return
        steps += 1

        current_sum = sum(partial)
        path_log.append({
            "step": steps,
            "subset": partial,
            "sum": current_sum,
            "score": score
        })

        if current_sum == target:
            solutions.append(partial)
            return
        if current_sum > target or idx >= len(numbers):
            return

        # EEG-guided : on trie les choix suivants par poids EEG
        choices = [(idx, eeg_weights[idx])]
        if idx + 1 < len(numbers):
            choices.append((idx + 1, eeg_weights[idx + 1]))
        choices.sort(key=lambda x: -x[1])  # descendante

        for next_idx, _ in choices:
            explore(partial + [numbers[next_idx]], next_idx + 1, score + eeg_weights[next_idx])
            explore(partial, next_idx + 1, score)

    explore([], 0, 0.0)
    return {
        "solutions": solutions,
        "steps": steps,
        "path": path_log
    }

# ============================================================================
# === MAIN TEST
# ============================================================================
if __name__ == "__main__":
    numbers = [3, 34, 4, 12, 5, 2]
    target = 9
    eeg_weights = simulate_eeg_weights(len(numbers))

    print("🎯 Problème : subset sum")
    print(f"  Nombres : {numbers}")
    print(f"  Cible   : {target}")
    print(f"🧠 Poids EEG simulés : {np.round(eeg_weights, 3)}")

    t0 = time.time()
    result = eeg_guided_subset_sum(numbers, target, eeg_weights, max_steps=1000)
    duration = time.time() - t0

    print(f"\n✅ Résolu en {result['steps']} étapes ({duration:.3f} sec)")
    print("🔍 Solutions trouvées :")
    for sol in result['solutions']:
        print(f"  ➤ {sol} = {sum(sol)}")

    # === Export JSON
    with open("eeg_guided_path.json", "w") as f:
        json.dump(result["path"], f, indent=2)
    print("🧾 Chemins explorés sauvegardés dans eeg_guided_path.json")
