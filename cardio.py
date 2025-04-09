"""
Licence : Creative Commons BY-NC-SA 4.0
Auteurs :
    - Kocupyr Romain (Auteur)
    - Multi_gpt_api
    - Grok
A adapter à la cardio
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Désactive le GPU si besoin

import numpy as np
import pandas as pd
import random
from datetime import datetime
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from multiprocessing import Pool

# =========================================================================
#                       Filtre de Kalman simplifié
# =========================================================================
class KalmanFilter:
    def __init__(self, process_noise=0.1, measurement_noise=1.0, error_cov=1.0):
        self.Q = process_noise
        self.R = measurement_noise
        self.P = error_cov
        self.x = None

    def update(self, measurement):
        if self.x is None:
            self.x = measurement
            return self.x
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * P_pred
        return self.x

# =========================================================================
#                     Fonctions utilitaires de features
# =========================================================================
def extract_features(draw, freq_dict, total_draws, previous_draws):
    balls = draw[:5]
    stars = draw[5:]
    S = sum(draw)
    D = np.std(draw)
    P = np.prod(draw)
    log_P = np.log10(P) if P > 0 else 0
    E = sum(x**2 for x in draw)

    freq_mean = sum(freq_dict.get(x, 0) / total_draws for x in draw) / 7
    recent_freq = 0
    if len(previous_draws) >= 10:
        recent_freq = sum(
            1 for d in previous_draws[-10:] for x in draw if x in d
        ) / (7 * 10)

    co_occurrence = 0
    if previous_draws:
        pair_count = 0
        total_pairs = 0
        for i, n1 in enumerate(draw):
            for n2 in draw[i+1:]:
                total_pairs += 1
                for past_draw in previous_draws[-10:]:
                    if n1 in past_draw and n2 in past_draw:
                        pair_count += 1
        if total_pairs > 0:
            co_occurrence = pair_count / (total_pairs * min(10, len(previous_draws)))

    return [S, D, log_P, E, freq_mean, recent_freq, co_occurrence]

def calculate_entropy(draw, freq_dict, total_draws):
    p = [freq_dict.get(x, 0) / total_draws if x in freq_dict else 0.001 for x in draw]
    return -sum(pi * np.log2(pi + 1e-10) for pi in p if pi > 0)

def calculate_correlation(draw, previous_draws):
    if not previous_draws:
        return 0
    common = [sum(1 for num in draw if num in prev) for prev in previous_draws[-10:]]
    return np.mean(common) / 7 if common else 0

def calculate_frequency_dynamics(draw, previous_draws):
    if len(previous_draws) < 20:
        return 0
    freq_now = {x: 0 for x in draw}
    freq_old = {x: 0 for x in draw}
    for d in previous_draws[-10:]:
        for v in d:
            if v in freq_now:
                freq_now[v] += 1
    for d in previous_draws[-20:-10]:
        for v in d:
            if v in freq_old:
                freq_old[v] += 1
    for x in draw:
        freq_now[x] /= 70
        freq_old[x] /= 70
    delta_f = sum(freq_now[x] - freq_old[x] for x in draw) / 7
    return delta_f

def calculate_inter_variance(draw, previous_draws):
    if len(previous_draws) < 10:
        return 0
    sorted_draw = sorted(draw[:5])
    gaps = [sorted_draw[i+1] - sorted_draw[i] for i in range(4)]

    gap_list = []
    for prev in previous_draws[-10:]:
        sd = sorted(prev[:5])
        gap_list.extend(sd[i+1] - sd[i] for i in range(4))

    if not gap_list:
        return 0
    mean_gap = np.mean(gap_list)
    diffs = [(g - mean_gap) for g in gaps]
    return np.var(diffs) if diffs else 0

def calculate_entropy_gradient(draw, previous_draws, total_draws, freq_dict):
    if len(previous_draws) < 10:
        return 0
    H_t = calculate_entropy(draw, freq_dict, total_draws)
    H_t10 = calculate_entropy(previous_draws[-10], freq_dict, total_draws - 10)
    delta_T = 180
    return (H_t - H_t10) / delta_T

def calculate_kl_divergence(draw, previous_draws):
    if len(previous_draws) < 10:
        return 0
    p_balls = np.zeros(50)
    p_stars = np.zeros(12)
    for x in draw[:5]:
        p_balls[x-1] = 1.0 / 7
    for x in draw[5:]:
        p_stars[x-1] = 1.0 / 7
    p = np.concatenate([p_balls, p_stars])

    ball_counts = np.zeros(50)
    star_counts = np.zeros(12)
    for d in previous_draws[-10:]:
        for b in d[:5]:
            ball_counts[b-1] += 1
        for s in d[5:]:
            star_counts[s-1] += 1
    total_picks = 10*5 + 10*2
    q_balls = ball_counts / total_picks
    q_stars = star_counts / total_picks
    q = np.concatenate([q_balls, q_stars])

    D_KL = 0.0
    for pi, qi in zip(p, q):
        if pi > 0 and qi > 0:
            D_KL += pi * np.log(pi / (qi + 1e-12))
    return D_KL

def calculate_temporal_feedback(predicted_draws, actual_draws):
    if not predicted_draws or not actual_draws or len(predicted_draws) < 10:
        return 0
    scores = []
    for pred, actual in zip(predicted_draws[-10:], actual_draws[-10:]):
        balls_correct = sum(1 for x in pred[:5] if x in actual[:5])
        stars_correct = sum(1 for x in pred[5:] if x in actual[5:])
        scores.append((balls_correct + 2*stars_correct) / 7)
    weights = [1 - 0.1*i for i in range(len(scores))]
    return sum(w*s for w,s in zip(weights, scores)) / sum(weights)

# =========================================================================
#                Score directionnel et score binaire
# =========================================================================
def calculate_directional_scores(pred_draw, real_draw):
    pred_balls = pred_draw[:5]
    real_balls = real_draw[:5]
    pred_stars = pred_draw[5:]
    real_stars = real_draw[5:]
    
    ball_score = 0
    used_real_balls = set()
    exact_matches_balls = [b for b in pred_balls if b in real_balls]
    for b in exact_matches_balls:
        ball_score += 50
        used_real_balls.add(b)
    remaining_pred_balls = [b for b in pred_balls if b not in exact_matches_balls]
    remaining_real_balls = [b for b in real_balls if b not in used_real_balls]
    for pb in remaining_pred_balls:
        if remaining_real_balls:
            closest_real = min(remaining_real_balls, key=lambda x: abs(pb - x))
            diff = abs(pb - closest_real)
            if diff < 50:
                ball_score += 50 - diff
            else:
                ball_score -= 49
            remaining_real_balls.remove(closest_real)

    star_score = 0
    used_real_stars = set()
    exact_matches_stars = [s for s in pred_stars if s in real_stars]
    for s in exact_matches_stars:
        star_score += 12
        used_real_stars.add(s)
    remaining_pred_stars = [s for s in pred_stars if s not in exact_matches_stars]
    remaining_real_stars = [s for s in real_stars if s not in used_real_stars]
    for ps in remaining_pred_stars:
        if remaining_real_stars:
            closest_real = min(remaining_real_stars, key=lambda x: abs(ps - x))
            diff = abs(ps - closest_real)
            if diff < 12:
                star_score += 12 - diff
            else:
                star_score -= 11
            remaining_real_stars.remove(closest_real)

    return ball_score, star_score

def calculate_binary_score(pred_draw, real_draw):
    pred_balls = pred_draw[:5]
    real_balls = real_draw[:5]
    pred_stars = pred_draw[5:]
    real_stars = real_draw[5:]
    return sum(1 for x in pred_balls if x in real_balls) \
         + sum(1 for x in pred_stars if x in real_stars)

# =========================================================================
#   Tirage aléatoire pondéré sans remise (pour la Monte-Carlo)
# =========================================================================
def weighted_sample_without_replacement(population, weights, k):
    selected = set()
    while len(selected) < k:
        probs = [w if i not in selected else 0 for i, w in enumerate(weights)]
        total = sum(probs)
        if total == 0:
            probs = [1 if i not in selected else 0 for i in range(len(population))]
            total = sum(probs)
        probs = [p/total for p in probs]
        choice = random.choices(range(len(population)), weights=probs, k=1)[0]
        selected.add(choice)
    return [population[i] for i in selected]

# =========================================================================
#   Fonction parallèle simulant un chunk de tirages Monte-Carlo
# =========================================================================
def simulate_chunk(chunk_size, features, p_balls, p_stars, last_real_draw,
                   freq_dict, total_draws, previous_draws, num_grilles):
    (S_lisse, D_lisse, log_P_lisse, E_lisse,
     freq_lisse, recent_freq_lisse, co_occurrence_lisse,
     C, F, V_I, G_H, D_KL, R_T, H_lisse, pair_score_lisse, *rest) = features
    freq_per_num_lisse = rest  # 62 valeurs lissées

    chunk_scores = []
    seen_in_chunk = set()

    for _ in range(chunk_size):
        balls = weighted_sample_without_replacement(list(range(1,51)), p_balls, 5)
        stars = weighted_sample_without_replacement(list(range(1,13)), p_stars, 2)
        draw = balls + stars
        draw_tuple = tuple(sorted(draw[:5])) + tuple(sorted(draw[5:]))

        if draw_tuple not in seen_in_chunk:
            seen_in_chunk.add(draw_tuple)

            (S_sim, D_sim, log_P_sim, E_sim, freq_sim,
             recent_freq_sim, co_occurrence_sim) = extract_features(
                draw, freq_dict, total_draws, previous_draws
            )

            # pair_score pour le tirage simulé
            pair_freq = {}
            if len(previous_draws) >= 10:
                for past_draw in previous_draws[-10:]:
                    for i in range(len(past_draw)):
                        for j in range(i+1, len(past_draw)):
                            pair = tuple(sorted([past_draw[i], past_draw[j]]))
                            pair_freq[pair] = pair_freq.get(pair, 0) + 1

            pair_score_sim = 0
            total_sim_pairs = 0
            for i in range(len(draw)):
                for j in range(i+1, len(draw)):
                    pair_ = tuple(sorted([draw[i], draw[j]]))
                    pair_score_sim += pair_freq.get(pair_, 0)
                    total_sim_pairs += 1
            pair_score_sim /= 70 if total_sim_pairs else 1

            freq_per_num_sim = [freq_dict.get(i, 0) / total_draws for i in range(1, 63)]
            freq_per_num_diff = sum((freq_per_num_sim[i] - freq_per_num_lisse[i])**2 
                                    for i in range(62)) / 62

            # =================================================================
            #   Calcul du score (calibrage des pénalités)
            # =================================================================
            feature_score = -(
                (S_sim - S_lisse)**2 / 25**2 * 1.0
                + (D_sim - D_lisse)**2 / 2**2 * 0.7
                + (log_P_sim - log_P_lisse)**2 / 0.7**2 * 0.3
                + (E_sim - E_lisse)**2 / 1000**2 * 0.9
                + (freq_sim - freq_lisse)**2 / 0.1**2 * 1.0
                + (recent_freq_sim - recent_freq_lisse)**2 / 0.1**2 * 2.0
                + (co_occurrence_sim - co_occurrence_lisse)**2 / 0.1**2 * 2.0
                + (calculate_correlation(draw, previous_draws[-10:]) - C)**2 * 5.0
                + (calculate_frequency_dynamics(draw, previous_draws[-10:]) - F)**2 * 4.0
                + (calculate_inter_variance(draw, previous_draws[-10:]) - V_I)**2 / 20**2 * 1.0
                + (calculate_entropy_gradient(draw, previous_draws[-10:], total_draws, freq_dict) - G_H)**2 * 0.5
                + (calculate_kl_divergence(draw, previous_draws[-10:]) - D_KL)**2 * 0.7
            )
            # Pénalise plus fortement freq_per_num_diff (ex. * 10.0)
            feature_score -= freq_per_num_diff * 10.0

            # Pénalise l'écart de pair_score (ex. * 5.0)
            pair_diff = (pair_score_sim - pair_score_lisse)**2
            feature_score -= pair_diff * 5.0

            # Score directionnel
            ball_score, star_score = calculate_directional_scores(draw, last_real_draw)
            directional_score = ball_score + star_score

            # Petit bonus proportionnel aux probas
            bonus = sum(p_balls[b-1] for b in balls) + sum(p_stars[s-1] for s in stars)

            total_score = feature_score + bonus*(1 + R_T) + directional_score*0.5
            chunk_scores.append((draw, total_score))

    # On ne garde que les num_grilles meilleurs
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    chunk_scores = chunk_scores[:num_grilles]
    return chunk_scores

# =========================================================================
#  Classe de prédiction avec pondération adaptative + NN + GridSearch
# =========================================================================
class EuroMillionsPredictorEnsemble:
    def __init__(self):
        # Filtres Kalman
        self.kf_S = KalmanFilter()
        self.kf_D = KalmanFilter()
        self.kf_logP = KalmanFilter()
        self.kf_E = KalmanFilter()
        self.kf_freq = KalmanFilter()
        self.kf_recent_freq = KalmanFilter()
        self.kf_co_occurrence = KalmanFilter()

        # 62 filtres Kalman pour freq_per_num
        self.kf_freq_per_num = [KalmanFilter() for _ in range(62)]

        self.freq_dict = {}
        self.total_draws = 0
        self.previous_draws = []
        self.predicted_draws = []
        self.actual_draws = []

        self.scaler = StandardScaler()

        # 4 modèles par numéro
        self.models_balls = {
            i: {
                "lr": LogisticRegression(max_iter=1000),
                "rf": RandomForestClassifier(random_state=42),
                "gb": GradientBoostingClassifier(random_state=42),
                "xgb": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            }
            for i in range(1, 51)
        }
        self.models_stars = {
            i: {
                "lr": LogisticRegression(max_iter=1000),
                "rf": RandomForestClassifier(random_state=42),
                "gb": GradientBoostingClassifier(random_state=42),
                "xgb": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            }
            for i in range(1, 13)
        }

        # Poids adaptatifs
        self.weights_balls = {
            i: {"lr": 0.25, "rf": 0.25, "gb": 0.25, "xgb": 0.25}
            for i in range(1, 51)
        }
        self.weights_stars = {
            i: {"lr": 0.25, "rf": 0.25, "gb": 0.25, "xgb": 0.25}
            for i in range(1, 13)
        }

        self.nn_weight = 1.0

        # NN => 15 features (de base) + 62 freq_per_num = 77
        self.nn_model = Sequential([
            Dense(128, activation='relu', input_shape=(77,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(62, activation='sigmoid')
        ])
        self.nn_model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    def update_features(self, draw):
        """
        Calcule un vecteur de 15 features brutes + 62 freq_per_num => 77 total.
        """
        self.previous_draws.append(draw)
        self.actual_draws.append(draw)
        self.total_draws += 1

        for num in draw:
            self.freq_dict[num] = self.freq_dict.get(num, 0) + 1

        # 1) Features basiques
        (S, D, log_P, E, freq_mean,
         recent_freq, co_occurrence) = extract_features(draw, self.freq_dict,
                                                        self.total_draws,
                                                        self.previous_draws)
        # 2) Lissage
        S_lisse = self.kf_S.update(S)
        D_lisse = self.kf_D.update(D)
        log_P_lisse = self.kf_logP.update(log_P)
        E_lisse = self.kf_E.update(E)
        freq_lisse = self.kf_freq.update(freq_mean)
        recent_freq_lisse = self.kf_recent_freq.update(recent_freq)
        co_occurrence_lisse = self.kf_co_occurrence.update(co_occurrence)

        # 3) Dérivées
        C = calculate_correlation(draw, self.previous_draws[:-1])
        F = calculate_frequency_dynamics(draw, self.previous_draws[:-1])
        V_I = calculate_inter_variance(draw, self.previous_draws[:-1])
        G_H = calculate_entropy_gradient(draw, self.previous_draws[:-1],
                                         self.total_draws, self.freq_dict)
        D_KL = calculate_kl_divergence(draw, self.previous_draws[:-1])
        R_T = calculate_temporal_feedback(self.predicted_draws, self.actual_draws)
        H = calculate_entropy(draw, self.freq_dict, self.total_draws)

        # 4) pair_score
        pair_freq = {}
        if len(self.previous_draws) >= 10:
            for past_draw in self.previous_draws[-10:]:
                for i in range(len(past_draw)):
                    for j in range(i+1, len(past_draw)):
                        pair = tuple(sorted([past_draw[i], past_draw[j]]))
                        pair_freq[pair] = pair_freq.get(pair, 0) + 1
        pair_score = 0
        total_pairs = 0
        for i in range(len(draw)):
            for j in range(i+1, len(draw)):
                pair_ = tuple(sorted([draw[i], draw[j]]))
                pair_score += pair_freq.get(pair_, 0)
                total_pairs += 1
        pair_score = pair_score / 70 if total_pairs else 0

        # 5) freq_per_num lissées
        freq_per_num = []
        for i in range(1, 63):
            raw_freq = self.freq_dict.get(i, 0) / self.total_draws
            freq_smooth = self.kf_freq_per_num[i-1].update(raw_freq)
            freq_per_num.append(freq_smooth)

        return [
            S_lisse, D_lisse, log_P_lisse, E_lisse,
            freq_lisse, recent_freq_lisse, co_occurrence_lisse,
            C, F, V_I, G_H, D_KL, R_T, H, pair_score
        ] + freq_per_num

    def train_models(self, draws):
        X = []
        y_balls = {i: [] for i in range(1, 51)}
        y_stars = {i: [] for i in range(1, 13)}
        feature_history = []

        # Paramètres élargis : learning_rate
        rf_param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
        gb_param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
        xgb_param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}

        for draw in draws:
            feats = self.update_features(draw)
            X.append(feats)
            for i in range(1, 51):
                y_balls[i].append(int(i in draw[:5]))
            for i in range(1, 13):
                y_stars[i].append(int(i in draw[5:]))
            feature_history.append(feats)

        X = np.array(X)
        X_scaled_full = self.scaler.fit_transform(X)

        # ======================
        #   GridSearch pour boules
        # ======================
        for i in range(1, 51):
            y_i = np.array(y_balls[i])
            X_train_, X_val_, y_train_, y_val_ = train_test_split(X_scaled_full, y_i,
                                                                  test_size=0.2,
                                                                  random_state=42)
            # LR
            self.models_balls[i]["lr"].fit(X_train_, y_train_)
            acc_lr = self.models_balls[i]["lr"].score(X_val_, y_val_)
            self.weights_balls[i]["lr"] = acc_lr

            # RandomForest
            rf_gridsearch = GridSearchCV(RandomForestClassifier(random_state=42),
                                         rf_param_grid, cv=3, n_jobs=-1)
            rf_gridsearch.fit(X_train_, y_train_)
            best_rf = rf_gridsearch.best_estimator_
            self.models_balls[i]["rf"] = best_rf
            acc_rf = best_rf.score(X_val_, y_val_)
            self.weights_balls[i]["rf"] = acc_rf

            # GradientBoosting
            gb_gridsearch = GridSearchCV(GradientBoostingClassifier(random_state=42),
                                         gb_param_grid, cv=3, n_jobs=-1)
            gb_gridsearch.fit(X_train_, y_train_)
            best_gb = gb_gridsearch.best_estimator_
            self.models_balls[i]["gb"] = best_gb
            acc_gb = best_gb.score(X_val_, y_val_)
            self.weights_balls[i]["gb"] = acc_gb

            # XGB
            xgb_gridsearch = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                                          xgb_param_grid, cv=3, n_jobs=-1)
            xgb_gridsearch.fit(X_train_, y_train_)
            best_xgb = xgb_gridsearch.best_estimator_
            self.models_balls[i]["xgb"] = best_xgb
            acc_xgb = best_xgb.score(X_val_, y_val_)
            self.weights_balls[i]["xgb"] = acc_xgb

        # ======================
        #   GridSearch pour étoiles
        # ======================
        for i in range(1, 13):
            y_i = np.array(y_stars[i])
            X_train_, X_val_, y_train_, y_val_ = train_test_split(X_scaled_full, y_i,
                                                                  test_size=0.2,
                                                                  random_state=42)
            # LR
            self.models_stars[i]["lr"].fit(X_train_, y_train_)
            acc_lr = self.models_stars[i]["lr"].score(X_val_, y_val_)
            self.weights_stars[i]["lr"] = acc_lr

            # RF
            rf_gridsearch = GridSearchCV(RandomForestClassifier(random_state=42),
                                         rf_param_grid, cv=3, n_jobs=-1)
            rf_gridsearch.fit(X_train_, y_train_)
            best_rf = rf_gridsearch.best_estimator_
            self.models_stars[i]["rf"] = best_rf
            acc_rf = best_rf.score(X_val_, y_val_)
            self.weights_stars[i]["rf"] = acc_rf

            # GB
            gb_gridsearch = GridSearchCV(GradientBoostingClassifier(random_state=42),
                                         gb_param_grid, cv=3, n_jobs=-1)
            gb_gridsearch.fit(X_train_, y_train_)
            best_gb = gb_gridsearch.best_estimator_
            self.models_stars[i]["gb"] = best_gb
            acc_gb = best_gb.score(X_val_, y_val_)
            self.weights_stars[i]["gb"] = acc_gb

            # XGB
            xgb_gridsearch = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                                          xgb_param_grid, cv=3, n_jobs=-1)
            xgb_gridsearch.fit(X_train_, y_train_)
            best_xgb = xgb_gridsearch.best_estimator_
            self.models_stars[i]["xgb"] = best_xgb
            acc_xgb = best_xgb.score(X_val_, y_val_)
            self.weights_stars[i]["xgb"] = acc_xgb

        # NN
        y_nn = np.zeros((len(draws), 62))
        for idx, draw in enumerate(draws):
            for num in draw[:5]:
                y_nn[idx, num-1] = 1
            for num in draw[5:]:
                y_nn[idx, 50 + num-1] = 1

        X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
            X_scaled_full, y_nn, test_size=0.2, random_state=42
        )
        self.nn_model.fit(X_train_nn, y_train_nn, epochs=100, batch_size=32, verbose=0)
        nn_loss = self.nn_model.evaluate(X_val_nn, y_val_nn, verbose=0)
        self.nn_weight = 1.0 / (nn_loss + 1e-6)

        return feature_history

    def get_ensemble_probability(self, X_scaled):
        """
        Calcule la proba de l'ensemble adaptatif normalisé + NN (alpha inversé).
        """
        p_balls_ensemble = {}
        p_stars_ensemble = {}

        # 1) Ensemble (4 sous-modèles pondérés)
        for i in range(1, 51):
            denom = sum(self.weights_balls[i].values()) + 1e-12
            prob_sum = 0.0
            for m, model_obj in self.models_balls[i].items():
                w = self.weights_balls[i][m]
                prob_m = model_obj.predict_proba(X_scaled)[0, 1]
                prob_sum += w * prob_m
            p_balls_ensemble[i] = prob_sum / denom

        for i in range(1, 13):
            denom = sum(self.weights_stars[i].values()) + 1e-12
            prob_sum = 0.0
            for m, model_obj in self.models_stars[i].items():
                w = self.weights_stars[i][m]
                prob_m = model_obj.predict_proba(X_scaled)[0, 1]
                prob_sum += w * prob_m
            p_stars_ensemble[i] = prob_sum / denom

        # 2) Normalisation brute => somme=1
        total_balls_ensemble = sum(p_balls_ensemble.values())
        total_stars_ensemble = sum(p_stars_ensemble.values())
        for i in range(1, 51):
            p_balls_ensemble[i] /= (total_balls_ensemble + 1e-12)
        for i in range(1, 13):
            p_stars_ensemble[i] /= (total_stars_ensemble + 1e-12)

        # 3) NN
        p_nn = self.nn_model.predict(X_scaled, verbose=0)[0]
        p_balls_nn = {i+1: p_nn[i] for i in range(50)}
        p_stars_nn = {i+1: p_nn[50 + i] for i in range(12)}

        # 4) Fusion alpha inversé
        final_p_balls_list = []
        final_p_stars_list = []

        for i in range(1, 51):
            sum_ensemble = sum(self.weights_balls[i].values())
            alpha = sum_ensemble / (sum_ensemble + self.nn_weight + 1e-12)
            ensemble_p = p_balls_ensemble[i]
            nn_p = p_balls_nn[i]
            final_prob = alpha*ensemble_p + (1 - alpha)*nn_p
            final_p_balls_list.append(final_prob)

        for i in range(1, 13):
            sum_ensemble = sum(self.weights_stars[i].values())
            alpha = sum_ensemble / (sum_ensemble + self.nn_weight + 1e-12)
            ensemble_p = p_stars_ensemble[i]
            nn_p = p_stars_nn[i]
            final_prob = alpha*ensemble_p + (1 - alpha)*nn_p
            final_p_stars_list.append(final_prob)

        # 5) Normalisation => somme(balls)=5, somme(stars)=2
        total_balls = sum(final_p_balls_list)
        total_stars = sum(final_p_stars_list)
        if total_balls == 0:
            total_balls = 1e-12
        if total_stars == 0:
            total_stars = 1e-12

        p_balls_norm = [p * 5 / total_balls for p in final_p_balls_list]
        p_stars_norm = [p * 2 / total_stars for p in final_p_stars_list]

        return p_balls_norm, p_stars_norm

    def monte_carlo_predict(self, features, date_future, num_grilles=4, real_draw=None):
        """
        Monte-Carlo : 2 000 000 simulations en 8 processus (exemple).
        Log final : top 5 boules/étoiles dans p_balls/p_stars + 
        top 5 boules/étoiles simulées.
        """
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        p_balls, p_stars = self.get_ensemble_probability(X_scaled)

        # Diagnostic
        if real_draw is not None:
            print(f"\n[Diagnostic] Test {date_future.strftime('%Y-%m-%d')}:")
            print(f"  Réel : {real_draw}")
            print(f"  p_balls pour {real_draw[:5]}: {[p_balls[x-1] for x in real_draw[:5]]}")
            print(f"  p_stars pour {real_draw[5:]}: {[p_stars[x-1] for x in real_draw[5:]]}")

        top_balls = sorted(enumerate(p_balls, 1), key=lambda x: x[1], reverse=True)[:5]
        top_stars = sorted(enumerate(p_stars, 1), key=lambda x: x[1], reverse=True)[:2]
        print(f"  Top 5 p_balls: {[(n, p) for n, p in top_balls]}")
        print(f"  Top 2 p_stars: {[(n, p) for n, p in top_stars]}")

        last_real_draw = self.actual_draws[-1] if self.actual_draws else [1,2,3,4,5,1,1]

        # Parallélisation
        simulations = 2000000
        nproc = 8
        chunk_size = simulations // nproc

        args_for_pool = []
        for _ in range(nproc):
            args_for_pool.append((chunk_size, features, p_balls, p_stars,
                                  last_real_draw, self.freq_dict,
                                  self.total_draws, self.previous_draws,
                                  num_grilles))

        with Pool(nproc) as p:
            results = p.starmap(simulate_chunk, args_for_pool)

        # Fusion
        draw_scores = [item for sublist in results for item in sublist]

        # Tri global final
        draw_scores.sort(key=lambda x: x[1], reverse=True)
        draw_scores = draw_scores[:num_grilles * nproc]

        # Log des numéros les plus fréquents dans les grilles simulées
        ball_counts = Counter()
        star_counts = Counter()
        for draw, _ in draw_scores:
            ball_counts.update(draw[:5])
            star_counts.update(draw[5:])

        print(f"  Top 5 boules simulées: {ball_counts.most_common(5)}")
        print(f"  Top 2 étoiles simulées: {star_counts.most_common(2)}")

        # Sélection finale
        best_draws = []
        seen = set()
        for draw, score in draw_scores:
            draw_tuple = tuple(sorted(draw[:5])) + tuple(sorted(draw[5:]))
            if draw_tuple not in seen:
                best_draws.append(draw)
                seen.add(draw_tuple)
                if len(best_draws) == num_grilles:
                    break

        self.predicted_draws.append(best_draws[0])
        return best_draws

# =========================================================================
#                            Script principal
# =========================================================================
if __name__ == "__main__":
    data = pd.read_excel("../data/euro-histo-last.xlsx")
    draws_data = data[['boule_1','boule_2','boule_3','boule_4','boule_5',
                       'etoile_1','etoile_2']].values.tolist()
    dates_data = pd.to_datetime(data['date_de_tirage'])

    train_size = 1800
    train_draws = draws_data[:train_size]
    test_draws = draws_data[train_size:]
    test_dates = dates_data[train_size:]

    predictor = EuroMillionsPredictorEnsemble()

    print("=== Entraînement des modèles (ensemble + NN + GridSearch) ===")
    feature_history = predictor.train_models(train_draws)

    print("\n=== Validation sur le jeu de test (sans triche) ===")
    for test_draw, test_date in zip(test_draws, test_dates):
        last_features = feature_history[-1]

        predicted_grids = predictor.monte_carlo_predict(
            last_features,
            date_future=test_date,
            num_grilles=4,
            real_draw=test_draw
        )

        print(f"\nTirage réel du {test_date.strftime('%Y-%m-%d')} : {test_draw}")
        for i, pg in enumerate(predicted_grids, 1):
            b_score, s_score = calculate_directional_scores(pg, test_draw)
            bin_score = calculate_binary_score(pg, test_draw)
            print(f"  Grille {i} => {pg}, "
                  f"Directional (boules/étoiles)=({b_score:.1f},{s_score:.1f}), "
                  f"Binaire={bin_score}/7")

        new_feat = predictor.update_features(test_draw)
        feature_history.append(new_feat)

    # Prédiction finale
    future_date = datetime(2025, 4, 15)
    last_features = feature_history[-1]
    future_preds = predictor.monte_carlo_predict(
        last_features,
        date_future=future_date,
        num_grilles=4,
        real_draw=None
    )
    print(f"\n=== Prédictions pour le {future_date.strftime('%Y-%m-%d')} ===")
    for i, draw in enumerate(future_preds, 1):
        print(f"   Grille {i} : {draw}")
    print("  (Probabilité de gain toujours extrêmement faible...)")

