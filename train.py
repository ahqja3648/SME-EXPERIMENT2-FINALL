"""
train.py
Smart Mobility Engineering Experiment 2 Final Project

Purpose
- Train and compare multiple README-compatible localization models.
- Save the selected compact final model as model.pkl.
- Save validation_results.csv for report.md.

Used packages are limited to the README standard environment:
numpy, scipy, scikit-learn, pandas.
"""

import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MAT_CANDIDATES = ["DH_FR1.mat", "InF_DH_FR1.mat"]
MODEL_PATH = "model.pkl"
RESULT_CSV_PATH = "validation_results.csv"


def load_data():
    mat_path = None
    for path in MAT_CANDIDATES:
        if os.path.exists(path):
            mat_path = path
            break
    if mat_path is None:
        raise FileNotFoundError("DH_FR1.mat 파일을 현재 폴더에 넣어주세요.")

    data = sio.loadmat(mat_path, squeeze_me=False)
    d_hat = np.asarray(data["d_hat"], dtype=float)
    p = np.asarray(data["p"], dtype=float)

    if "BS_positions" in data:
        bs_positions = np.asarray(data["BS_positions"], dtype=float)
    elif "p_bs" in data:
        bs_positions = np.asarray(data["p_bs"], dtype=float)
    else:
        raise KeyError("BS_positions 또는 p_bs 변수를 찾을 수 없습니다.")

    if d_hat.shape[0] != 18:
        raise ValueError(f"d_hat shape 오류: {d_hat.shape}")
    if bs_positions.shape != (2, 18):
        raise ValueError(f"BS_positions shape 오류: {bs_positions.shape}")
    if p.shape[0] != 2 or p.shape[1] != d_hat.shape[1]:
        raise ValueError(f"p와 d_hat 사용자 수가 다릅니다. p={p.shape}, d_hat={d_hat.shape}")
    return d_hat, bs_positions, p, mat_path


def localization_metrics(y_true, y_pred):
    error = np.linalg.norm(y_pred - y_true, axis=1)
    return {
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(error)),
        "Median_error": float(np.median(error)),
        "m90": float(np.percentile(error, 90)),
        "Max_error": float(np.max(error)),
    }


def true_distances(p_xy, bs_positions):
    return np.linalg.norm(p_xy.T[:, None, :] - bs_positions.T[None, :, :], axis=2)


def rank_matrix(D):
    ranks = np.zeros_like(D, dtype=float)
    ranks[np.arange(D.shape[0])[:, None], np.argsort(D, axis=1)] = np.arange(D.shape[1])
    return ranks / (D.shape[1] - 1)


def user_stat_features(D):
    return np.vstack([
        D.mean(axis=1), np.median(D, axis=1), D.std(axis=1),
        D.min(axis=1), D.max(axis=1), np.ptp(D, axis=1),
        np.percentile(D, 10, axis=1), np.percentile(D, 25, axis=1),
        np.percentile(D, 75, axis=1), np.percentile(D, 90, axis=1),
    ]).T


def anchor_level_features(D, bs_positions):
    anchors = bs_positions.T
    N = D.shape[0]
    stats = user_stat_features(D)
    ranks = rank_matrix(D)
    sorted_small = np.sort(D, axis=1)[:, :8]
    inv = 1.0 / (D + 1e-6)
    inv = inv / inv.sum(axis=1, keepdims=True)
    inverse_center = inv @ anchors

    blocks = []
    for i in range(18):
        onehot = np.zeros((N, 18), dtype=float)
        onehot[:, i] = 1.0
        diff_to_anchor = (D - D[:, [i]]) / 100.0
        anchor_xy = np.repeat(anchors[[i]], N, axis=0)
        blocks.append(np.hstack([
            D[:, [i]], ranks[:, [i]], D[:, [i]] - stats[:, [0]], D[:, [i]] - stats[:, [1]],
            stats, sorted_small, inverse_center, anchor_xy, onehot, diff_to_anchor,
        ]))
    return np.vstack(blocks)


def train_anchor_error_model(d_train, p_train, bs_positions):
    D = d_train.T
    true_D = true_distances(p_train, bs_positions)
    target_error = np.abs(D - true_D)
    X_anchor = anchor_level_features(D, bs_positions)
    y_anchor = np.concatenate([target_error[:, i] for i in range(18)])
    model = ExtraTreesRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
    )
    model.fit(X_anchor, np.log1p(y_anchor))
    return model


def predict_anchor_error(anchor_error_model, D, bs_positions):
    X_anchor = anchor_level_features(D, bs_positions)
    pred = np.expm1(anchor_error_model.predict(X_anchor))
    N = D.shape[0]
    out = np.zeros((N, 18), dtype=float)
    for i in range(18):
        out[:, i] = pred[i * N : (i + 1) * N]
    return np.maximum(out, 0.05)


def linear_wls_candidate(d, bs_positions, indices, weights=None):
    anchors = bs_positions.T
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 3:
        indices = np.argsort(d)[:3]
    ref = indices[0]
    others = indices[1:]
    a0 = anchors[ref]
    M = 2.0 * (a0[None, :] - anchors[others])
    b = d[others] ** 2 - d[ref] ** 2 - np.sum(anchors[others] ** 2, axis=1) + np.dot(a0, a0)
    if weights is None:
        w = np.ones(len(others), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)[1:]
    sqrt_w = np.sqrt(np.maximum(w, 1e-6))
    try:
        x = np.linalg.lstsq(M * sqrt_w[:, None], b * sqrt_w, rcond=None)[0]
    except Exception:
        inv = 1.0 / (d[indices] + 1e-6)
        x = np.sum(anchors[indices] * inv[:, None], axis=0) / inv.sum()
    x[0] = np.clip(x[0], -80.0, 80.0)
    x[1] = np.clip(x[1], -50.0, 50.0)
    return x


def candidate_position_features(D, bs_positions, pred_error):
    N = D.shape[0]
    positions = []
    residual_stats = []
    for n in range(N):
        d = D[n]
        e = pred_error[n]
        row_positions = []
        row_stats = []
        candidate_specs = [(np.arange(18), 1.0 / (e + 1.0))]
        for k in [5, 7, 9, 12]:
            idx = np.argsort(e)[:k]
            candidate_specs.append((idx, 1.0 / (e[idx] + 1.0)))
        for k in [5, 7, 9, 12]:
            idx = np.argsort(d)[:k]
            candidate_specs.append((idx, None))
        for idx, weights in candidate_specs:
            pos = linear_wls_candidate(d, bs_positions, idx, weights)
            residual = np.linalg.norm(pos[None, :] - bs_positions.T, axis=1) - d
            abs_residual = np.abs(residual)
            row_positions.extend(pos)
            row_stats.extend([
                abs_residual.mean(), np.median(abs_residual), abs_residual.std(),
                abs_residual.min(), abs_residual.max(), np.percentile(abs_residual, 75),
            ])
        positions.append(row_positions)
        residual_stats.append(row_stats)
    return np.asarray(positions), np.asarray(residual_stats)


def basic_user_features(D, bs_positions):
    anchors = bs_positions.T
    N = D.shape[0]
    blocks = [D, np.log1p(D), np.sqrt(D), (D ** 2) / 100.0, user_stat_features(D), rank_matrix(D)]
    for k in [1, 2, 3, 4, 5, 6, 8]:
        mask = np.zeros_like(D, dtype=float)
        idx = np.argsort(D, axis=1)[:, :k]
        for n in range(N):
            mask[n, idx[n]] = 1.0
        blocks.append(mask)
    for power in [1, 2, 3]:
        W = 1.0 / (D + 1e-6) ** power
        W = W / W.sum(axis=1, keepdims=True)
        blocks.append(W @ anchors)
    blocks.append(np.sort(D, axis=1)[:, :10])
    for k in [3, 5, 7, 9, 12]:
        idx = np.argsort(D, axis=1)[:, :k]
        centers = np.asarray([anchors[i].mean(axis=0) for i in idx])
        spreads = np.asarray([anchors[i].std(axis=0) for i in idx])
        dstats = np.asarray([[D[n, idx[n]].mean(), D[n, idx[n]].std(), D[n, idx[n]].min(), D[n, idx[n]].max()] for n in range(N)])
        blocks.extend([centers, spreads, dstats])
    pairwise = []
    for i in range(18):
        for j in range(i + 1, 18):
            pairwise.append(D[:, i] - D[:, j])
    blocks.append(np.vstack(pairwise).T / 100.0)
    return np.hstack(blocks)


def build_features(d_matrix, bs_positions, anchor_error_model):
    D = d_matrix.T
    pred_error = predict_anchor_error(anchor_error_model, D, bs_positions)
    candidate_pos, candidate_stats = candidate_position_features(D, bs_positions, pred_error)
    return np.hstack([basic_user_features(D, bs_positions), pred_error, 1.0 / (pred_error + 1.0), candidate_pos, candidate_stats])


def candidate_models():
    return [
        ("ridge_alpha12", make_pipeline(StandardScaler(), Ridge(alpha=12.0))),
        ("ridge_alpha50", make_pipeline(StandardScaler(), Ridge(alpha=50.0))),
        ("pls_regression", make_pipeline(StandardScaler(), PLSRegression(n_components=20))),
        ("knn_5_distance", make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="distance"))),
        ("knn_9_distance", make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=9, weights="distance"))),
        ("random_forest_60", RandomForestRegressor(n_estimators=60, random_state=RANDOM_STATE, min_samples_leaf=1, max_features=0.65, n_jobs=-1)),
        ("random_forest_120", RandomForestRegressor(n_estimators=120, random_state=RANDOM_STATE, min_samples_leaf=1, max_features=0.65, n_jobs=-1)),
        ("extra_trees_60", ExtraTreesRegressor(n_estimators=60, random_state=RANDOM_STATE, min_samples_leaf=1, max_features=0.65, n_jobs=-1)),
        ("extra_trees_80_depth20", ExtraTreesRegressor(n_estimators=80, random_state=RANDOM_STATE, min_samples_leaf=1, max_features=0.65, max_depth=20, n_jobs=-1)),
        ("gradient_boosting", MultiOutputRegressor(GradientBoostingRegressor(n_estimators=60, learning_rate=0.08, max_depth=3, random_state=RANDOM_STATE))),
    ]


def evaluate_candidates(X_train, y_train, X_valid, y_valid):
    rows = []
    trained = {}
    predictions = {}
    for name, template in candidate_models():
        t0 = time.time()
        model = clone(template)
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        metrics = localization_metrics(y_valid, pred)
        metrics.update({"model": name, "mode": "single", "fit_sec": float(time.time() - t0)})
        rows.append(metrics)
        trained[name] = model
        predictions[name] = pred
        print(metrics, flush=True)

    fixed_ensembles = [
        ("ensemble_ridge_extra_gbr", ["ridge_alpha50", "extra_trees_60", "gradient_boosting"]),
        ("ensemble_rf_extra_gbr", ["random_forest_120", "extra_trees_60", "gradient_boosting"]),
        ("ensemble_ridge_rf_extra_gbr", ["ridge_alpha50", "random_forest_120", "extra_trees_60", "gradient_boosting"]),
    ]
    for ens_name, members in fixed_ensembles:
        inv_rmse = np.asarray([1.0 / [r for r in rows if r["model"] == m][0]["RMSE"] for m in members])
        weights = inv_rmse / inv_rmse.sum()
        pred = sum(weights[i] * predictions[members[i]] for i in range(len(members)))
        metrics = localization_metrics(y_valid, pred)
        metrics.update({"model": ens_name, "mode": "ensemble", "fit_sec": 0.0})
        rows.append(metrics)
        print(metrics, flush=True)

    df = pd.DataFrame(rows).sort_values(["RMSE", "Median_error"]).reset_index(drop=True)
    return df


def make_final_artifact(d_hat, bs_positions, p, selected_model_name):
    anchor_model = train_anchor_error_model(d_hat, p, bs_positions)
    X_all = build_features(d_hat, bs_positions, anchor_model)
    y_all = p.T
    templates = {name: model for name, model in candidate_models()}

    ensemble_members = {
        "ensemble_ridge_extra_gbr": ["ridge_alpha50", "extra_trees_60", "gradient_boosting"],
        "ensemble_rf_extra_gbr": ["random_forest_120", "extra_trees_60", "gradient_boosting"],
        "ensemble_ridge_rf_extra_gbr": ["ridge_alpha50", "random_forest_120", "extra_trees_60", "gradient_boosting"],
    }

    if selected_model_name in ensemble_members:
        models = []
        # Fixed weights are recalculated from validation and stored by train.py below.
        for member in ensemble_members[selected_model_name]:
            print(f"Training final ensemble member: {member}", flush=True)
            model = clone(templates[member])
            model.fit(X_all, y_all)
            models.append((member, model))
        target_type = "ensemble"
        position_model = models
    else:
        print(f"Training final selected model: {selected_model_name}", flush=True)
        model = clone(templates[selected_model_name])
        model.fit(X_all, y_all)
        target_type = "single"
        position_model = selected_model_name, model

    return {
        "author": "양준범 / 스마트모빌리티공학과 / 12223639",
        "model_name": selected_model_name,
        "target_type": target_type,
        "anchor_error_model": anchor_model,
        "position_model": position_model,
        "feature_version": "reliability_multi_candidate_v5",
        "random_state": RANDOM_STATE,
    }


def add_ensemble_weights_to_artifact(artifact, results_df):
    if artifact["target_type"] != "ensemble":
        return artifact
    members = [name for name, _ in artifact["position_model"]]
    rmse_map = {str(row["model"]): float(row["RMSE"]) for _, row in results_df.iterrows()}
    inv = np.asarray([1.0 / rmse_map[m] for m in members])
    weights = inv / inv.sum()
    artifact["ensemble_weights"] = weights
    artifact["ensemble_members"] = members
    return artifact


def main():
    start = time.time()
    d_hat, bs_positions, p, mat_path = load_data()
    print(f"Loaded {mat_path}: d_hat={d_hat.shape}, BS_positions={bs_positions.shape}, p={p.shape}", flush=True)
    print("Using only README standard packages. requirements.txt is not needed.", flush=True)

    idx = np.arange(d_hat.shape[1])
    train_idx, valid_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)
    anchor_model = train_anchor_error_model(d_hat[:, train_idx], p[:, train_idx], bs_positions)
    X_train = build_features(d_hat[:, train_idx], bs_positions, anchor_model)
    X_valid = build_features(d_hat[:, valid_idx], bs_positions, anchor_model)
    y_train = p[:, train_idx].T
    y_valid = p[:, valid_idx].T

    results = evaluate_candidates(X_train, y_train, X_valid, y_valid)
    results.to_csv(RESULT_CSV_PATH, index=False)
    print("\nValidation results")
    print(results.to_string(index=False), flush=True)

    selected_model_name = str(results.iloc[0]["model"])
    print(f"Training final artifact from all provided samples: {selected_model_name}", flush=True)
    artifact = make_final_artifact(d_hat, bs_positions, p, selected_model_name)
    artifact = add_ensemble_weights_to_artifact(artifact, results)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"\nSelected model: {selected_model_name}", flush=True)
    print(f"Saved {MODEL_PATH} ({model_size_mb:.2f} MB)", flush=True)
    print(f"Total train.py time: {time.time() - start:.1f} sec", flush=True)


if __name__ == "__main__":
    main()
