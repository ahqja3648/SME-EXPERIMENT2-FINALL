"""
main.py
Smart Mobility Engineering Experiment 2 Final Project

The grader calls main(). It loads DH_FR1.mat and model.pkl, predicts user positions,
and returns a numpy array with shape (2, num_user).
"""

import os
import pickle

import numpy as np
import scipy.io as sio

MAT_PATH = "DH_FR1.mat"
MODEL_PATH = "model.pkl"


def load_input_data():
    if not os.path.exists(MAT_PATH):
        raise FileNotFoundError("DH_FR1.mat 파일이 현재 폴더에 있어야 합니다.")

    data = sio.loadmat(MAT_PATH, squeeze_me=False)
    d_hat = np.asarray(data["d_hat"], dtype=float)

    if "BS_positions" in data:
        bs_positions = np.asarray(data["BS_positions"], dtype=float)
    elif "p_bs" in data:
        bs_positions = np.asarray(data["p_bs"], dtype=float)
    else:
        raise KeyError("BS_positions 또는 p_bs 변수를 찾을 수 없습니다.")

    return d_hat, bs_positions


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


def main():
    d_hat, bs_positions = load_input_data()
    num_user = d_hat.shape[1]

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("model.pkl 파일이 현재 폴더에 있어야 합니다. train.py를 실행하거나 제출용 model.pkl을 포함하세요.")

    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    anchor_error_model = artifact["anchor_error_model"]
    X = build_features(d_hat, bs_positions, anchor_error_model)

    if artifact.get("target_type") == "ensemble":
        weights = np.asarray(artifact["ensemble_weights"], dtype=float)
        predictions = []
        for _, model in artifact["position_model"]:
            predictions.append(model.predict(X))
        pred = np.zeros_like(predictions[0], dtype=float)
        for w, p_model in zip(weights, predictions):
            pred += w * p_model
    else:
        _, model = artifact["position_model"]
        pred = model.predict(X)

    pred[:, 0] = np.clip(pred[:, 0], -80.0, 80.0)
    pred[:, 1] = np.clip(pred[:, 1], -50.0, 50.0)

    p_hat = np.asarray(pred.T, dtype=float)
    if p_hat.shape != (2, num_user):
        raise ValueError(f"p_hat shape 오류: {p_hat.shape}, expected={(2, num_user)}")
    return p_hat


if __name__ == "__main__":
    output = main()
    print("p_hat shape:", output.shape)
