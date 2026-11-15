#!/usr/bin/env python3
"""Absolute trajectory error (ATE) of a TUM-format estimate against EuRoC ground truth.

    python scripts/eval_ate.py <gt data.csv> <estimate.txt> [--plot out.png] [--json out.json]

Steps: associate each estimated pose with the nearest ground-truth sample (<= 5 ms apart),
align with Umeyama's method (SE(3) and Sim(3)), and report the RMSE of position errors.
Only positions are compared, the standard ATE used in VIO papers.
"""
import argparse
import json

import numpy as np


def load_euroc_gt(path):
    d = np.loadtxt(path, delimiter=",", comments="#")
    return d[:, 0].astype(np.int64), d[:, 1:4]


def load_tum(path):
    # Parse timestamps as text so nanoseconds survive (float64 has ~0.2 us resolution here).
    t, p = [], []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            s = line.split()
            sec, _, frac = s[0].partition(".")
            t.append(int(sec) * 1_000_000_000 + int(frac.ljust(9, "0")[:9]))
            p.append([float(x) for x in s[1:4]])
    return np.array(t, dtype=np.int64), np.array(p)


def associate(t_gt, t_est, max_dt_ns=5_000_000):
    idx = np.clip(np.searchsorted(t_gt, t_est), 1, len(t_gt) - 1)
    left, right = t_gt[idx - 1], t_gt[idx]
    idx = np.where(np.abs(t_est - left) < np.abs(right - t_est), idx - 1, idx)
    ok = np.abs(t_gt[idx] - t_est) <= max_dt_ns
    return idx[ok], np.nonzero(ok)[0]


def umeyama(src, dst, with_scale):
    """Finds s, R, t minimizing sum ||dst - (s R src + t)||^2 (Umeyama, PAMI 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1  # avoid a reflection
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum() if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def ate(gt, est, with_scale):
    s, R, t = umeyama(est, gt, with_scale)
    aligned = (s * (R @ est.T)).T + t
    err = np.linalg.norm(aligned - gt, axis=1)
    return float(np.sqrt((err**2).mean())), float(s), aligned, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt")
    ap.add_argument("est")
    ap.add_argument("--plot")
    ap.add_argument("--json")
    a = ap.parse_args()

    t_gt, p_gt = load_euroc_gt(a.gt)
    t_est, p_est = load_tum(a.est)
    i_gt, i_est = associate(t_gt, t_est)
    gt, est = p_gt[i_gt], p_est[i_est]

    rmse_se3, _, aligned, err = ate(gt, est, with_scale=False)
    rmse_sim3, scale, _, _ = ate(gt, est, with_scale=True)
    length = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
    res = {
        "poses": int(len(gt)),
        "duration_s": float((t_est[i_est[-1]] - t_est[i_est[0]]) * 1e-9),
        "gt_length_m": length,
        "ate_se3_rmse_m": rmse_se3,
        "ate_se3_max_m": float(err.max()),
        "ate_sim3_rmse_m": rmse_sim3,
        "scale": scale,
        "drift_pct": 100 * rmse_se3 / length,
    }
    for k, v in res.items():
        print(f"{k:16s} {v:.4f}" if isinstance(v, float) else f"{k:16s} {v}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)
    if a.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
        ax[0].plot(gt[:, 0], gt[:, 1], "k--", lw=1.2, label="ground truth")
        sc = ax[0].scatter(aligned[:, 0], aligned[:, 1], c=err, s=2, cmap="viridis")
        fig.colorbar(sc, ax=ax[0], label="position error [m]")
        ax[0].set(xlabel="x [m]", ylabel="y [m]", title="Top view (SE(3)-aligned)", aspect="equal")
        ax[0].legend(loc="best")
        tt = (t_est[i_est] - t_est[i_est[0]]) * 1e-9
        ax[1].plot(tt, err, lw=1)
        ax[1].set(xlabel="time [s]", ylabel="error [m]",
                  title=f"ATE RMSE {rmse_se3:.3f} m  (Sim3 {rmse_sim3:.3f} m, scale {scale:.3f})")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=110)


if __name__ == "__main__":
    main()
