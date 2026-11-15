# Classical baseline (M10)

Pipeline: KLT front-end and GTSAM `BatchFixedLagSmoother` (3 s lag) with `CombinedImuFactor`,
Huber projection factors and ZUPT. Config `config/euroc_mono.yaml`, monocular cam0 + imu0.
Measured 2026-11-13 on an RTX 3070 Ti laptop (CPU only, i.e. no GPU in this pipeline).
Reproduce with `PYTHON=python ./scripts/eval_euroc.sh`.

ATE = RMSE of position error after aligning the estimate to ground truth (`scripts/eval_ate.py`).
The estimate is the online (causal) pose at every camera frame.

| sequence | length [m] | ATE SE3 RMSE [m] | ATE Sim3 RMSE [m] | scale | drift [%] |
|---|---|---|---|---|---|
| MH_01_easy | 73.1 | 0.114 | 0.112 | 0.995 | 0.16 |
| MH_02_easy | 63.7 | 0.201 | 0.197 | 0.991 | 0.32 |
| MH_03_medium | 127.4 | 0.273 | 0.270 | 0.989 | 0.21 |
| MH_04_difficult | 88.6 | 0.235 | 0.193 | 0.981 | 0.26 |
| MH_05_difficult | 97.5 | 0.244 | 0.205 | 0.981 | 0.25 |

Timing per frame: front-end 6–8 ms. Back-end averages 27–45 ms, with spikes up to about
340 ms when LM re-solves the full window.

Notes
- Every sequence starts after a static initialization window, so the first seconds of
  MH_01/02 (hand-held shaking) are not estimated.
- The scale is 1-2 % short on the difficult sequences, and SE3 vs Sim3 differ more there.
  Candidates to investigate: IMU noise tuning, accel-bias observability, more keyframes during
  aggressive motion.
