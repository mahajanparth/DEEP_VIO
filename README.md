# DEEP_VIO

Monocular visual-inertial odometry in C++, built from scratch to learn SLAM/VIO. The classical
pipeline below is the baseline that later stages (a learned front-end and a learned inertial
model) will be measured against.

## Pipeline

```
 EuRoC cam0 (20 Hz) ──► KLT tracker ──► feature tracks (id, undistorted px)
                                                     │
 EuRoC imu0 (200 Hz) ─► IMU preintegration ───────┐  │
                                                  ▼  ▼
                     static init ──► keyframe? ──► fixed-lag smoother (GTSAM)
                                         │          X,V,B per keyframe + landmarks
                                         └──no──► IMU-propagated pose (20 Hz output)
```

1. **Dataset** (`src/dataset`): parses EuRoC CSV/YAML and yields one packet per image holding
   the IMU samples in `(t_prev, t]`. Timestamps stay int64 nanoseconds throughout.
2. **Front-end** (`src/frontend/klt_tracker.cpp`): CLAHE, then pyramidal Lucas-Kanade with a
   forward-backward check, then fundamental-matrix RANSAC, then Shi-Tomasi corners with a
   minimum spacing. About 150 features per frame, at roughly 5 ms per frame.
3. **Static initialization** (`src/imu/static_initializer.cpp`): waits for 1 s of stationary
   IMU data. The mean accelerometer reading gives gravity, which fixes roll and pitch. The mean
   gyro reading gives the gyro bias. Yaw and position are unobservable and set to 0.
4. **IMU preintegration**: GTSAM's `PreintegratedCombinedMeasurements` summarizes all IMU
   samples between keyframes into one relative-motion constraint. A hand-written version in
   `src/imu/preintegration.cpp` (Forster et al. 2017) is checked against GTSAM in the tests.
5. **Keyframes**: a new keyframe is made when ≥ 0.5 s have passed, or after ≥ 0.1 s if the mean
   parallax is ≥ 10 px or fewer than 60 % of the last keyframe's features are still tracked.
6. **Back-end** (`src/backend/vio_estimator.cpp`): a factor graph over a 3 s sliding window.
   - `X(k)` pose, `V(k)` velocity, `B(k)` IMU bias per keyframe; `L(id)` 3-D landmarks.
   - `CombinedImuFactor` between consecutive keyframes (motion plus bias random walk).
   - `GenericProjectionFactor` with a Huber kernel for each landmark observation.
   - A landmark is created once a track has ≥ 3 keyframe observations with ≥ 1° parallax. It
     is triangulated by DLT (`src/geometry/triangulation.cpp`) and kept only if its
     reprojection error is < 3 px in every view.
   - Observations that disagree with the IMU-predicted pose by more than 6 px are gated out.
   - Zero-velocity update: when the image is static and the accelerometer is quiet, the
     keyframe gets a `v = 0` prior.
   - Old states are marginalized into a linear prior by `BatchFixedLagSmoother`.

Mono VIO gets **metric scale from the IMU**: the accelerometer measures metric acceleration,
which fixes the scale of the visual structure. This is why the Sim(3)-aligned and
SE(3)-aligned errors below are nearly identical.

## Build

Needs Eigen 3.4, OpenCV 4 and GTSAM 4.3. The apt `libgtsam` on this machine targets Ubuntu
20.04 (Eigen 3.3.7 / TBB 2020) and does not link here, so GTSAM is built locally first:

```bash
./scripts/build_gtsam.sh            # ~15 min, installs to third_party/install
cmake -S . -B build && cmake --build build -j
```

## Run

```bash
MH01=datasets/euroc/machine_hall/MH_01_easy/MH_01_easy/mav0
./build/play_euroc  $MH01                 # raw vs undistorted images, IMU per frame
./build/track_euroc $MH01 --viz           # KLT tracks (blue = new, red = long-lived)
./build/run_euroc   $MH01 config/euroc_mono.yaml results/runs/MH_01_easy.txt --viz
python scripts/eval_ate.py $MH01/state_groundtruth_estimate0/data.csv \
       results/runs/MH_01_easy.txt --plot mh01.png
PYTHON=python ./scripts/eval_euroc.sh     # all MH sequences -> results/runs/summary.md
```

## Tests

```bash
EUROC_DIR=$PWD/$MH01 ./build/deep_vio_tests   # dataset, SO(3), camera, triangulation, IMU
./build/deep_vio_gtsam_tests                  # hand-written math vs GTSAM, toy optimization
```

## Results (classical baseline)

See [results/baseline_classical.md](results/baseline_classical.md).
