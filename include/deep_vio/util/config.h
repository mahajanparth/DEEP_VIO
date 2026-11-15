#pragma once

#include <string>

#include "deep_vio/frontend/klt_tracker.h"
#include "deep_vio/imu/static_initializer.h"

namespace deep_vio {

struct ImuNoiseConfig {
  // Values used by the estimator. Non-positive means "dataset value x inflation".
  double gyro_noise_density = -1, accel_noise_density = -1;
  double gyro_random_walk = -1, accel_random_walk = -1;
  double noise_inflation = 1.0;
  double gravity = 9.81;
};

struct KeyframeConfig {
  double min_dt = 0.1;             // [s]
  double max_dt = 0.5;             // [s]
  double min_parallax_px = 10.0;   // mean feature displacement since the last keyframe
  double min_track_ratio = 0.6;    // fraction of last-keyframe features still tracked
};

struct BackendConfig {
  std::string smoother = "batch";  // "batch" (LM) or "isam2"
  double lag_s = 3.0;
  int lm_max_iterations = 10;
  double pixel_sigma = 1.0;
  double huber_k = 1.5;            // in units of sigma
  int min_track_obs = 3;           // keyframe observations before triangulating
  double min_parallax_deg = 1.0;
  double max_reproj_px = 3.0;      // triangulation acceptance and prediction gating
  double min_depth = 0.2, max_depth = 40.0;
  // Zero-velocity update when the image is static.
  double zupt_max_parallax_px = 0.5;
  double zupt_max_acc_std = 0.2;
  double zupt_vel_sigma = 0.01;
  // Initial state priors.
  double prior_rot_sigma = 1e-3, prior_pos_sigma = 1e-3, prior_vel_sigma = 0.05;
  double prior_ba_sigma = 0.1, prior_bg_sigma = 0.01;
};

struct VioConfig {
  KltParams klt;
  StaticInitializer::Params init;
  ImuNoiseConfig imu;
  KeyframeConfig kf;
  BackendConfig backend;
};

// Reads an OpenCV-style YAML file (starting with "%YAML:1.0"). Missing keys keep their defaults.
VioConfig loadConfig(const std::string& path);

}  // namespace deep_vio
