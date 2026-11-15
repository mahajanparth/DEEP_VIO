#include "deep_vio/util/config.h"

#include <opencv2/core/persistence.hpp>
#include <stdexcept>

namespace deep_vio {

namespace {

template <typename T>
void read(const cv::FileNode& parent, const char* key, T& value) {
  const cv::FileNode n = parent[key];
  if (n.empty()) return;
  if constexpr (std::is_same_v<T, bool>) {
    value = static_cast<int>(n) != 0;
  } else if constexpr (std::is_same_v<T, std::string>) {
    value = static_cast<std::string>(n);
  } else {
    value = static_cast<T>(static_cast<double>(n));
  }
}

}  // namespace

VioConfig loadConfig(const std::string& path) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) throw std::runtime_error("Cannot open config " + path);
  VioConfig c;

  const auto t = fs["tracker"];
  read(t, "max_features", c.klt.max_features);
  read(t, "min_distance", c.klt.min_distance);
  read(t, "quality_level", c.klt.quality_level);
  read(t, "win_size", c.klt.win_size);
  read(t, "pyramid_levels", c.klt.pyramid_levels);
  read(t, "fb_threshold", c.klt.fb_threshold);
  read(t, "ransac_threshold", c.klt.ransac_threshold);
  read(t, "use_clahe", c.klt.use_clahe);

  const auto i = fs["init"];
  read(i, "window_s", c.init.window_s);
  read(i, "max_acc_std", c.init.max_acc_std);
  read(i, "max_gyro_std", c.init.max_gyro_std);

  const auto m = fs["imu"];
  read(m, "gyro_noise_density", c.imu.gyro_noise_density);
  read(m, "accel_noise_density", c.imu.accel_noise_density);
  read(m, "gyro_random_walk", c.imu.gyro_random_walk);
  read(m, "accel_random_walk", c.imu.accel_random_walk);
  read(m, "noise_inflation", c.imu.noise_inflation);
  read(m, "gravity", c.imu.gravity);
  c.init.gravity = c.imu.gravity;

  const auto k = fs["keyframe"];
  read(k, "min_dt", c.kf.min_dt);
  read(k, "max_dt", c.kf.max_dt);
  read(k, "min_parallax_px", c.kf.min_parallax_px);
  read(k, "min_track_ratio", c.kf.min_track_ratio);

  const auto b = fs["backend"];
  read(b, "smoother", c.backend.smoother);
  read(b, "lag_s", c.backend.lag_s);
  read(b, "lm_max_iterations", c.backend.lm_max_iterations);
  read(b, "pixel_sigma", c.backend.pixel_sigma);
  read(b, "huber_k", c.backend.huber_k);
  read(b, "min_track_obs", c.backend.min_track_obs);
  read(b, "min_parallax_deg", c.backend.min_parallax_deg);
  read(b, "max_reproj_px", c.backend.max_reproj_px);
  read(b, "min_depth", c.backend.min_depth);
  read(b, "max_depth", c.backend.max_depth);
  read(b, "zupt_max_parallax_px", c.backend.zupt_max_parallax_px);
  read(b, "zupt_max_acc_std", c.backend.zupt_max_acc_std);
  read(b, "zupt_vel_sigma", c.backend.zupt_vel_sigma);
  read(b, "prior_rot_sigma", c.backend.prior_rot_sigma);
  read(b, "prior_pos_sigma", c.backend.prior_pos_sigma);
  read(b, "prior_vel_sigma", c.backend.prior_vel_sigma);
  read(b, "prior_ba_sigma", c.backend.prior_ba_sigma);
  read(b, "prior_bg_sigma", c.backend.prior_bg_sigma);
  return c;
}

}  // namespace deep_vio
