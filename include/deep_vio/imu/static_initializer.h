#pragma once

#include <deque>
#include <optional>

#include "deep_vio/dataset/euroc_dataset.h"

namespace deep_vio {

struct ImuInitResult {
  int64_t t_ns;
  Eigen::Matrix3d R_WB;  // gravity-aligned (world z up), arbitrary yaw
  Eigen::Vector3d bg, ba;
};

// Static (stand-still) initialization. While the platform is at rest, the accelerometer measures
// only the reaction to gravity, a = R_BW * (0, 0, g), and the gyro measures only its bias.
// This gives roll/pitch, the gyro bias, and the accel bias component along gravity. Yaw is
// unobservable, so it is set to the minimal rotation.
class StaticInitializer {
 public:
  struct Params {
    double window_s = 1.0;
    double max_acc_std = 0.08;   // std of |a| over the window [m/s^2]
    double max_gyro_std = 0.02;  // std of each gyro axis over the window [rad/s]
    double gravity = 9.81;
  };
  explicit StaticInitializer(const Params& p) : p_(p) {}

  void addImu(const ImuSample& s);
  // Succeeds once the last `window_s` seconds of IMU data look stationary.
  std::optional<ImuInitResult> tryInitialize() const;

 private:
  Params p_;
  std::deque<ImuSample> buf_;
};

}  // namespace deep_vio
