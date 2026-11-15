#pragma once

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/FixedLagSmoother.h>
#include <map>
#include <memory>
#include <unordered_map>

#include "deep_vio/dataset/euroc_dataset.h"
#include "deep_vio/frontend/feature_tracker.h"
#include "deep_vio/geometry/camera_model.h"
#include "deep_vio/imu/static_initializer.h"
#include "deep_vio/util/config.h"

namespace deep_vio {

// Tightly coupled monocular VIO with a sliding window (fixed-lag smoother) in GTSAM.
//
// Factor graph per keyframe k:
//   X(k): Pose3 T_WB    V(k): velocity in W    B(k): IMU bias    L(id): landmark in W
//   CombinedImuFactor(X,V,B @ k-1 -> X,V,B @ k)    preintegrated IMU + bias random walk
//   GenericProjectionFactor(X(k), L(id))           robust (Huber) reprojection error
//   Prior on V(k) = 0 (+ X(k-1)~X(k))              zero-velocity update while static
// Keys older than `lag_s` are marginalized into a linear prior by the smoother.
class VioEstimator {
 public:
  struct Output {
    int64_t t_ns = 0;
    gtsam::NavState state;
    gtsam::imuBias::ConstantBias bias;
    bool is_keyframe = false;
  };
  struct Stats {
    int keyframes = 0;
    int landmarks_in_window = 0;
    int new_landmarks = 0;
    int visual_factors = 0;
    int gated_observations = 0;
    bool zupt = false;
    double solve_ms = 0;
  };

  VioEstimator(const VioConfig& cfg, const CameraCalib& cam, const ImuCalib& imu);

  // Returns true once initialized; `output()` then holds the state at `pkt.t_ns`.
  bool process(const SyncPacket& pkt, const FeatureFrame& frame);

  bool initialized() const { return initialized_; }
  const Output& output() const { return out_; }
  const Stats& stats() const { return stats_; }
  std::vector<Eigen::Vector3d> landmarks() const;

 private:
  struct Observation {
    int kf;
    Eigen::Vector2d px;  // undistorted pixel
  };
  struct Track {
    std::vector<Observation> obs;
    bool in_graph = false;
    bool rejected = false;
  };

  void initialize(const SyncPacket& pkt, const FeatureFrame& frame, const ImuInitResult& init);
  void integrateImu(const SyncPacket& pkt);
  void addKeyframe(const SyncPacket& pkt, const FeatureFrame& frame, const gtsam::NavState& pred,
                   bool zupt);
  bool kfAlive(int kf) const;
  gtsam::Pose3 kfPose(int kf) const;
  double toSec(int64_t t_ns) const { return (t_ns - t0_ns_) * 1e-9; }
  // Reprojection error in pixels; negative if the point is behind the camera.
  double reprojError(const gtsam::Pose3& T_WB, const gtsam::Point3& p_W,
                     const Eigen::Vector2d& px) const;

  VioConfig cfg_;
  PinholeRadtan cam_model_;
  gtsam::Cal3_S2::shared_ptr K_;
  gtsam::Pose3 T_BC_;
  gtsam::SharedNoiseModel pixel_noise_;
  std::shared_ptr<gtsam::PreintegrationCombinedParams> imu_params_;
  std::unique_ptr<gtsam::PreintegratedCombinedMeasurements> pim_;
  std::unique_ptr<gtsam::FixedLagSmoother> smoother_;
  StaticInitializer static_init_;

  bool initialized_ = false;
  int64_t t0_ns_ = 0;
  int64_t last_frame_t_ns_ = 0;
  ImuSample last_imu_;
  std::vector<double> acc_norms_since_kf_;

  int kf_ = -1;                      // index of the latest keyframe
  std::map<int, int64_t> kf_times_;  // keyframe index -> timestamp
  int64_t last_kf_t_ns_ = 0;
  gtsam::NavState kf_state_;
  gtsam::imuBias::ConstantBias kf_bias_;
  std::unordered_map<uint64_t, Eigen::Vector2d> last_kf_px_;  // feature id -> undistorted px
  std::unordered_map<uint64_t, Track> tracks_;
  gtsam::Values estimate_;

  Output out_;
  Stats stats_;
};

}  // namespace deep_vio
