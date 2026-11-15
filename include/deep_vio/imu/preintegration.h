#pragma once

#include <Eigen/Core>

namespace deep_vio {

// Hand-written on-manifold IMU preintegration (Forster et al., "On-Manifold Preintegration for
// Real-Time Visual-Inertial Odometry", TRO 2017). It is used to learn and verify the math. The
// estimator itself uses gtsam::PreintegratedCombinedMeasurements, and tests check the two agree.
//
// Between keyframes i and j, with bias estimate b:
//   dR = prod Exp((w_k - bg) dt)
//   dv = sum dR_k (a_k - ba) dt
//   dp = sum dv_k dt + 0.5 dR_k (a_k - ba) dt^2
// These relate the states by
//   R_j = R_i dR,  v_j = v_i + g T + R_i dv,  p_j = p_i + v_i T + 0.5 g T^2 + R_i dp.
class ImuPreintegrator {
 public:
  struct Noise {
    double gyro_noise_density;   // sigma_g [rad/s/sqrt(Hz)]
    double accel_noise_density;  // sigma_a [m/s^2/sqrt(Hz)]
  };
  struct State {
    Eigen::Matrix3d R;
    Eigen::Vector3d v, p;
  };

  ImuPreintegrator(const Noise& noise, const Eigen::Vector3d& bg, const Eigen::Vector3d& ba);

  void integrate(const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro, double dt);
  void reset(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba);

  // Deltas corrected to first order for a new bias estimate (avoids re-integration).
  Eigen::Matrix3d deltaR(const Eigen::Vector3d& bg) const;
  Eigen::Vector3d deltaV(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) const;
  Eigen::Vector3d deltaP(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) const;

  State predict(const State& s_i, const Eigen::Vector3d& gravity, const Eigen::Vector3d& bg,
                const Eigen::Vector3d& ba) const;

  const Eigen::Matrix3d& deltaR() const { return dR_; }
  const Eigen::Vector3d& deltaV() const { return dv_; }
  const Eigen::Vector3d& deltaP() const { return dp_; }
  double deltaT() const { return dt_; }
  // Covariance of [dtheta, dp, dv] (same order as GTSAM's ImuFactor).
  const Eigen::Matrix<double, 9, 9>& covariance() const { return cov_; }

 private:
  Noise noise_;
  Eigen::Vector3d bg_, ba_;  // linearization point of the bias
  Eigen::Matrix3d dR_;
  Eigen::Vector3d dv_, dp_;
  double dt_ = 0;
  Eigen::Matrix<double, 9, 9> cov_;
  // Bias Jacobians.
  Eigen::Matrix3d dR_dbg_, dv_dbg_, dv_dba_, dp_dbg_, dp_dba_;
};

}  // namespace deep_vio
