#include "deep_vio/imu/preintegration.h"

#include "deep_vio/geometry/so3.h"

namespace deep_vio {

using Eigen::Matrix3d;
using Eigen::Vector3d;

ImuPreintegrator::ImuPreintegrator(const Noise& noise, const Vector3d& bg, const Vector3d& ba)
    : noise_(noise) {
  reset(bg, ba);
}

void ImuPreintegrator::reset(const Vector3d& bg, const Vector3d& ba) {
  bg_ = bg;
  ba_ = ba;
  dR_.setIdentity();
  dv_.setZero();
  dp_.setZero();
  dt_ = 0;
  cov_.setZero();
  dR_dbg_.setZero();
  dv_dbg_.setZero();
  dv_dba_.setZero();
  dp_dbg_.setZero();
  dp_dba_.setZero();
}

void ImuPreintegrator::integrate(const Vector3d& acc, const Vector3d& gyro, double dt) {
  const Vector3d a = acc - ba_;
  const Vector3d w = gyro - bg_;
  const Matrix3d dR_inc = so3::exp(w * dt);
  const Matrix3d Jr = so3::rightJacobian(w * dt);
  const Matrix3d I = Matrix3d::Identity();
  const Matrix3d dR_a_hat = dR_ * so3::hat(a);

  // --- Covariance propagation: x_{k+1} = A x_k + B n_k, x = [dtheta, dp, dv], n = [n_g, n_a] ---
  Eigen::Matrix<double, 9, 9> A = Eigen::Matrix<double, 9, 9>::Identity();
  A.block<3, 3>(0, 0) = dR_inc.transpose();
  A.block<3, 3>(3, 0) = -0.5 * dR_a_hat * dt * dt;
  A.block<3, 3>(3, 6) = I * dt;
  A.block<3, 3>(6, 0) = -dR_a_hat * dt;
  Eigen::Matrix<double, 9, 6> B = Eigen::Matrix<double, 9, 6>::Zero();
  B.block<3, 3>(0, 0) = Jr * dt;
  B.block<3, 3>(3, 3) = 0.5 * dR_ * dt * dt;
  B.block<3, 3>(6, 3) = dR_ * dt;
  // Continuous noise density -> discrete covariance: sigma_d^2 = sigma^2 / dt.
  Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
  Q.topLeftCorner<3, 3>() = I * noise_.gyro_noise_density * noise_.gyro_noise_density / dt;
  Q.bottomRightCorner<3, 3>() = I * noise_.accel_noise_density * noise_.accel_noise_density / dt;
  cov_ = A * cov_ * A.transpose() + B * Q * B.transpose();

  // --- Bias Jacobians (must use dR_ before it is updated) ---
  dp_dba_ += dv_dba_ * dt - 0.5 * dR_ * dt * dt;
  dp_dbg_ += dv_dbg_ * dt - 0.5 * dR_a_hat * dR_dbg_ * dt * dt;
  dv_dba_ += -dR_ * dt;
  dv_dbg_ += -dR_a_hat * dR_dbg_ * dt;
  dR_dbg_ = dR_inc.transpose() * dR_dbg_ - Jr * dt;

  // --- Deltas: position uses the old velocity and rotation, so update it first ---
  dp_ += dv_ * dt + 0.5 * dR_ * a * dt * dt;
  dv_ += dR_ * a * dt;
  dR_ = so3::normalize(dR_ * dR_inc);
  dt_ += dt;
}

Matrix3d ImuPreintegrator::deltaR(const Vector3d& bg) const {
  return dR_ * so3::exp(dR_dbg_ * (bg - bg_));
}

Vector3d ImuPreintegrator::deltaV(const Vector3d& bg, const Vector3d& ba) const {
  return dv_ + dv_dbg_ * (bg - bg_) + dv_dba_ * (ba - ba_);
}

Vector3d ImuPreintegrator::deltaP(const Vector3d& bg, const Vector3d& ba) const {
  return dp_ + dp_dbg_ * (bg - bg_) + dp_dba_ * (ba - ba_);
}

ImuPreintegrator::State ImuPreintegrator::predict(const State& s, const Vector3d& g,
                                                  const Vector3d& bg, const Vector3d& ba) const {
  State out;
  out.R = s.R * deltaR(bg);
  out.v = s.v + g * dt_ + s.R * deltaV(bg, ba);
  out.p = s.p + s.v * dt_ + 0.5 * g * dt_ * dt_ + s.R * deltaP(bg, ba);
  return out;
}

}  // namespace deep_vio
