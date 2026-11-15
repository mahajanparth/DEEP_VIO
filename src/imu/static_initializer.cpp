#include "deep_vio/imu/static_initializer.h"

#include <cmath>

namespace deep_vio {

void StaticInitializer::addImu(const ImuSample& s) {
  buf_.push_back(s);
  const int64_t window_ns = static_cast<int64_t>(p_.window_s * 1e9);
  while (!buf_.empty() && s.t_ns - buf_.front().t_ns > window_ns) buf_.pop_front();
}

std::optional<ImuInitResult> StaticInitializer::tryInitialize() const {
  if (buf_.size() < 10) return std::nullopt;
  if ((buf_.back().t_ns - buf_.front().t_ns) * 1e-9 < 0.95 * p_.window_s) return std::nullopt;

  const double n = static_cast<double>(buf_.size());
  Eigen::Vector3d a_mean = Eigen::Vector3d::Zero(), w_mean = Eigen::Vector3d::Zero();
  double a_norm_mean = 0;
  for (const auto& s : buf_) {
    a_mean += s.acc;
    w_mean += s.gyro;
    a_norm_mean += s.acc.norm();
  }
  a_mean /= n;
  w_mean /= n;
  a_norm_mean /= n;

  double a_norm_var = 0;
  Eigen::Vector3d w_var = Eigen::Vector3d::Zero();
  for (const auto& s : buf_) {
    a_norm_var += std::pow(s.acc.norm() - a_norm_mean, 2);
    w_var += (s.gyro - w_mean).cwiseAbs2();
  }
  if (std::sqrt(a_norm_var / n) > p_.max_acc_std) return std::nullopt;
  if (std::sqrt(w_var.maxCoeff() / n) > p_.max_gyro_std) return std::nullopt;

  ImuInitResult r;
  r.t_ns = buf_.back().t_ns;
  // R_WB rotates the measured "up" direction (body frame) onto world +z.
  r.R_WB = Eigen::Quaterniond::FromTwoVectors(a_mean, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  r.bg = w_mean;
  // Only the accel bias component along gravity is observable at rest.
  r.ba = a_mean - p_.gravity * a_mean.normalized();
  return r;
}

}  // namespace deep_vio
