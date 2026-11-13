#include "deep_vio/geometry/so3.h"

#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>

namespace deep_vio::so3 {

namespace {
constexpr double kSmallAngle = 1e-8;
}

Eigen::Matrix3d hat(const Eigen::Vector3d& w) {
  Eigen::Matrix3d W;
  W << 0, -w.z(), w.y(),  //
      w.z(), 0, -w.x(),   //
      -w.y(), w.x(), 0;
  return W;
}

Eigen::Vector3d vee(const Eigen::Matrix3d& W) { return {W(2, 1), W(0, 2), W(1, 0)}; }

Eigen::Matrix3d exp(const Eigen::Vector3d& theta) {
  const double a = theta.norm();
  const Eigen::Matrix3d W = hat(theta);
  if (a < kSmallAngle) return Eigen::Matrix3d::Identity() + W + 0.5 * W * W;  // Taylor
  return Eigen::Matrix3d::Identity() + (std::sin(a) / a) * W + ((1 - std::cos(a)) / (a * a)) * W * W;
}

Eigen::Vector3d log(const Eigen::Matrix3d& R) {
  const double c = std::clamp(0.5 * (R.trace() - 1.0), -1.0, 1.0);
  const double a = std::acos(c);
  if (a < kSmallAngle) return 0.5 * vee(R - R.transpose());  // sin(a)/a -> 1
  if (M_PI - a < 1e-6) {
    // Near pi, sin(a) ~ 0 so the generic formula is ill-conditioned. Use R = 2nn^T - I.
    const Eigen::Matrix3d B = 0.5 * (R + Eigen::Matrix3d::Identity());
    int i;
    B.diagonal().maxCoeff(&i);
    Eigen::Vector3d n = B.col(i) / std::sqrt(B(i, i));
    // Fix the sign of n using the (small) antisymmetric part.
    if (n.dot(vee(R - R.transpose())) < 0) n = -n;
    return a * n;
  }
  return (a / (2 * std::sin(a))) * vee(R - R.transpose());
}

Eigen::Matrix3d rightJacobian(const Eigen::Vector3d& theta) {
  const double a = theta.norm();
  const Eigen::Matrix3d W = hat(theta);
  if (a < 1e-5) return Eigen::Matrix3d::Identity() - 0.5 * W + (1.0 / 6.0) * W * W;
  return Eigen::Matrix3d::Identity() - ((1 - std::cos(a)) / (a * a)) * W +
         ((a - std::sin(a)) / (a * a * a)) * W * W;
}

Eigen::Matrix3d rightJacobianInverse(const Eigen::Vector3d& theta) {
  const double a = theta.norm();
  const Eigen::Matrix3d W = hat(theta);
  if (a < 1e-5) return Eigen::Matrix3d::Identity() + 0.5 * W + (1.0 / 12.0) * W * W;
  return Eigen::Matrix3d::Identity() + 0.5 * W +
         (1.0 / (a * a) - (1 + std::cos(a)) / (2 * a * std::sin(a))) * W * W;
}

Eigen::Matrix3d normalize(const Eigen::Matrix3d& M) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
  if (R.determinant() < 0) {
    Eigen::Matrix3d U = svd.matrixU();
    U.col(2) *= -1;
    R = U * svd.matrixV().transpose();
  }
  return R;
}

}  // namespace deep_vio::so3
