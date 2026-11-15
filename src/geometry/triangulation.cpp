#include "deep_vio/geometry/triangulation.h"

#include <Eigen/SVD>
#include <algorithm>
#include <cmath>

namespace deep_vio {

std::optional<Eigen::Vector3d> triangulateDLT(const std::vector<Eigen::Isometry3d>& T_CW,
                                              const std::vector<Eigen::Vector2d>& x) {
  const size_t n = T_CW.size();
  if (n < 2 || x.size() != n) return std::nullopt;
  Eigen::MatrixXd A(2 * n, 4);
  for (size_t i = 0; i < n; ++i) {
    const Eigen::Matrix<double, 3, 4> P = T_CW[i].matrix().topRows<3>();
    A.row(2 * i) = x[i].x() * P.row(2) - P.row(0);
    A.row(2 * i + 1) = x[i].y() * P.row(2) - P.row(1);
  }
  // The solution is the right singular vector with the smallest singular value.
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  const Eigen::Vector4d X = svd.matrixV().col(3);
  if (std::abs(X[3]) < 1e-12) return std::nullopt;
  return Eigen::Vector3d(X.head<3>() / X[3]);
}

double maxParallaxAngle(const std::vector<Eigen::Isometry3d>& T_CW, const Eigen::Vector3d& p_W) {
  double best = 0;
  for (size_t i = 0; i < T_CW.size(); ++i) {
    const Eigen::Vector3d ci = T_CW[i].inverse().translation();
    for (size_t j = i + 1; j < T_CW.size(); ++j) {
      const Eigen::Vector3d cj = T_CW[j].inverse().translation();
      const Eigen::Vector3d a = (p_W - ci).normalized(), b = (p_W - cj).normalized();
      best = std::max(best, std::acos(std::clamp(a.dot(b), -1.0, 1.0)));
    }
  }
  return best;
}

}  // namespace deep_vio
