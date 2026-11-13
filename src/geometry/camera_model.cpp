#include "deep_vio/geometry/camera_model.h"

#include <Eigen/LU>

namespace deep_vio {

Eigen::Vector2d PinholeRadtan::distort(const Eigen::Vector2d& x, Eigen::Matrix2d* J) const {
  const double k1 = dist_[0], k2 = dist_[1], p1 = dist_[2], p2 = dist_[3];
  const double u = x.x(), v = x.y();
  const double r2 = u * u + v * v;
  const double radial = 1 + k1 * r2 + k2 * r2 * r2;
  Eigen::Vector2d xd(u * radial + 2 * p1 * u * v + p2 * (r2 + 2 * u * u),
                     v * radial + p1 * (r2 + 2 * v * v) + 2 * p2 * u * v);
  if (J) {
    const double dradial_dr2 = k1 + 2 * k2 * r2;
    // d r2 / du = 2u, d r2 / dv = 2v
    (*J)(0, 0) = radial + u * dradial_dr2 * 2 * u + 2 * p1 * v + p2 * 6 * u;
    (*J)(0, 1) = u * dradial_dr2 * 2 * v + 2 * p1 * u + p2 * 2 * v;
    (*J)(1, 0) = v * dradial_dr2 * 2 * u + p1 * 2 * u + 2 * p2 * v;
    (*J)(1, 1) = radial + v * dradial_dr2 * 2 * v + p1 * 6 * v + 2 * p2 * u;
  }
  return xd;
}

Eigen::Vector2d PinholeRadtan::undistortNormalized(const Eigen::Vector2d& xd) const {
  // Solve distort(x) = xd for x with Gauss-Newton, starting from x = xd.
  Eigen::Vector2d x = xd;
  for (int it = 0; it < 20; ++it) {
    Eigen::Matrix2d J;
    const Eigen::Vector2d r = distort(x, &J) - xd;
    if (r.squaredNorm() < 1e-24) break;
    x -= J.inverse() * r;
  }
  return x;
}

std::optional<Eigen::Vector2d> PinholeRadtan::project(const Eigen::Vector3d& p_C,
                                                      Eigen::Matrix<double, 2, 3>* J) const {
  if (p_C.z() <= 1e-6) return std::nullopt;
  const double iz = 1.0 / p_C.z();
  const Eigen::Vector2d x(p_C.x() * iz, p_C.y() * iz);
  Eigen::Matrix2d Jd;
  const Eigen::Vector2d xd = distort(x, J ? &Jd : nullptr);
  if (J) {
    Eigen::Matrix<double, 2, 3> Jx;  // d x / d p_C
    Jx << iz, 0, -p_C.x() * iz * iz,  //
        0, iz, -p_C.y() * iz * iz;
    const Eigen::Matrix2d Jk = Eigen::Vector2d(fx_, fy_).asDiagonal();
    *J = Jk * Jd * Jx;
  }
  return normalizedToPixel(xd);
}

Eigen::Vector2d PinholeRadtan::pixelToNormalized(const Eigen::Vector2d& uv) const {
  const Eigen::Vector2d xd((uv.x() - cx_) / fx_, (uv.y() - cy_) / fy_);
  return undistortNormalized(xd);
}

Eigen::Vector2d PinholeRadtan::undistortPixel(const Eigen::Vector2d& uv) const {
  return normalizedToPixel(pixelToNormalized(uv));
}

}  // namespace deep_vio
