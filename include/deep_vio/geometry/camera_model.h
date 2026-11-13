#pragma once

#include <Eigen/Core>
#include <optional>

#include "deep_vio/dataset/euroc_dataset.h"

namespace deep_vio {

// Pinhole camera with radial-tangential (Brown-Conrady / OpenCV k1 k2 p1 p2) distortion.
//   p_C -> normalized x = (X/Z, Y/Z) -> distorted xd = d(x) -> pixel u = K xd
class PinholeRadtan {
 public:
  PinholeRadtan() = default;
  explicit PinholeRadtan(const CameraCalib& c)
      : fx_(c.fx), fy_(c.fy), cx_(c.cx), cy_(c.cy), dist_(c.dist), w_(c.width), h_(c.height) {}

  // Distortion on the normalized image plane. J (optional) = d xd / d x.
  Eigen::Vector2d distort(const Eigen::Vector2d& x, Eigen::Matrix2d* J = nullptr) const;
  // Inverse of distort() via Gauss-Newton; converges in a few iterations for EuRoC-level distortion.
  Eigen::Vector2d undistortNormalized(const Eigen::Vector2d& xd) const;

  // Full projection of a camera-frame point; nullopt if behind the camera.
  // J (optional) = d pixel / d p_C.
  std::optional<Eigen::Vector2d> project(const Eigen::Vector3d& p_C,
                                         Eigen::Matrix<double, 2, 3>* J = nullptr) const;

  // Raw pixel -> undistorted normalized coordinates (x, y) with z = 1.
  Eigen::Vector2d pixelToNormalized(const Eigen::Vector2d& uv) const;
  // Raw pixel -> pixel of the equivalent ideal (undistorted) pinhole camera with the same K.
  Eigen::Vector2d undistortPixel(const Eigen::Vector2d& uv) const;
  Eigen::Vector2d normalizedToPixel(const Eigen::Vector2d& x) const {
    return {fx_ * x.x() + cx_, fy_ * x.y() + cy_};
  }

  bool inImage(const Eigen::Vector2d& uv, double border = 0) const {
    return uv.x() >= border && uv.y() >= border && uv.x() < w_ - border && uv.y() < h_ - border;
  }

  double fx() const { return fx_; }
  double fy() const { return fy_; }
  double cx() const { return cx_; }
  double cy() const { return cy_; }

 private:
  double fx_ = 1, fy_ = 1, cx_ = 0, cy_ = 0;
  Eigen::Vector4d dist_ = Eigen::Vector4d::Zero();
  int w_ = 0, h_ = 0;
};

}  // namespace deep_vio
