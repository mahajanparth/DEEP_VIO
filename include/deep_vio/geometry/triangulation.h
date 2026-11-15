#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <optional>
#include <vector>

namespace deep_vio {

// Linear (DLT) multi-view triangulation.
//   T_CW[i]: world-to-camera transform of view i.
//   x[i]:    undistorted normalized observation (X/Z, Y/Z) in view i.
// Each view contributes two rows of A X = 0 (X homogeneous):
//   x_u * P.row(2) - P.row(0) = 0,  x_v * P.row(2) - P.row(1) = 0,  P = [R_CW | t_CW]
// Returns nullopt if fewer than 2 views or the solution is at infinity.
std::optional<Eigen::Vector3d> triangulateDLT(const std::vector<Eigen::Isometry3d>& T_CW,
                                              const std::vector<Eigen::Vector2d>& x);

// Largest angle [rad] between the observation rays of a world point across the views.
double maxParallaxAngle(const std::vector<Eigen::Isometry3d>& T_CW, const Eigen::Vector3d& p_W);

}  // namespace deep_vio
