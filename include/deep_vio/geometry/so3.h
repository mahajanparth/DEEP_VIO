#pragma once

#include <Eigen/Core>

// Hand-written SO(3) toolkit. Conventions follow Forster et al. (TRO 2017) and GTSAM:
// perturbations are on the right, R * Exp(dtheta).
namespace deep_vio::so3 {

// Skew-symmetric matrix: hat(w) * v == w.cross(v).
Eigen::Matrix3d hat(const Eigen::Vector3d& w);
Eigen::Vector3d vee(const Eigen::Matrix3d& W);

// Rodrigues' formula: Exp(theta) = I + sin|t|/|t| [t]x + (1-cos|t|)/|t|^2 [t]x^2.
Eigen::Matrix3d exp(const Eigen::Vector3d& theta);
// Inverse of exp for rotation angles in [0, pi].
Eigen::Vector3d log(const Eigen::Matrix3d& R);

// Right Jacobian Jr: Exp(theta + d) ~= Exp(theta) * Exp(Jr(theta) d).
Eigen::Matrix3d rightJacobian(const Eigen::Vector3d& theta);
// Inverse right Jacobian: Log(Exp(theta) * Exp(d)) ~= theta + Jr^-1(theta) d.
Eigen::Matrix3d rightJacobianInverse(const Eigen::Vector3d& theta);

// Projects an arbitrary 3x3 matrix onto the closest rotation (SVD).
Eigen::Matrix3d normalize(const Eigen::Matrix3d& M);

}  // namespace deep_vio::so3
