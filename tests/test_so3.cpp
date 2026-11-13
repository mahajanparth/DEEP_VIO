#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include "deep_vio/geometry/so3.h"

using namespace deep_vio;
using Eigen::Matrix3d;
using Eigen::Vector3d;

namespace {
std::vector<Vector3d> testVectors() {
  return {Vector3d(0.1, -0.2, 0.3), Vector3d(1e-9, 0, 0), Vector3d(1.2, 0.5, -2.0),
          Vector3d(0, 0, 3.1), Vector3d(-0.7, 0.01, 0.4)};
}
}  // namespace

TEST(SO3, HatVee) {
  const Vector3d w(1, 2, 3), v(-4, 5, 0.5);
  EXPECT_TRUE((so3::hat(w) * v).isApprox(w.cross(v)));
  EXPECT_TRUE(so3::vee(so3::hat(w)).isApprox(w));
}

TEST(SO3, ExpMatchesAngleAxisAndLogInverts) {
  for (const auto& t : testVectors()) {
    const Matrix3d R = so3::exp(t);
    const Matrix3d R_ref = t.norm() > 0 ? Eigen::AngleAxisd(t.norm(), t.normalized()).toRotationMatrix()
                                        : Matrix3d::Identity();
    EXPECT_TRUE(R.isApprox(R_ref, 1e-12)) << t.transpose();
    EXPECT_TRUE(so3::log(R).isApprox(t, 1e-9)) << t.transpose();
    EXPECT_NEAR((R.transpose() * R - Matrix3d::Identity()).norm(), 0, 1e-12);
    EXPECT_NEAR(R.determinant(), 1.0, 1e-12);
  }
}

TEST(SO3, LogNearPi) {
  const Vector3d t = Vector3d(1, -2, 0.5).normalized() * (M_PI - 1e-8);
  EXPECT_TRUE(so3::exp(so3::log(so3::exp(t))).isApprox(so3::exp(t), 1e-6));
}

TEST(SO3, RightJacobianMatchesNumeric) {
  for (const auto& t : testVectors()) {
    const Matrix3d Jr = so3::rightJacobian(t);
    EXPECT_TRUE((so3::rightJacobianInverse(t) * Jr).isApprox(Matrix3d::Identity(), 1e-9));
    // Exp(t + d) ~= Exp(t) Exp(Jr d)  =>  Jr ~= Log(Exp(t)^T Exp(t + d)) / d
    Matrix3d Jnum;
    const double h = 1e-6;
    for (int i = 0; i < 3; ++i)
      Jnum.col(i) = so3::log(so3::exp(t).transpose() * so3::exp(t + Vector3d::Unit(i) * h)) / h;
    EXPECT_TRUE(Jr.isApprox(Jnum, 1e-5)) << "Jr=\n" << Jr << "\nnum=\n" << Jnum;
  }
}

TEST(SO3, NormalizeProjectsToRotation) {
  Matrix3d M = so3::exp(Vector3d(0.3, 0.2, -0.1));
  M(0, 1) += 1e-3;  // simulate numerical drift
  const Matrix3d R = so3::normalize(M);
  EXPECT_NEAR((R.transpose() * R - Matrix3d::Identity()).norm(), 0, 1e-12);
  EXPECT_LT((R - M).norm(), 2e-3);
}
