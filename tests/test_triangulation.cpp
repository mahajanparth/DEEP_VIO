#include <gtest/gtest.h>

#include <random>

#include "deep_vio/geometry/so3.h"
#include "deep_vio/geometry/triangulation.h"

using namespace deep_vio;

namespace {
Eigen::Isometry3d cameraAt(const Eigen::Vector3d& c, const Eigen::Vector3d& rotvec) {
  Eigen::Isometry3d T_WC = Eigen::Isometry3d::Identity();
  T_WC.linear() = so3::exp(rotvec);
  T_WC.translation() = c;
  return T_WC.inverse();  // T_CW
}
Eigen::Vector2d observe(const Eigen::Isometry3d& T_CW, const Eigen::Vector3d& p_W) {
  const Eigen::Vector3d p = T_CW * p_W;
  return p.head<2>() / p.z();
}
}  // namespace

TEST(Triangulation, ExactWithoutNoise) {
  const Eigen::Vector3d p_W(0.5, -0.3, 4.0);
  std::vector<Eigen::Isometry3d> T;
  std::vector<Eigen::Vector2d> x;
  for (int i = 0; i < 4; ++i) {
    T.push_back(cameraAt({0.2 * i, 0.05 * i, 0}, {0, 0.02 * i, 0}));
    x.push_back(observe(T.back(), p_W));
  }
  const auto est = triangulateDLT(T, x);
  ASSERT_TRUE(est);
  EXPECT_NEAR((*est - p_W).norm(), 0, 1e-9);
  EXPECT_GT(maxParallaxAngle(T, p_W), 0.1);
}

TEST(Triangulation, RobustToPixelNoise) {
  std::mt19937 rng(42);
  std::normal_distribution<double> noise(0, 1.0 / 460.0);  // ~1 px at EuRoC focal length
  const Eigen::Vector3d p_W(-0.4, 0.2, 3.0);
  std::vector<Eigen::Isometry3d> T;
  std::vector<Eigen::Vector2d> x;
  for (int i = 0; i < 5; ++i) {
    T.push_back(cameraAt({0.15 * i, 0, 0.05 * i}, {0.01 * i, -0.01 * i, 0}));
    x.push_back(observe(T.back(), p_W) + Eigen::Vector2d(noise(rng), noise(rng)));
  }
  const auto est = triangulateDLT(T, x);
  ASSERT_TRUE(est);
  EXPECT_LT((*est - p_W).norm(), 0.1);
}

TEST(Triangulation, NeedsTwoViews) {
  EXPECT_FALSE(triangulateDLT({Eigen::Isometry3d::Identity()}, {Eigen::Vector2d::Zero()}));
}
