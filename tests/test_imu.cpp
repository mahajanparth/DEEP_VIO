#include <gtest/gtest.h>

#include <random>

#include "deep_vio/geometry/so3.h"
#include "deep_vio/imu/imu_segments.h"
#include "deep_vio/imu/preintegration.h"
#include "deep_vio/imu/static_initializer.h"

using namespace deep_vio;
using Eigen::Matrix3d;
using Eigen::Vector3d;

namespace {

// Smooth, EuRoC-like synthetic measurements.
std::vector<ImuSegment> syntheticMotion(int n, double dt) {
  std::vector<ImuSegment> m;
  for (int k = 0; k < n; ++k) {
    const double t = k * dt;
    m.push_back({Vector3d(0.3 * std::sin(2 * t), 9.81 + 0.5 * std::cos(3 * t), -0.4 * std::sin(t)),
                 Vector3d(0.4 * std::cos(t), -0.3 * std::sin(1.5 * t), 0.6 * std::sin(0.7 * t)), dt});
  }
  return m;
}

}  // namespace

TEST(ImuSegments, CoversIntervalExactly) {
  std::vector<ImuSample> s;
  for (int i = 0; i < 5; ++i) s.push_back({i * 5'000'000LL, Vector3d::Constant(i), Vector3d::Constant(i)});
  // Interval starts mid-sample and extends past the last sample.
  const auto seg = imuSegments(s, 2'500'000, 23'000'000);
  double total = 0;
  for (const auto& g : seg) total += g.dt;
  EXPECT_NEAR(total, 0.0205, 1e-12);
  EXPECT_NEAR(seg.front().dt, 0.0025, 1e-12);
  EXPECT_DOUBLE_EQ(seg.front().acc.x(), 0.5);  // midpoint of samples 0 and 1
  EXPECT_DOUBLE_EQ(seg.back().acc.x(), 4.0);   // hold after the last sample
}

TEST(Preintegration, BiasCorrectionIsFirstOrderAccurate) {
  const ImuPreintegrator::Noise noise{1e-3, 1e-2};
  const Vector3d bg0(0, 0, 0), ba0(0, 0, 0);
  const Vector3d dbg(2e-3, -1e-3, 3e-3), dba(0.02, -0.01, 0.03);
  ImuPreintegrator lin(noise, bg0, ba0), exact(noise, bg0 + dbg, ba0 + dba);
  for (const auto& m : syntheticMotion(200, 0.005)) {
    lin.integrate(m.acc, m.gyro, m.dt);
    exact.integrate(m.acc, m.gyro, m.dt);
  }
  const Vector3d bg = bg0 + dbg, ba = ba0 + dba;
  // First-order correction error should be much smaller than the bias effect itself.
  const double effect_v = (lin.deltaV() - exact.deltaV()).norm();
  const double err_v = (lin.deltaV(bg, ba) - exact.deltaV()).norm();
  const double err_p = (lin.deltaP(bg, ba) - exact.deltaP()).norm();
  const double err_R = so3::log(lin.deltaR(bg).transpose() * exact.deltaR()).norm();
  EXPECT_LT(err_v, 0.02 * effect_v);
  EXPECT_LT(err_p, 1e-4);
  EXPECT_LT(err_R, 1e-6);
}

TEST(Preintegration, PredictsFreeFallAndRest) {
  ImuPreintegrator pre({1e-3, 1e-2}, Vector3d::Zero(), Vector3d::Zero());
  // At rest (z up): accelerometer reads +g, gyro reads zero.
  for (int i = 0; i < 200; ++i) pre.integrate(Vector3d(0, 0, 9.81), Vector3d::Zero(), 0.005);
  const ImuPreintegrator::State s0{Matrix3d::Identity(), Vector3d::Zero(), Vector3d::Zero()};
  const auto s1 = pre.predict(s0, Vector3d(0, 0, -9.81), Vector3d::Zero(), Vector3d::Zero());
  EXPECT_LT(s1.p.norm(), 1e-9);
  EXPECT_LT(s1.v.norm(), 1e-9);
}

TEST(StaticInitializer, RecoversGravityDirectionAndGyroBias) {
  StaticInitializer::Params p;
  StaticInitializer init(p);
  const Matrix3d R_WB = so3::exp(Vector3d(0.3, -1.2, 0.7));
  const Vector3d bg(0.003, -0.02, 0.08);
  std::mt19937 rng(3);
  std::normal_distribution<double> n(0, 0.01);
  for (int i = 0; i < 400; ++i) {
    ImuSample s;
    s.t_ns = i * 5'000'000LL;
    s.acc = R_WB.transpose() * Vector3d(0, 0, 9.81) + Vector3d(n(rng), n(rng), n(rng));
    s.gyro = bg + 0.1 * Vector3d(n(rng), n(rng), n(rng));
    init.addImu(s);
  }
  const auto r = init.tryInitialize();
  ASSERT_TRUE(r);
  EXPECT_LT((r->bg - bg).norm(), 1e-3);
  // Gravity must map to world z (yaw is arbitrary, so compare only the up direction).
  const Vector3d up_est = r->R_WB.transpose() * Vector3d::UnitZ();
  const Vector3d up_true = R_WB.transpose() * Vector3d::UnitZ();
  EXPECT_LT((up_est - up_true).norm(), 1e-3);
}

TEST(StaticInitializer, RejectsMotion) {
  StaticInitializer init(StaticInitializer::Params{});
  for (int i = 0; i < 400; ++i) {
    ImuSample s;
    s.t_ns = i * 5'000'000LL;
    s.acc = Vector3d(0, 0, 9.81 + 2.0 * std::sin(i * 0.1));
    init.addImu(s);
  }
  EXPECT_FALSE(init.tryInitialize());
}
