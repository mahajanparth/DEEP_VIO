// Cross-checks the hand-written math against GTSAM, which the backend relies on.
#include <gtest/gtest.h>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/SO3.h>
#include <gtsam/navigation/ImuFactor.h>

#include "deep_vio/geometry/so3.h"
#include "deep_vio/imu/preintegration.h"

using namespace deep_vio;
using Eigen::Matrix3d;
using Eigen::Vector3d;

TEST(GtsamConsistency, SO3ExpLogAndRightJacobian) {
  for (const Vector3d& t : {Vector3d(0.1, -0.2, 0.3), Vector3d(1.2, 0.5, -2.0), Vector3d(0, 0, 3.1)}) {
    EXPECT_TRUE(so3::exp(t).isApprox(gtsam::Rot3::Expmap(t).matrix(), 1e-12));
    EXPECT_TRUE(so3::log(so3::exp(t)).isApprox(gtsam::Rot3::Logmap(gtsam::Rot3::Expmap(t)), 1e-9));
    EXPECT_TRUE(so3::rightJacobian(t).isApprox(gtsam::SO3::ExpmapDerivative(t), 1e-9));
  }
}

TEST(GtsamConsistency, PreintegrationMatchesGtsam) {
  const ImuPreintegrator::Noise noise{1.7e-3, 2e-2};
  const Vector3d bg(0.01, -0.02, 0.03), ba(0.05, 0.1, -0.08);
  ImuPreintegrator mine(noise, bg, ba);

  auto p = gtsam::PreintegrationParams::MakeSharedU(9.81);
  p->setGyroscopeCovariance(Matrix3d::Identity() * noise.gyro_noise_density * noise.gyro_noise_density);
  p->setAccelerometerCovariance(Matrix3d::Identity() * noise.accel_noise_density *
                                noise.accel_noise_density);
  p->setIntegrationCovariance(Matrix3d::Zero());
  gtsam::PreintegratedImuMeasurements theirs(p, gtsam::imuBias::ConstantBias(ba, bg));

  const double dt = 0.005;
  for (int k = 0; k < 100; ++k) {  // 0.5 s at 200 Hz
    const double t = k * dt;
    const Vector3d acc(0.3 * std::sin(2 * t), 9.81 + 0.5 * std::cos(3 * t), -0.4 * std::sin(t));
    const Vector3d gyro(0.4 * std::cos(t), -0.3 * std::sin(1.5 * t), 0.6 * std::sin(0.7 * t));
    mine.integrate(acc, gyro, dt);
    theirs.integrateMeasurement(acc, gyro, dt);
  }
  EXPECT_NEAR(mine.deltaT(), theirs.deltaTij(), 1e-12);
  // GTSAM may integrate rotation in the tangent space, which differs at second order.
  EXPECT_LT(so3::log(mine.deltaR().transpose() * theirs.deltaRij().matrix()).norm(), 1e-5);
  EXPECT_LT((mine.deltaV() - theirs.deltaVij()).norm(), 1e-5);
  EXPECT_LT((mine.deltaP() - theirs.deltaPij()).norm(), 1e-5);
  const Eigen::Matrix<double, 9, 9> C = theirs.preintMeasCov();
  EXPECT_LT((mine.covariance() - C).norm() / C.norm(), 1e-2)
      << "mine:\n" << mine.covariance() << "\ngtsam:\n" << C;
}
