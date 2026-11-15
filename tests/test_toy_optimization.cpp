// M3: the same nonlinear least-squares problem solved with a hand-written Gauss-Newton loop and
// with a GTSAM factor graph. Fit y = exp(a x^2 + b x + c) to noisy samples.
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <random>

namespace {

struct Sample {
  double x, y;
};

std::vector<Sample> makeData(const Eigen::Vector3d& abc) {
  std::mt19937 rng(1);
  std::normal_distribution<double> n(0, 0.05);
  std::vector<Sample> d;
  for (int i = 0; i < 100; ++i) {
    const double x = i / 100.0;
    d.push_back({x, std::exp(abc[0] * x * x + abc[1] * x + abc[2]) + n(rng)});
  }
  return d;
}

// Hand-written Gauss-Newton: solve (J^T J) dx = -J^T r and iterate.
Eigen::Vector3d gaussNewton(const std::vector<Sample>& data, Eigen::Vector3d est) {
  for (int it = 0; it < 50; ++it) {
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    for (const auto& s : data) {
      const double f = std::exp(est[0] * s.x * s.x + est[1] * s.x + est[2]);
      const double r = f - s.y;
      const Eigen::Vector3d J(f * s.x * s.x, f * s.x, f);  // d r / d (a, b, c)
      H += J * J.transpose();
      g += J * r;
    }
    const Eigen::Vector3d dx = H.ldlt().solve(-g);
    est += dx;
    if (dx.norm() < 1e-10) break;
  }
  return est;
}

// The GTSAM version: one unary factor per sample on a Vector3 variable.
class CurveFactor : public gtsam::NoiseModelFactorN<gtsam::Vector3> {
  using Base = gtsam::NoiseModelFactorN<gtsam::Vector3>;

 public:
  CurveFactor(gtsam::Key k, Sample s, const gtsam::SharedNoiseModel& m)
      : Base(m, k), s_(s) {}
  gtsam::Vector evaluateError(const gtsam::Vector3& abc, gtsam::OptionalMatrixType H) const override {
    const double f = std::exp(abc[0] * s_.x * s_.x + abc[1] * s_.x + abc[2]);
    if (H) *H = (gtsam::Matrix13() << f * s_.x * s_.x, f * s_.x, f).finished();
    return gtsam::Vector1(f - s_.y);
  }

 private:
  Sample s_;
};

}  // namespace

TEST(ToyOptimization, GaussNewtonAndGtsamAgree) {
  const Eigen::Vector3d truth(1.0, 2.0, 1.0);
  const auto data = makeData(truth);
  const Eigen::Vector3d init(0.5, 1.5, 0.5);

  const Eigen::Vector3d gn = gaussNewton(data, init);
  EXPECT_LT((gn - truth).norm(), 0.1);

  gtsam::NonlinearFactorGraph graph;
  const gtsam::Key k = gtsam::Symbol('p', 0);
  const auto noise = gtsam::noiseModel::Isotropic::Sigma(1, 0.05);
  for (const auto& s : data) graph.emplace_shared<CurveFactor>(k, s, noise);
  gtsam::Values values;
  values.insert(k, gtsam::Vector3(init));
  gtsam::LevenbergMarquardtParams params;
  // Marquardt scaling (lambda * diag(H) instead of lambda * I). Here H ~ 1e8, so plain lambda*I
  // damping hits GTSAM's lambda upper bound (1e5) before it can stop the steep model overshooting.
  params.setDiagonalDamping(true);
  const auto result = gtsam::LevenbergMarquardtOptimizer(graph, values, params).optimize();
  EXPECT_LT((result.at<gtsam::Vector3>(k) - gn).norm(), 1e-6);
}
