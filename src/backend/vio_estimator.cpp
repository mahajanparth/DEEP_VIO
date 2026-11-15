#include "deep_vio/backend/vio_estimator.h"

#include <chrono>
#include <cmath>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/BatchFixedLagSmoother.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/nonlinear/IncrementalFixedLagSmoother.h>
#include <iostream>
#include <unordered_set>

#include "deep_vio/geometry/triangulation.h"
#include "deep_vio/imu/imu_segments.h"

namespace deep_vio {

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::L;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;
using ProjectionFactor = gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>;
namespace nm = gtsam::noiseModel;

namespace {

double pick(double configured, double dataset, double inflation) {
  return configured > 0 ? configured : dataset * inflation;
}

}  // namespace

VioEstimator::VioEstimator(const VioConfig& cfg, const CameraCalib& cam, const ImuCalib& imu)
    : cfg_(cfg), cam_model_(cam), static_init_(cfg.init) {
  K_ = std::make_shared<gtsam::Cal3_S2>(cam.fx, cam.fy, 0.0, cam.cx, cam.cy);
  T_BC_ = gtsam::Pose3(cam.T_BC);
  pixel_noise_ = nm::Robust::Create(nm::mEstimator::Huber::Create(cfg.backend.huber_k),
                                    nm::Isotropic::Sigma(2, cfg.backend.pixel_sigma));

  const auto& in = cfg.imu;
  const double sg = pick(in.gyro_noise_density, imu.gyro_noise_density, in.noise_inflation);
  const double sa = pick(in.accel_noise_density, imu.accel_noise_density, in.noise_inflation);
  const double wg = pick(in.gyro_random_walk, imu.gyro_random_walk, in.noise_inflation);
  const double wa = pick(in.accel_random_walk, imu.accel_random_walk, in.noise_inflation);
  // "U" = navigation frame with z up, gravity = (0, 0, -g).
  imu_params_ = gtsam::PreintegrationCombinedParams::MakeSharedU(in.gravity);
  const gtsam::Matrix3 I = gtsam::Matrix3::Identity();
  imu_params_->setGyroscopeCovariance(I * sg * sg);
  imu_params_->setAccelerometerCovariance(I * sa * sa);
  imu_params_->setIntegrationCovariance(I * 1e-8);
  imu_params_->setBiasOmegaCovariance(I * wg * wg);
  imu_params_->setBiasAccCovariance(I * wa * wa);
  imu_params_->setBiasAccOmegaInit(gtsam::Matrix6::Identity() * 1e-5);
  pim_ = std::make_unique<gtsam::PreintegratedCombinedMeasurements>(imu_params_);

  if (cfg.backend.smoother == "isam2") {
    gtsam::ISAM2Params p;
    p.relinearizeThreshold = 0.01;
    p.relinearizeSkip = 1;
    smoother_ = std::make_unique<gtsam::IncrementalFixedLagSmoother>(cfg.backend.lag_s, p);
  } else {
    gtsam::LevenbergMarquardtParams p;
    p.maxIterations = cfg.backend.lm_max_iterations;
    smoother_ = std::make_unique<gtsam::BatchFixedLagSmoother>(cfg.backend.lag_s, p);
  }
}

bool VioEstimator::process(const SyncPacket& pkt, const FeatureFrame& frame) {
  stats_ = Stats{};
  if (!initialized_) {
    for (const auto& s : pkt.imu) static_init_.addImu(s);
    if (!pkt.imu.empty()) last_imu_ = pkt.imu.back();
    if (auto init = static_init_.tryInitialize()) initialize(pkt, frame, *init);
    return initialized_;
  }

  integrateImu(pkt);
  const gtsam::NavState pred = pim_->predict(kf_state_, kf_bias_);

  // Keyframe decision from time, parallax and how many last-keyframe features survive.
  double parallax = 0;
  int common = 0;
  for (const auto& f : frame.features) {
    auto it = last_kf_px_.find(f.id);
    if (it == last_kf_px_.end()) continue;
    parallax += (f.px_undist - it->second).norm();
    ++common;
  }
  parallax = common > 0 ? parallax / common : 0;
  const double dt_kf = (pkt.t_ns - last_kf_t_ns_) * 1e-9;
  const auto& kc = cfg_.kf;
  const bool is_kf =
      dt_kf >= kc.max_dt ||
      (dt_kf >= kc.min_dt && (parallax >= kc.min_parallax_px ||
                              common < kc.min_track_ratio * static_cast<double>(last_kf_px_.size())));

  if (!is_kf) {
    out_ = {pkt.t_ns, pred, kf_bias_, false};
    return true;
  }

  // Zero-velocity detection: the image barely moved and the IMU sees no acceleration change.
  double acc_std = 0;
  if (!acc_norms_since_kf_.empty()) {
    double m = 0;
    for (double a : acc_norms_since_kf_) m += a;
    m /= acc_norms_since_kf_.size();
    for (double a : acc_norms_since_kf_) acc_std += (a - m) * (a - m);
    acc_std = std::sqrt(acc_std / acc_norms_since_kf_.size());
  }
  const bool zupt = common > 20 && parallax < cfg_.backend.zupt_max_parallax_px &&
                    acc_std < cfg_.backend.zupt_max_acc_std;

  addKeyframe(pkt, frame, pred, zupt);
  return true;
}

void VioEstimator::initialize(const SyncPacket& pkt, const FeatureFrame& frame,
                              const ImuInitResult& init) {
  t0_ns_ = pkt.t_ns;
  kf_ = 0;
  kf_times_[0] = pkt.t_ns;
  last_kf_t_ns_ = last_frame_t_ns_ = pkt.t_ns;
  kf_state_ = gtsam::NavState(gtsam::Rot3(init.R_WB), gtsam::Point3::Zero(), gtsam::Vector3::Zero());
  kf_bias_ = gtsam::imuBias::ConstantBias(init.ba, init.bg);
  pim_->resetIntegrationAndSetBias(kf_bias_);

  const auto& bc = cfg_.backend;
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;
  gtsam::Vector6 pose_sigmas;
  pose_sigmas << gtsam::Vector3::Constant(bc.prior_rot_sigma), gtsam::Vector3::Constant(bc.prior_pos_sigma);
  gtsam::Vector6 bias_sigmas;
  bias_sigmas << gtsam::Vector3::Constant(bc.prior_ba_sigma), gtsam::Vector3::Constant(bc.prior_bg_sigma);
  graph.addPrior(X(0), kf_state_.pose(), nm::Diagonal::Sigmas(pose_sigmas));
  graph.addPrior(V(0), kf_state_.v(), nm::Isotropic::Sigma(3, bc.prior_vel_sigma));
  graph.addPrior(B(0), kf_bias_, nm::Diagonal::Sigmas(bias_sigmas));
  values.insert(X(0), kf_state_.pose());
  values.insert(V(0), kf_state_.v());
  values.insert(B(0), kf_bias_);
  gtsam::FixedLagSmoother::KeyTimestampMap stamps{{X(0), 0.0}, {V(0), 0.0}, {B(0), 0.0}};
  smoother_->update(graph, values, stamps);
  estimate_ = smoother_->calculateEstimate();

  last_kf_px_.clear();
  for (const auto& f : frame.features) {
    last_kf_px_[f.id] = f.px_undist;
    tracks_[f.id].obs.push_back({0, f.px_undist});
  }
  initialized_ = true;
  out_ = {pkt.t_ns, kf_state_, kf_bias_, true};
  std::cout << "[vio] initialized at t=" << pkt.t_ns << "  bg=" << init.bg.transpose()
            << "  ba=" << init.ba.transpose() << std::endl;
}

void VioEstimator::integrateImu(const SyncPacket& pkt) {
  std::vector<ImuSample> samples;
  samples.reserve(pkt.imu.size() + 1);
  samples.push_back(last_imu_);  // covers the start of the interval
  samples.insert(samples.end(), pkt.imu.begin(), pkt.imu.end());
  for (const auto& seg : imuSegments(samples, last_frame_t_ns_, pkt.t_ns)) {
    pim_->integrateMeasurement(seg.acc, seg.gyro, seg.dt);
    acc_norms_since_kf_.push_back(seg.acc.norm());
  }
  if (!pkt.imu.empty()) last_imu_ = pkt.imu.back();
  last_frame_t_ns_ = pkt.t_ns;
}

bool VioEstimator::kfAlive(int kf) const {
  // Only reference keyframes that won't be marginalized by this update (small safety margin).
  auto it = kf_times_.find(kf);
  if (it == kf_times_.end() || !estimate_.exists(X(kf))) return false;
  return toSec(it->second) > toSec(kf_times_.at(kf_)) - cfg_.backend.lag_s + 0.1;
}

gtsam::Pose3 VioEstimator::kfPose(int kf) const { return estimate_.at<gtsam::Pose3>(X(kf)); }

double VioEstimator::reprojError(const gtsam::Pose3& T_WB, const gtsam::Point3& p_W,
                                 const Eigen::Vector2d& px) const {
  const gtsam::Point3 p_C = (T_WB * T_BC_).transformTo(p_W);
  if (p_C.z() < cfg_.backend.min_depth) return -1;
  const Eigen::Vector2d uv(K_->fx() * p_C.x() / p_C.z() + K_->px(),
                           K_->fy() * p_C.y() / p_C.z() + K_->py());
  return (uv - px).norm();
}

void VioEstimator::addKeyframe(const SyncPacket& pkt, const FeatureFrame& frame,
                               const gtsam::NavState& pred, bool zupt) {
  const auto& bc = cfg_.backend;
  const int k = ++kf_;
  kf_times_[k] = pkt.t_ns;
  const double ts = toSec(pkt.t_ns);

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;
  gtsam::FixedLagSmoother::KeyTimestampMap stamps;
  values.insert(X(k), pred.pose());
  values.insert(V(k), pred.v());
  values.insert(B(k), kf_bias_);
  stamps[X(k)] = stamps[V(k)] = stamps[B(k)] = ts;
  graph.emplace_shared<gtsam::CombinedImuFactor>(X(k - 1), V(k - 1), X(k), V(k), B(k - 1), B(k), *pim_);

  if (zupt) {
    graph.addPrior(V(k), gtsam::Vector3(gtsam::Vector3::Zero()), nm::Isotropic::Sigma(3, bc.zupt_vel_sigma));
    gtsam::Vector6 s;
    s << gtsam::Vector3::Constant(1e-3), gtsam::Vector3::Constant(1e-3);
    graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(X(k - 1), X(k), gtsam::Pose3(),
                                                             nm::Diagonal::Sigmas(s));
    stats_.zupt = true;
  }

  // While static there is no parallax: skip visual updates so landmarks aren't triangulated from
  // (and observations aren't accumulated over) a single viewpoint.
  if (!zupt) {
    for (const auto& f : frame.features) {
      Track& tr = tracks_[f.id];
      if (tr.rejected) continue;
      tr.obs.push_back({k, f.px_undist});

      if (tr.in_graph) {
        if (!estimate_.exists(L(f.id))) {  // landmark was marginalized: start over
          tr.in_graph = false;
          tr.obs = {tr.obs.back()};
          continue;
        }
        const auto p_W = estimate_.at<gtsam::Point3>(L(f.id));
        const double e = reprojError(pred.pose(), p_W, f.px_undist);
        if (e < 0 || e > 2 * bc.max_reproj_px) {  // inconsistent with prediction: stop using it
          tr.rejected = true;
          ++stats_.gated_observations;
          continue;
        }
        graph.emplace_shared<ProjectionFactor>(gtsam::Point2(f.px_undist), pixel_noise_, X(k),
                                               L(f.id), K_, T_BC_);
        stamps[L(f.id)] = ts;
        ++stats_.visual_factors;
        continue;
      }

      // Not yet a landmark: keep observations from keyframes still in the window.
      std::vector<Observation> alive;
      for (const auto& o : tr.obs)
        if (o.kf == k || kfAlive(o.kf)) alive.push_back(o);
      tr.obs = alive;
      if (static_cast<int>(tr.obs.size()) < bc.min_track_obs) continue;

      std::vector<gtsam::Pose3> T_WB;
      std::vector<Eigen::Isometry3d> T_CW;
      std::vector<Eigen::Vector2d> xn;
      for (const auto& o : tr.obs) {
        T_WB.push_back(o.kf == k ? pred.pose() : kfPose(o.kf));
        T_CW.emplace_back((T_WB.back() * T_BC_).inverse().matrix());
        xn.emplace_back((o.px.x() - K_->px()) / K_->fx(), (o.px.y() - K_->py()) / K_->fy());
      }
      const auto p = triangulateDLT(T_CW, xn);
      if (!p) continue;
      if (maxParallaxAngle(T_CW, *p) < bc.min_parallax_deg * M_PI / 180.0) continue;  // wait

      bool ok = true;
      for (size_t i = 0; i < tr.obs.size() && ok; ++i) {
        const double e = reprojError(T_WB[i], *p, tr.obs[i].px);
        const double depth = (T_CW[i] * *p).z();
        ok = e >= 0 && e < bc.max_reproj_px && depth < bc.max_depth;
      }
      if (!ok) {
        // Enough parallax but geometrically inconsistent: almost surely a bad track.
        tr.rejected = true;
        continue;
      }

      values.insert(L(f.id), gtsam::Point3(*p));
      stamps[L(f.id)] = ts;
      for (const auto& o : tr.obs) {
        graph.emplace_shared<ProjectionFactor>(gtsam::Point2(o.px), pixel_noise_, X(o.kf), L(f.id),
                                               K_, T_BC_);
        ++stats_.visual_factors;
      }
      tr.in_graph = true;
      ++stats_.new_landmarks;
    }
  }

  const auto t_start = std::chrono::steady_clock::now();
  smoother_->update(graph, values, stamps);
  estimate_ = smoother_->calculateEstimate();
  stats_.solve_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();

  kf_state_ = gtsam::NavState(estimate_.at<gtsam::Pose3>(X(k)), estimate_.at<gtsam::Vector3>(V(k)));
  kf_bias_ = estimate_.at<gtsam::imuBias::ConstantBias>(B(k));
  pim_->resetIntegrationAndSetBias(kf_bias_);
  acc_norms_since_kf_.clear();
  last_kf_t_ns_ = pkt.t_ns;

  // Bookkeeping: forget tracks the front-end lost and keyframes that left the window.
  std::unordered_set<uint64_t> seen;
  last_kf_px_.clear();
  for (const auto& f : frame.features) {
    seen.insert(f.id);
    last_kf_px_[f.id] = f.px_undist;
  }
  for (auto it = tracks_.begin(); it != tracks_.end();)
    it = seen.count(it->first) ? std::next(it) : tracks_.erase(it);
  for (auto it = kf_times_.begin(); it != kf_times_.end();)
    it = estimate_.exists(X(it->first)) ? std::next(it) : kf_times_.erase(it);

  for (const auto& kv : estimate_)
    if (gtsam::Symbol(kv.key).chr() == 'l') ++stats_.landmarks_in_window;
  stats_.keyframes = static_cast<int>(kf_times_.size());
  out_ = {pkt.t_ns, kf_state_, kf_bias_, true};
}

std::vector<Eigen::Vector3d> VioEstimator::landmarks() const {
  std::vector<Eigen::Vector3d> out;
  for (const auto& kv : estimate_)
    if (gtsam::Symbol(kv.key).chr() == 'l') out.push_back(kv.value.cast<gtsam::Point3>());
  return out;
}

}  // namespace deep_vio
