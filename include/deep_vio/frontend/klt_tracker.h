#pragma once

#include "deep_vio/frontend/feature_tracker.h"

namespace deep_vio {

struct KltParams {
  int max_features = 150;
  double min_distance = 30;       // [px] between features (spreads them over the image)
  double quality_level = 0.01;    // Shi-Tomasi
  int win_size = 21;              // LK window
  int pyramid_levels = 3;
  double fb_threshold = 0.5;      // [px] forward-backward consistency
  double ransac_threshold = 1.0;  // [px] fundamental-matrix RANSAC
  int border = 5;                 // [px]
  bool use_clahe = true;
};

// VINS-Mono-style tracker: CLAHE, pyramidal LK with a forward-backward check,
// fundamental-matrix RANSAC, and min-distance Shi-Tomasi detection to keep the count up.
class KltTracker : public FeatureTracker {
 public:
  KltTracker(const KltParams& p, const PinholeRadtan& cam) : p_(p), cam_(cam) {}
  FeatureFrame track(int64_t t_ns, const cv::Mat& gray) override;

 private:
  KltParams p_;
  PinholeRadtan cam_;
  cv::Mat prev_img_;
  std::vector<cv::Point2f> prev_pts_;
  std::vector<uint64_t> ids_;
  std::vector<int> ages_;
  uint64_t next_id_ = 0;
};

}  // namespace deep_vio
