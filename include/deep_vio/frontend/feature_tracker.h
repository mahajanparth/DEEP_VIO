#pragma once

#include <cstdint>
#include <opencv2/core.hpp>
#include <vector>

#include "deep_vio/geometry/camera_model.h"

namespace deep_vio {

struct Feature {
  uint64_t id;
  Eigen::Vector2d px;         // raw (distorted) pixel
  Eigen::Vector2d px_undist;  // pixel in the equivalent undistorted pinhole camera (same K)
  int age;                    // number of consecutive frames this feature has been tracked
};

struct FeatureFrame {
  int64_t t_ns = 0;
  std::vector<Feature> features;
};

// Front-end interface: turns images into persistent feature tracks with stable IDs.
// The classical KLT tracker and (later) learned trackers implement this.
class FeatureTracker {
 public:
  virtual ~FeatureTracker() = default;
  virtual FeatureFrame track(int64_t t_ns, const cv::Mat& gray) = 0;
};

// Draws features colored by track age, with a line to their position in `prev`.
cv::Mat drawFeatureTracks(const cv::Mat& gray, const FeatureFrame& prev, const FeatureFrame& cur);

}  // namespace deep_vio
