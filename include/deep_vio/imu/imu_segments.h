#pragma once

#include <vector>

#include "deep_vio/dataset/euroc_dataset.h"

namespace deep_vio {

// A constant-measurement piece of IMU data, ready to feed an integrator.
struct ImuSegment {
  Eigen::Vector3d acc, gyro;
  double dt;  // [s]
};

// Splits the time interval [t_begin, t_end] into segments between consecutive samples.
// Each segment uses the midpoint of its two bounding samples. Outside the sample range the
// nearest sample is held constant. `samples` must be sorted by time; to integrate exactly from
// t_begin it should include the last sample at or before t_begin.
std::vector<ImuSegment> imuSegments(const std::vector<ImuSample>& samples, int64_t t_begin,
                                    int64_t t_end);

}  // namespace deep_vio
