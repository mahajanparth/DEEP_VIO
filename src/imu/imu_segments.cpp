#include "deep_vio/imu/imu_segments.h"

#include <algorithm>

namespace deep_vio {

std::vector<ImuSegment> imuSegments(const std::vector<ImuSample>& s, int64_t t_begin,
                                    int64_t t_end) {
  std::vector<ImuSegment> out;
  if (s.empty() || t_end <= t_begin) return out;
  auto push = [&](int64_t a, int64_t b, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro) {
    if (b > a) out.push_back({acc, gyro, (b - a) * 1e-9});
  };
  // Before the first sample: hold it.
  push(t_begin, std::min(t_end, s.front().t_ns), s.front().acc, s.front().gyro);
  for (size_t k = 0; k + 1 < s.size(); ++k) {
    const int64_t a = std::max(t_begin, s[k].t_ns), b = std::min(t_end, s[k + 1].t_ns);
    push(a, b, 0.5 * (s[k].acc + s[k + 1].acc), 0.5 * (s[k].gyro + s[k + 1].gyro));
  }
  // After the last sample: hold it.
  push(std::max(t_begin, s.back().t_ns), t_end, s.back().acc, s.back().gyro);
  return out;
}

}  // namespace deep_vio
