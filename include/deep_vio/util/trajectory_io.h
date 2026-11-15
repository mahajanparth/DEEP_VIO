#pragma once

#include <Eigen/Geometry>
#include <fstream>
#include <string>

namespace deep_vio {

// Writes poses in TUM format: "t[s] x y z qx qy qz qw" (readable by evo).
class TumWriter {
 public:
  explicit TumWriter(const std::string& path);
  void write(int64_t t_ns, const Eigen::Matrix3d& R_WB, const Eigen::Vector3d& p_WB);

 private:
  std::ofstream f_;
};

}  // namespace deep_vio
