#include "deep_vio/util/trajectory_io.h"

#include <iomanip>
#include <stdexcept>

namespace deep_vio {

TumWriter::TumWriter(const std::string& path) : f_(path) {
  if (!f_) throw std::runtime_error("Cannot write " + path);
  f_ << std::fixed;
}

void TumWriter::write(int64_t t_ns, const Eigen::Matrix3d& R_WB, const Eigen::Vector3d& p) {
  const Eigen::Quaterniond q(R_WB);
  // Split seconds/nanoseconds to print the timestamp without double rounding.
  f_ << t_ns / 1000000000 << '.' << std::setw(9) << std::setfill('0') << t_ns % 1000000000
     << std::setfill(' ') << std::setprecision(9) << ' ' << p.x() << ' ' << p.y() << ' ' << p.z()
     << ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
}

}  // namespace deep_vio
