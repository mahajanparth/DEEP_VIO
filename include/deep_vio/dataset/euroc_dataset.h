#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace deep_vio {

struct ImuSample {
  int64_t t_ns = 0;
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();  // [rad/s], IMU frame
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();   // [m/s^2], specific force, IMU frame
};

struct CameraCalib {
  double fx = 0, fy = 0, cx = 0, cy = 0;
  Eigen::Vector4d dist = Eigen::Vector4d::Zero();  // radtan: k1 k2 p1 p2
  int width = 0, height = 0;
  // Camera pose in the body (= IMU) frame: p_B = T_BC * p_C.
  Eigen::Matrix4d T_BC = Eigen::Matrix4d::Identity();
};

struct ImuCalib {
  double gyro_noise_density = 0;   // [rad/s/sqrt(Hz)]
  double gyro_random_walk = 0;     // [rad/s^2/sqrt(Hz)]
  double accel_noise_density = 0;  // [m/s^2/sqrt(Hz)]
  double accel_random_walk = 0;    // [m/s^3/sqrt(Hz)]
  double rate_hz = 0;
};

// Body (IMU) state in the Vicon/Leica world frame R.
struct GroundTruthState {
  int64_t t_ns = 0;
  Eigen::Vector3d p_WB;
  Eigen::Quaterniond q_WB;
  Eigen::Vector3d v_W, bg, ba;
};

// One camera frame plus the IMU samples with t in (t_prev_frame, t_frame].
struct SyncPacket {
  int64_t t_ns = 0;
  cv::Mat img;  // 8-bit grayscale, raw (distorted)
  std::vector<ImuSample> imu;
};

namespace euroc_csv {
// Each returns false for comment/header/malformed lines.
bool parseImuLine(const std::string& line, ImuSample& out);
bool parseImageLine(const std::string& line, int64_t& t_ns, std::string& filename);
bool parseGroundTruthLine(const std::string& line, GroundTruthState& out);
}  // namespace euroc_csv

class EurocDataset {
 public:
  // `mav0_dir` is the directory containing cam0/, imu0/, state_groundtruth_estimate0/.
  explicit EurocDataset(const std::string& mav0_dir);

  bool next(SyncPacket& out);  // false at end of sequence
  void reset() { next_image_ = 0; next_imu_ = 0; }

  size_t numImages() const { return image_times_.size(); }
  const CameraCalib& camCalib() const { return cam_; }
  const ImuCalib& imuCalib() const { return imu_calib_; }
  const std::vector<ImuSample>& imu() const { return imu_; }
  const std::vector<GroundTruthState>& groundTruth() const { return gt_; }

 private:
  std::string dir_;
  CameraCalib cam_;
  ImuCalib imu_calib_;
  std::vector<int64_t> image_times_;
  std::vector<std::string> image_files_;
  std::vector<ImuSample> imu_;
  std::vector<GroundTruthState> gt_;
  size_t next_image_ = 0;
  size_t next_imu_ = 0;
};

}  // namespace deep_vio
