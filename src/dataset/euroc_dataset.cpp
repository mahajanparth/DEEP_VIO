#include "deep_vio/dataset/euroc_dataset.h"

#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <stdexcept>

namespace deep_vio {
namespace euroc_csv {
namespace {

// Splits a comma-separated line; returns false for empty or '#' lines.
bool splitCsv(const std::string& line, std::vector<std::string>& fields) {
  fields.clear();
  if (line.empty() || line[0] == '#') return false;
  std::stringstream ss(line);
  std::string f;
  while (std::getline(ss, f, ',')) {
    // Trim whitespace and a trailing '\r' from Windows line endings.
    const auto b = f.find_first_not_of(" \t\r");
    const auto e = f.find_last_not_of(" \t\r");
    fields.push_back(b == std::string::npos ? "" : f.substr(b, e - b + 1));
  }
  return !fields.empty();
}

}  // namespace

bool parseImuLine(const std::string& line, ImuSample& out) {
  std::vector<std::string> f;
  if (!splitCsv(line, f) || f.size() < 7) return false;
  try {
    out.t_ns = std::stoll(f[0]);
    // Column order in EuRoC: t, w_x, w_y, w_z, a_x, a_y, a_z (gyro first).
    out.gyro = {std::stod(f[1]), std::stod(f[2]), std::stod(f[3])};
    out.acc = {std::stod(f[4]), std::stod(f[5]), std::stod(f[6])};
  } catch (const std::exception&) {
    return false;
  }
  return true;
}

bool parseImageLine(const std::string& line, int64_t& t_ns, std::string& filename) {
  std::vector<std::string> f;
  if (!splitCsv(line, f) || f.size() < 2) return false;
  try {
    t_ns = std::stoll(f[0]);
  } catch (const std::exception&) {
    return false;
  }
  filename = f[1];
  return true;
}

bool parseGroundTruthLine(const std::string& line, GroundTruthState& out) {
  std::vector<std::string> f;
  if (!splitCsv(line, f) || f.size() < 17) return false;
  try {
    double v[17];
    out.t_ns = std::stoll(f[0]);
    for (int i = 1; i < 17; ++i) v[i] = std::stod(f[i]);
    out.p_WB = {v[1], v[2], v[3]};
    out.q_WB = Eigen::Quaterniond(v[4], v[5], v[6], v[7]).normalized();  // w x y z
    out.v_W = {v[8], v[9], v[10]};
    out.bg = {v[11], v[12], v[13]};
    out.ba = {v[14], v[15], v[16]};
  } catch (const std::exception&) {
    return false;
  }
  return true;
}

}  // namespace euroc_csv

namespace {

std::vector<double> readSeq(const cv::FileNode& n) {
  std::vector<double> v;
  for (const auto& e : n) v.push_back(static_cast<double>(e));
  return v;
}

// cv::FileStorage requires a "%YAML:1.0" directive, which EuRoC's sensor.yaml files lack.
cv::FileStorage openYaml(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("Cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  std::string text = ss.str();
  if (text.rfind("%YAML", 0) != 0) text = "%YAML:1.0\n" + text;
  cv::FileStorage fs(text, cv::FileStorage::READ | cv::FileStorage::MEMORY);
  if (!fs.isOpened()) throw std::runtime_error("Cannot parse " + path);
  return fs;
}

}  // namespace

EurocDataset::EurocDataset(const std::string& mav0_dir) : dir_(mav0_dir) {
  // --- camera calibration ---
  {
    auto fs = openYaml(dir_ + "/cam0/sensor.yaml");
    const auto K = readSeq(fs["intrinsics"]);
    const auto d = readSeq(fs["distortion_coefficients"]);
    const auto res = readSeq(fs["resolution"]);
    const auto T = readSeq(fs["T_BS"]["data"]);
    if (K.size() != 4 || d.size() != 4 || res.size() != 2 || T.size() != 16)
      throw std::runtime_error("Unexpected cam0/sensor.yaml format");
    cam_.fx = K[0], cam_.fy = K[1], cam_.cx = K[2], cam_.cy = K[3];
    cam_.dist = {d[0], d[1], d[2], d[3]};
    cam_.width = static_cast<int>(res[0]);
    cam_.height = static_cast<int>(res[1]);
    for (int i = 0; i < 16; ++i) cam_.T_BC(i / 4, i % 4) = T[i];  // row-major
  }
  // --- IMU calibration ---
  {
    auto fs = openYaml(dir_ + "/imu0/sensor.yaml");
    imu_calib_.gyro_noise_density = fs["gyroscope_noise_density"];
    imu_calib_.gyro_random_walk = fs["gyroscope_random_walk"];
    imu_calib_.accel_noise_density = fs["accelerometer_noise_density"];
    imu_calib_.accel_random_walk = fs["accelerometer_random_walk"];
    imu_calib_.rate_hz = fs["rate_hz"];
  }
  // --- image list ---
  {
    std::ifstream f(dir_ + "/cam0/data.csv");
    if (!f) throw std::runtime_error("Cannot open " + dir_ + "/cam0/data.csv");
    std::string line, name;
    int64_t t;
    while (std::getline(f, line))
      if (euroc_csv::parseImageLine(line, t, name)) {
        image_times_.push_back(t);
        image_files_.push_back(dir_ + "/cam0/data/" + name);
      }
  }
  // --- IMU ---
  {
    std::ifstream f(dir_ + "/imu0/data.csv");
    if (!f) throw std::runtime_error("Cannot open " + dir_ + "/imu0/data.csv");
    std::string line;
    ImuSample s;
    while (std::getline(f, line))
      if (euroc_csv::parseImuLine(line, s)) imu_.push_back(s);
  }
  // --- ground truth (optional) ---
  {
    std::ifstream f(dir_ + "/state_groundtruth_estimate0/data.csv");
    std::string line;
    GroundTruthState g;
    while (f && std::getline(f, line))
      if (euroc_csv::parseGroundTruthLine(line, g)) gt_.push_back(g);
  }
  if (image_times_.empty() || imu_.empty()) throw std::runtime_error("Empty EuRoC sequence");
}

bool EurocDataset::next(SyncPacket& out) {
  // Skip frames that start before the IMU stream: they have no inertial data to pair with.
  while (next_image_ < image_times_.size() && image_times_[next_image_] < imu_.front().t_ns)
    ++next_image_;
  if (next_image_ >= image_times_.size()) return false;
  const int64_t t = image_times_[next_image_];
  if (t > imu_.back().t_ns) return false;  // no IMU coverage after this point

  out.t_ns = t;
  out.imu.clear();
  while (next_imu_ < imu_.size() && imu_[next_imu_].t_ns <= t) out.imu.push_back(imu_[next_imu_++]);
  out.img = cv::imread(image_files_[next_image_], cv::IMREAD_GRAYSCALE);
  if (out.img.empty()) throw std::runtime_error("Cannot read " + image_files_[next_image_]);
  ++next_image_;
  return true;
}

}  // namespace deep_vio
