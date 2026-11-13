// M1 check: plays a EuRoC sequence, prints IMU samples per frame, and shows raw vs undistorted
// images side by side.
// Usage: play_euroc <mav0_dir> [--no-gui]
#include <iostream>
#include <map>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "deep_vio/dataset/euroc_dataset.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <mav0_dir> [--no-gui]\n";
    return 1;
  }
  const bool gui = !(argc > 2 && std::string(argv[2]) == "--no-gui");
  deep_vio::EurocDataset ds(argv[1]);
  const auto& c = ds.camCalib();
  std::cout << "images: " << ds.numImages() << "  imu: " << ds.imu().size()
            << "  gt: " << ds.groundTruth().size() << "\n"
            << "K: " << c.fx << " " << c.fy << " " << c.cx << " " << c.cy
            << "  dist: " << c.dist.transpose() << "\nT_BC:\n" << c.T_BC << "\n";

  const cv::Mat K = (cv::Mat_<double>(3, 3) << c.fx, 0, c.cx, 0, c.fy, c.cy, 0, 0, 1);
  const cv::Mat D = (cv::Mat_<double>(1, 4) << c.dist[0], c.dist[1], c.dist[2], c.dist[3]);
  cv::Mat map1, map2;
  cv::initUndistortRectifyMap(K, D, cv::Mat(), K, {c.width, c.height}, CV_16SC2, map1, map2);

  std::map<size_t, int> histogram;  // #IMU samples per frame -> count
  deep_vio::SyncPacket pkt;
  int64_t t_prev = -1;
  int n = 0;
  while (ds.next(pkt)) {
    ++histogram[pkt.imu.size()];
    if (n++ % 200 == 0)
      std::cout << "t=" << pkt.t_ns << "  dt=" << (t_prev < 0 ? 0 : (pkt.t_ns - t_prev) * 1e-6)
                << " ms  imu=" << pkt.imu.size() << "\n";
    t_prev = pkt.t_ns;
    if (!gui) continue;
    cv::Mat undist, both;
    cv::remap(pkt.img, undist, map1, map2, cv::INTER_LINEAR);
    cv::hconcat(pkt.img, undist, both);
    cv::putText(both, "raw", {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.8, 255, 2);
    cv::putText(both, "undistorted", {c.width + 10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.8, 255, 2);
    cv::imshow("play_euroc", both);
    if (cv::waitKey(1) == 27) break;
  }
  std::cout << "IMU samples per frame (count -> frames):\n";
  for (const auto& [k, v] : histogram) std::cout << "  " << k << " -> " << v << "\n";
  return 0;
}
