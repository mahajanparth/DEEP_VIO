// M4 check: runs the KLT front-end on a EuRoC sequence and reports tracking statistics.
// Usage: track_euroc <mav0_dir> [--viz] [--save <dir>] [--frames N]
#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "deep_vio/dataset/euroc_dataset.h"
#include "deep_vio/frontend/klt_tracker.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <mav0_dir> [--viz] [--save <dir>] [--frames N]\n";
    return 1;
  }
  bool viz = false;
  std::string save_dir;
  int max_frames = -1;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--viz") viz = true;
    if (a == "--save" && i + 1 < argc) save_dir = argv[++i];
    if (a == "--frames" && i + 1 < argc) max_frames = std::stoi(argv[++i]);
  }

  using namespace deep_vio;
  EurocDataset ds(argv[1]);
  KltTracker tracker(KltParams{}, PinholeRadtan(ds.camCalib()));

  SyncPacket pkt;
  FeatureFrame prev;
  long n = 0, total_features = 0, total_survived = 0, total_prev = 0, total_age = 0;
  double ms = 0;
  while (ds.next(pkt) && (max_frames < 0 || n < max_frames)) {
    const auto t0 = std::chrono::steady_clock::now();
    const FeatureFrame cur = tracker.track(pkt.t_ns, pkt.img);
    ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    long survived = 0;
    for (const auto& f : cur.features) {
      survived += f.age > 1;
      total_age += f.age;
    }
    total_features += cur.features.size();
    total_survived += survived;
    total_prev += prev.features.size();

    if (viz || (!save_dir.empty() && n % 200 == 0)) {
      const cv::Mat vis = drawFeatureTracks(pkt.img, prev, cur);
      if (!save_dir.empty() && n % 200 == 0)
        cv::imwrite(save_dir + "/tracks_" + std::to_string(n) + ".png", vis);
      if (viz) {
        cv::imshow("tracks", vis);
        if (cv::waitKey(1) == 27) break;
      }
    }
    prev = cur;
    ++n;
  }
  std::cout << std::fixed << std::setprecision(2) << "frames: " << n
            << "\nfeatures/frame: " << double(total_features) / n
            << "\nframe-to-frame survival: " << 100.0 * total_survived / std::max(1L, total_prev) << " %"
            << "\nmean track age: " << double(total_age) / total_features << " frames"
            << "\ntracking time: " << ms / n << " ms/frame\n";
  return 0;
}
