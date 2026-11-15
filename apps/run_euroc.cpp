// Runs the full VIO pipeline on a EuRoC sequence and writes a TUM trajectory of the body (IMU).
// Usage: run_euroc <mav0_dir> <config.yaml> <out_traj.txt> [--viz]
#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/highgui.hpp>

#include "deep_vio/backend/vio_estimator.h"
#include "deep_vio/dataset/euroc_dataset.h"
#include "deep_vio/frontend/klt_tracker.h"
#include "deep_vio/util/config.h"
#include "deep_vio/util/trajectory_io.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <mav0_dir> <config.yaml> <out_traj.txt> [--viz]\n";
    return 1;
  }
  bool viz = false;
  for (int i = 4; i < argc; ++i) viz |= std::string(argv[i]) == "--viz";

  using namespace deep_vio;
  EurocDataset ds(argv[1]);
  const VioConfig cfg = loadConfig(argv[2]);
  TumWriter writer(argv[3]);
  const PinholeRadtan cam(ds.camCalib());
  KltTracker tracker(cfg.klt, cam);
  VioEstimator vio(cfg, ds.camCalib(), ds.imuCalib());

  SyncPacket pkt;
  FeatureFrame prev_frame;
  int frame_idx = 0;
  double front_ms = 0, back_ms = 0, max_back_ms = 0;
  int keyframes = 0;
  while (ds.next(pkt)) {
    const auto t0 = std::chrono::steady_clock::now();
    const FeatureFrame frame = tracker.track(pkt.t_ns, pkt.img);
    const auto t1 = std::chrono::steady_clock::now();
    const bool ok = vio.process(pkt, frame);
    const auto t2 = std::chrono::steady_clock::now();
    front_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double b = std::chrono::duration<double, std::milli>(t2 - t1).count();
    back_ms += b;
    max_back_ms = std::max(max_back_ms, b);

    if (ok) {
      const auto& o = vio.output();
      writer.write(o.t_ns, o.state.pose().rotation().matrix(), o.state.pose().translation());
      if (o.is_keyframe) {
        ++keyframes;
        const auto& s = vio.stats();
        if (keyframes % 20 == 0)
          std::cout << std::fixed << std::setprecision(3) << "[kf " << keyframes << "] frame "
                    << frame_idx << "  p=" << o.state.pose().translation().transpose()
                    << "  |v|=" << o.state.v().norm() << "  lm=" << s.landmarks_in_window
                    << " new=" << s.new_landmarks << " vis=" << s.visual_factors
                    << " gated=" << s.gated_observations << (s.zupt ? " ZUPT" : "")
                    << "  solve=" << s.solve_ms << "ms\n";
      }
    }
    if (viz) {
      cv::imshow("tracks", drawFeatureTracks(pkt.img, prev_frame, frame));
      if (cv::waitKey(1) == 27) break;
    }
    prev_frame = frame;
    ++frame_idx;
  }
  std::cout << "frames=" << frame_idx << " keyframes=" << keyframes
            << "  avg frontend=" << front_ms / frame_idx << "ms  avg backend=" << back_ms / frame_idx
            << "ms  max backend=" << max_back_ms << "ms\n";
  return 0;
}
