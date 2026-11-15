#include "deep_vio/frontend/klt_tracker.h"

#include <numeric>
#include <unordered_map>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace deep_vio {

namespace {

template <typename T>
void keepIf(std::vector<T>& v, const std::vector<uchar>& keep) {
  size_t j = 0;
  for (size_t i = 0; i < v.size(); ++i)
    if (keep[i]) v[j++] = v[i];
  v.resize(j);
}

Eigen::Vector2d toEigen(const cv::Point2f& p) { return {p.x, p.y}; }

}  // namespace

FeatureFrame KltTracker::track(int64_t t_ns, const cv::Mat& gray_in) {
  cv::Mat img;
  if (p_.use_clahe) {
    // Local contrast equalization makes LK more robust to the exposure changes in EuRoC.
    static thread_local auto clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(gray_in, img);
  } else {
    img = gray_in;
  }

  // 1) Track existing features with LK, then verify by tracking back (forward-backward check).
  if (!prev_pts_.empty()) {
    std::vector<cv::Point2f> cur, back;
    std::vector<uchar> st_f, st_b;
    std::vector<float> err;
    const cv::Size win(p_.win_size, p_.win_size);
    cv::calcOpticalFlowPyrLK(prev_img_, img, prev_pts_, cur, st_f, err, win, p_.pyramid_levels);
    cv::calcOpticalFlowPyrLK(img, prev_img_, cur, back, st_b, err, win, p_.pyramid_levels);
    std::vector<uchar> keep(cur.size());
    for (size_t i = 0; i < cur.size(); ++i) {
      const bool in = cur[i].x >= p_.border && cur[i].y >= p_.border &&
                      cur[i].x < img.cols - p_.border && cur[i].y < img.rows - p_.border;
      keep[i] = st_f[i] && st_b[i] && in && cv::norm(back[i] - prev_pts_[i]) < p_.fb_threshold;
    }
    keepIf(prev_pts_, keep);
    keepIf(cur, keep);
    keepIf(ids_, keep);
    keepIf(ages_, keep);

    // 2) Reject outliers with epipolar geometry on undistorted points (removes features on
    //    moving objects and bad LK matches).
    if (cur.size() >= 8) {
      std::vector<cv::Point2f> un_prev, un_cur;
      for (size_t i = 0; i < cur.size(); ++i) {
        const auto a = cam_.undistortPixel(toEigen(prev_pts_[i]));
        const auto b = cam_.undistortPixel(toEigen(cur[i]));
        un_prev.emplace_back(a.x(), a.y());
        un_cur.emplace_back(b.x(), b.y());
      }
      std::vector<uchar> inl;
      cv::findFundamentalMat(un_prev, un_cur, cv::FM_RANSAC, p_.ransac_threshold, 0.99, inl);
      if (inl.size() == cur.size()) {
        keepIf(cur, inl);
        keepIf(ids_, inl);
        keepIf(ages_, inl);
      }
    }
    prev_pts_ = cur;
    for (auto& a : ages_) ++a;
  }

  // 3) Enforce a minimum distance, giving priority to the longest tracks.
  std::vector<size_t> order(prev_pts_.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return ages_[a] > ages_[b]; });
  cv::Mat mask(img.size(), CV_8UC1, cv::Scalar(255));
  std::vector<uchar> keep(prev_pts_.size(), 0);
  for (size_t i : order) {
    const cv::Point pt(cvRound(prev_pts_[i].x), cvRound(prev_pts_[i].y));
    if (mask.at<uchar>(pt) == 0) continue;
    keep[i] = 1;
    cv::circle(mask, pt, static_cast<int>(p_.min_distance), cv::Scalar(0), -1);
  }
  keepIf(prev_pts_, keep);
  keepIf(ids_, keep);
  keepIf(ages_, keep);

  // 4) Detect new corners in the empty regions.
  const int need = p_.max_features - static_cast<int>(prev_pts_.size());
  if (need > 0) {
    std::vector<cv::Point2f> fresh;
    cv::goodFeaturesToTrack(img, fresh, need, p_.quality_level, p_.min_distance, mask);
    for (const auto& f : fresh) {
      prev_pts_.push_back(f);
      ids_.push_back(next_id_++);
      ages_.push_back(1);
    }
  }
  prev_img_ = img;

  FeatureFrame out;
  out.t_ns = t_ns;
  out.features.reserve(prev_pts_.size());
  for (size_t i = 0; i < prev_pts_.size(); ++i) {
    const Eigen::Vector2d px = toEigen(prev_pts_[i]);
    out.features.push_back({ids_[i], px, cam_.undistortPixel(px), ages_[i]});
  }
  return out;
}

cv::Mat drawFeatureTracks(const cv::Mat& gray, const FeatureFrame& prev, const FeatureFrame& cur) {
  cv::Mat vis;
  cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
  std::unordered_map<uint64_t, Eigen::Vector2d> prev_px;
  for (const auto& f : prev.features) prev_px[f.id] = f.px;
  for (const auto& f : cur.features) {
    // Blue (new) -> red (tracked for 20+ frames).
    const double s = std::min(1.0, f.age / 20.0);
    const cv::Scalar color(255 * (1 - s), 0, 255 * s);
    const cv::Point2f p(f.px.x(), f.px.y());
    cv::circle(vis, p, 2, color, 2);
    auto it = prev_px.find(f.id);
    if (it != prev_px.end()) cv::line(vis, p, cv::Point2f(it->second.x(), it->second.y()), {0, 255, 0}, 1);
  }
  return vis;
}

}  // namespace deep_vio
