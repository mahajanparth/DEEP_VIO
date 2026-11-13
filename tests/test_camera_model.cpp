#include <gtest/gtest.h>

#include <opencv2/calib3d.hpp>

#include "deep_vio/geometry/camera_model.h"

using namespace deep_vio;

namespace {
CameraCalib eurocCam0() {
  CameraCalib c;
  c.fx = 458.654, c.fy = 457.296, c.cx = 367.215, c.cy = 248.375;
  c.dist = {-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05};
  c.width = 752, c.height = 480;
  return c;
}
}  // namespace

TEST(PinholeRadtan, ProjectMatchesOpenCV) {
  const auto c = eurocCam0();
  const PinholeRadtan cam(c);
  std::vector<cv::Point3d> pts = {{0.1, -0.2, 1.5}, {-0.6, 0.3, 2.0}, {0.0, 0.0, 4.0}, {0.9, 0.5, 1.2}};
  std::vector<cv::Point2d> cv_px;
  const cv::Mat K = (cv::Mat_<double>(3, 3) << c.fx, 0, c.cx, 0, c.fy, c.cy, 0, 0, 1);
  const cv::Mat D = (cv::Mat_<double>(1, 4) << c.dist[0], c.dist[1], c.dist[2], c.dist[3]);
  cv::projectPoints(pts, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), K, D, cv_px);
  for (size_t i = 0; i < pts.size(); ++i) {
    const auto uv = cam.project({pts[i].x, pts[i].y, pts[i].z});
    ASSERT_TRUE(uv);
    EXPECT_NEAR(uv->x(), cv_px[i].x, 1e-9);
    EXPECT_NEAR(uv->y(), cv_px[i].y, 1e-9);
  }
  EXPECT_FALSE(cam.project({0, 0, -1}));
}

TEST(PinholeRadtan, UndistortRoundTrip) {
  const PinholeRadtan cam(eurocCam0());
  for (double u = 20; u < 740; u += 60)
    for (double v = 20; v < 470; v += 50) {
      const Eigen::Vector2d px(u, v);
      const Eigen::Vector2d xn = cam.pixelToNormalized(px);
      const auto back = cam.project({xn.x(), xn.y(), 1.0});
      ASSERT_TRUE(back);
      EXPECT_NEAR((*back - px).norm(), 0, 1e-6) << u << "," << v;
    }
}

TEST(PinholeRadtan, ProjectionJacobianMatchesNumeric) {
  const PinholeRadtan cam(eurocCam0());
  const Eigen::Vector3d p(0.4, -0.3, 1.7);
  Eigen::Matrix<double, 2, 3> J, Jnum;
  cam.project(p, &J);
  const double h = 1e-7;
  for (int i = 0; i < 3; ++i) {
    const Eigen::Vector3d d = Eigen::Vector3d::Unit(i) * h;
    Jnum.col(i) = (*cam.project(p + d) - *cam.project(p - d)) / (2 * h);
  }
  EXPECT_TRUE(J.isApprox(Jnum, 1e-6)) << J << "\n" << Jnum;
}
