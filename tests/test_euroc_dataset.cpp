#include <gtest/gtest.h>

#include <cstdlib>

#include "deep_vio/dataset/euroc_dataset.h"

using namespace deep_vio;

TEST(EurocCsv, ParsesImuLine) {
  ImuSample s;
  ASSERT_TRUE(euroc_csv::parseImuLine(
      "1403636579758555392,-0.0991,0.1473,0.0272,8.1477,-0.3759,-2.4026\r", s));
  EXPECT_EQ(s.t_ns, 1403636579758555392LL);  // exact: int64, not double
  EXPECT_DOUBLE_EQ(s.gyro.x(), -0.0991);     // gyro comes first
  EXPECT_DOUBLE_EQ(s.acc.x(), 8.1477);
  EXPECT_DOUBLE_EQ(s.acc.z(), -2.4026);
}

TEST(EurocCsv, RejectsHeaderAndGarbage) {
  ImuSample s;
  EXPECT_FALSE(euroc_csv::parseImuLine("#timestamp [ns],w_RS_S_x", s));
  EXPECT_FALSE(euroc_csv::parseImuLine("", s));
  EXPECT_FALSE(euroc_csv::parseImuLine("1,2,3", s));
  EXPECT_FALSE(euroc_csv::parseImuLine("abc,1,2,3,4,5,6", s));
}

TEST(EurocCsv, ParsesImageLine) {
  int64_t t;
  std::string name;
  ASSERT_TRUE(euroc_csv::parseImageLine("1403636579763555584,1403636579763555584.png", t, name));
  EXPECT_EQ(t, 1403636579763555584LL);
  EXPECT_EQ(name, "1403636579763555584.png");
}

TEST(EurocCsv, ParsesGroundTruthQuaternionAsWxyz) {
  GroundTruthState g;
  ASSERT_TRUE(euroc_csv::parseGroundTruthLine(
      "1,1,2,3,0.534108,-0.153029,-0.827383,-0.082152,0,0,0,0,0,0,0,0,0", g));
  EXPECT_NEAR(g.q_WB.w(), 0.534108, 1e-5);
  EXPECT_NEAR(g.q_WB.z(), -0.082152, 1e-5);
}

// Runs only when EUROC_DIR points at a mav0 directory.
TEST(EurocDataset, SyncedPacketsHaveTenImuSamples) {
  const char* dir = std::getenv("EUROC_DIR");
  if (!dir) GTEST_SKIP() << "set EUROC_DIR=<.../mav0> to run";
  EurocDataset ds(dir);
  EXPECT_NEAR(ds.camCalib().fx, 458.654, 1e-6);
  EXPECT_NEAR(ds.camCalib().T_BC(0, 1), -0.999880929698, 1e-9);
  EXPECT_NEAR(ds.imuCalib().rate_hz, 200, 1e-9);

  SyncPacket pkt;
  int frames = 0, ten = 0;
  int64_t prev = 0;
  while (ds.next(pkt) && frames < 500) {
    ASSERT_FALSE(pkt.img.empty());
    for (const auto& s : pkt.imu) {
      EXPECT_LE(s.t_ns, pkt.t_ns);
      if (frames > 0) {
        EXPECT_GT(s.t_ns, prev);  // in (t_prev, t]
      }
    }
    if (frames > 0 && pkt.imu.size() >= 9 && pkt.imu.size() <= 11) ++ten;
    prev = pkt.t_ns;
    ++frames;
  }
  EXPECT_GT(ten, 0.95 * (frames - 1));
}
