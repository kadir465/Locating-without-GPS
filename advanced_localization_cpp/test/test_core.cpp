#include "advanced_localization_cpp/eskf.hpp"
#include "advanced_localization_cpp/failsafe.hpp"
#include "advanced_localization_cpp/map_store.hpp"
#include "advanced_localization_cpp/math_utils.hpp"
#include "advanced_localization_cpp/ring_buffer.hpp"
#include "advanced_localization_cpp/video_testing.hpp"
#include "advanced_localization_cpp/visual_localizer.hpp"

#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <filesystem>

using namespace advanced_localization;

TEST(RingBuffer, TimeQueriesHandleSortedAndInsertedItems)
{
  TimeIndexedRingBuffer<std::string> buffer(4);
  buffer.append(0.0, "a");
  buffer.append(0.3, "d");
  buffer.append(0.1, "b");
  buffer.append(0.2, "c");

  std::string value;
  double ts = 0.0;
  ASSERT_TRUE(buffer.at_or_before(0.19, ts, value));
  EXPECT_DOUBLE_EQ(ts, 0.1);
  EXPECT_EQ(value, "b");
  ASSERT_TRUE(buffer.at_nearest(0.24, ts, value));
  EXPECT_DOUBLE_EQ(ts, 0.2);
  EXPECT_EQ(value, "c");
}

TEST(MathUtils, QuaternionYawRoundTrip)
{
  const double yaw = 0.73;
  EXPECT_NEAR(yaw_from_quaternion(quaternion_from_yaw(yaw)), yaw, 1e-12);
}

TEST(ESKF, RejectsLargeVisionOutlier)
{
  SystemConfig config;
  config.gate.vmax_mps = 1e6;
  config.gate.kinematic_tolerance_m = 1e6;
  CascadedESKF filter(config);
  filter.initialize(Vector3::Zero(), 0.0, 0.0);

  VisionMeasurement measurement;
  measurement.timestamp_s = 0.0;
  measurement.position_xy_enu_m = Vector2(100.0, 100.0);
  measurement.yaw_rad = 0.0;
  measurement.covariance = Matrix3::Identity() * 0.25;
  measurement.source = "LightGlue";
  measurement.vision_dof = 3;

  auto result = filter.update_from_vision(measurement);
  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(result.reason, "mahalanobis_gate_rejected");
}

TEST(Failsafe, EntersLoiterOnLightglueRejectionLimit)
{
  FailsafeManager manager(2, 100.0);
  FilterHealth health;
  health.consecutive_lightglue_rejections = 2;
  health.position_trace_xy_m2 = 1.0;
  auto decision = manager.evaluate(health, std::nullopt);
  EXPECT_TRUE(decision.changed);
  EXPECT_EQ(decision.mode, "LOITER");
  EXPECT_EQ(decision.code, FailsafeCode::Inconsistency);
}

TEST(VideoTesting, ParsesDjiSrtAndConvertsHeading)
{
  const auto path = std::filesystem::temp_directory_path() / "advanced_localization_cpp_video_test.srt";
  {
    std::ofstream file(path);
    file << "1\n";
    file << "00:00:01,000 --> 00:00:01,033\n";
    file << "latitude: 39.000000 longitude: 32.000000 rel_alt: 101.5 heading: 90.0\n\n";
    file << "2\n";
    file << "00:00:02,000 --> 00:00:02,033\n";
    file << "latitude=39.000010 longitude=32.000020 compass_heading=0.0\n";
  }

  const auto samples = parse_dji_srt(path.string());
  ASSERT_EQ(samples.size(), 2U);
  EXPECT_DOUBLE_EQ(samples[0].timestamp_s, 1.0);
  ASSERT_TRUE(samples[0].heading_deg.has_value());
  EXPECT_DOUBLE_EQ(*samples[0].heading_deg, 90.0);
  ASSERT_TRUE(samples[0].rel_alt_m.has_value());
  EXPECT_DOUBLE_EQ(*samples[0].rel_alt_m, 101.5);
  EXPECT_NEAR(north_cw_heading_deg_to_math_rad(90.0), 0.0, 1e-12);
  EXPECT_NEAR(north_cw_heading_deg_to_math_rad(0.0), 1.5707963267948966, 1e-12);
}

TEST(VideoTesting, FlowConversionMatchesPythonConvention)
{
  FlowSignConfig sign;
  const auto [enu, body] = flow_to_enu(2.0, 4.0, 0.0, 0.5, sign);
  EXPECT_NEAR(body.x(), 2.0, 1e-12);
  EXPECT_NEAR(body.y(), -1.0, 1e-12);
  EXPECT_NEAR(enu.x(), 2.0, 1e-12);
  EXPECT_NEAR(enu.y(), -1.0, 1e-12);

  const Vector2 projected = project_xy_forward_onto_direction(Vector2(-1.0, 2.0), Vector2(1.0, 0.0));
  EXPECT_NEAR(projected.x(), 0.0, 1e-12);
  EXPECT_NEAR(projected.y(), 0.0, 1e-12);
}

TEST(VideoTesting, TrackerScaleUpdateIsBounded)
{
  VisionOnlyTracker tracker(500.0);
  EXPECT_NEAR(tracker.meters_per_pixel(), 0.145, 1e-12);
  tracker.update_scale_from_measurement(1.5, 1.0, 0.4, 1.2, 3);
  EXPECT_LE(tracker.learned_scale, 1.2);
  tracker.update_scale_from_measurement(0.1, 1.0, 0.4, 1.2, 3);
  EXPECT_GE(tracker.learned_scale, 0.4);
}

TEST(VideoTesting, CenterCropOutputOverloadMatchesReturnValue)
{
  cv::Mat image(90, 140, CV_8UC3);
  for (int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x) {
      image.at<cv::Vec3b>(y, x) = cv::Vec3b(
        static_cast<unsigned char>(x % 251),
        static_cast<unsigned char>(y % 251),
        static_cast<unsigned char>((x + y) % 251));
    }
  }

  const cv::Mat returned = center_crop_square(image, 64);
  cv::Mat output;
  center_crop_square(image, 64, output);
  ASSERT_EQ(returned.size(), output.size());
  ASSERT_EQ(returned.type(), output.type());
  cv::Mat diff;
  cv::absdiff(returned, output, diff);
  EXPECT_EQ(cv::countNonZero(diff.reshape(1)), 0);
}

TEST(VideoTesting, StaticGeoMapOutputOverloadMatchesReturnValue)
{
  cv::Mat map(120, 100, CV_8UC1);
  for (int y = 0; y < map.rows; ++y) {
    for (int x = 0; x < map.cols; ++x) {
      map.at<unsigned char>(y, x) = static_cast<unsigned char>((3 * x + 5 * y) % 251);
    }
  }
  StaticGeoMap geo_map(map, 0.5);

  const cv::Mat returned = geo_map.crop_from_enu(Vector2(2.0, -3.0), 20.0, 64, 18.0, 22.0);
  cv::Mat output;
  geo_map.crop_from_enu(Vector2(2.0, -3.0), 20.0, 64, 18.0, 22.0, output);
  ASSERT_EQ(returned.size(), output.size());
  ASSERT_EQ(returned.type(), output.type());
  cv::Mat diff;
  cv::absdiff(returned, output, diff);
  EXPECT_EQ(cv::countNonZero(diff), 0);
}

TEST(VideoTesting, TrackerShiftedFrameFlowStillWorksWithBufferSwap)
{
  cv::Mat frame_0 = cv::Mat::zeros(120, 160, CV_8UC3);
  cv::circle(frame_0, cv::Point(40, 60), 5, cv::Scalar(255, 255, 255), -1);
  cv::circle(frame_0, cv::Point(80, 45), 4, cv::Scalar(255, 255, 255), -1);
  cv::circle(frame_0, cv::Point(120, 80), 6, cv::Scalar(255, 255, 255), -1);
  cv::Mat frame_1 = cv::Mat::zeros(frame_0.size(), frame_0.type());
  frame_0(cv::Rect(0, 0, frame_0.cols - 4, frame_0.rows)).copyTo(
    frame_1(cv::Rect(4, 0, frame_0.cols - 4, frame_0.rows)));

  VisionOnlyTracker tracker(300.0, 100.0, 200, 3, false);
  auto [valid0, du0, dv0, std0] = tracker.estimate_flow_step(frame_0);
  (void)du0;
  (void)dv0;
  (void)std0;
  EXPECT_FALSE(valid0);
  auto [valid1, du1, dv1, std1] = tracker.estimate_flow_step(frame_1);
  EXPECT_TRUE(valid1);
  EXPECT_NEAR(du1, 4.0, 0.25);
  EXPECT_NEAR(dv1, 0.0, 0.25);
  EXPECT_LT(std1, 1.0);
}

TEST(VideoTesting, TrackerScaledFlowReturnsOriginalPixelShift)
{
  cv::Mat frame_0 = cv::Mat::zeros(240, 320, CV_8UC3);
  cv::circle(frame_0, cv::Point(80, 120), 8, cv::Scalar(255, 255, 255), -1);
  cv::circle(frame_0, cv::Point(160, 90), 7, cv::Scalar(255, 255, 255), -1);
  cv::circle(frame_0, cv::Point(240, 160), 9, cv::Scalar(255, 255, 255), -1);
  cv::Mat frame_1 = cv::Mat::zeros(frame_0.size(), frame_0.type());
  frame_0(cv::Rect(0, 0, frame_0.cols - 12, frame_0.rows)).copyTo(
    frame_1(cv::Rect(12, 0, frame_0.cols - 12, frame_0.rows)));

  VisionOnlyTracker tracker(600.0, 100.0, 200, 3, false, 0.60, 0.95, 160);
  auto [valid0, du0, dv0, std0] = tracker.estimate_flow_step(frame_0);
  (void)du0;
  (void)dv0;
  (void)std0;
  EXPECT_FALSE(valid0);
  auto [valid1, du1, dv1, std1] = tracker.estimate_flow_step(frame_1);
  EXPECT_TRUE(valid1);
  EXPECT_NEAR(du1, 12.0, 0.75);
  EXPECT_NEAR(dv1, 0.0, 0.75);
  EXPECT_LT(std1, 2.0);
}

TEST(VideoTesting, ImageSequenceFrameSourceReadsSortedFrames)
{
  const auto dir = std::filesystem::temp_directory_path() / "advanced_localization_cpp_sequence_test";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  cv::Mat frame_a(24, 32, CV_8UC3, cv::Scalar(10, 20, 30));
  cv::Mat frame_b(24, 32, CV_8UC3, cv::Scalar(40, 50, 60));
  ASSERT_TRUE(cv::imwrite((dir / "0002.png").string(), frame_b));
  ASSERT_TRUE(cv::imwrite((dir / "0001.png").string(), frame_a));

  VideoTestingOptions options;
  options.video_path = dir.string();
  options.frame_source = "image-sequence";
  options.image_sequence_fps = 20.0;
  auto source = make_video_test_frame_source(options);
  source->open();

  const auto metadata = source->metadata();
  EXPECT_EQ(metadata.backend, "image-sequence");
  EXPECT_EQ(metadata.width, 32);
  EXPECT_EQ(metadata.height, 24);
  EXPECT_EQ(metadata.frame_count, 2);
  EXPECT_DOUBLE_EQ(metadata.fps, 20.0);

  Frame frame;
  ASSERT_TRUE(source->read(frame));
  EXPECT_EQ(frame.index, 0);
  EXPECT_DOUBLE_EQ(frame.timestamp_s, 0.0);
  EXPECT_EQ(frame.image.at<cv::Vec3b>(0, 0), cv::Vec3b(10, 20, 30));
  ASSERT_TRUE(source->read(frame));
  EXPECT_EQ(frame.index, 1);
  EXPECT_NEAR(frame.timestamp_s, 0.05, 1e-12);
  EXPECT_EQ(frame.image.at<cv::Vec3b>(0, 0), cv::Vec3b(40, 50, 60));
  EXPECT_FALSE(source->read(frame));
  EXPECT_EQ(source->profile().frames_read, 2);
}

TEST(VideoTesting, RawGrayFrameSourceReadsFrames)
{
  const auto path = std::filesystem::temp_directory_path() / "advanced_localization_cpp_raw_gray_test.raw";
  {
    std::ofstream file(path, std::ios::binary);
    const unsigned char bytes[] = {
      1, 2, 3, 4, 5, 6,
      7, 8, 9, 10, 11, 12};
    file.write(reinterpret_cast<const char *>(bytes), static_cast<std::streamsize>(sizeof(bytes)));
  }

  VideoTestingOptions options;
  options.video_path = path.string();
  options.frame_source = "raw-gray";
  options.raw_width_px = 3;
  options.raw_height_px = 2;
  options.raw_fps = 25.0;
  auto source = make_video_test_frame_source(options);
  source->open();

  const auto metadata = source->metadata();
  EXPECT_EQ(metadata.backend, "raw-gray");
  EXPECT_EQ(metadata.codec, "predecoded-gray8");
  EXPECT_EQ(metadata.width, 3);
  EXPECT_EQ(metadata.height, 2);
  EXPECT_EQ(metadata.frame_count, 2);
  EXPECT_DOUBLE_EQ(metadata.fps, 25.0);

  Frame frame;
  ASSERT_TRUE(source->read(frame));
  EXPECT_EQ(frame.index, 0);
  EXPECT_EQ(frame.image.type(), CV_8UC1);
  EXPECT_EQ(frame.image.at<unsigned char>(0, 0), 1);
  EXPECT_EQ(frame.image.at<unsigned char>(1, 2), 6);
  ASSERT_TRUE(source->read(frame));
  EXPECT_EQ(frame.index, 1);
  EXPECT_NEAR(frame.timestamp_s, 0.04, 1e-12);
  EXPECT_EQ(frame.image.at<unsigned char>(0, 0), 7);
  EXPECT_EQ(frame.image.at<unsigned char>(1, 2), 12);
  EXPECT_FALSE(source->read(frame));
  EXPECT_EQ(source->profile().frames_read, 2);
}

TEST(VideoTesting, VisualLocalizerSyntheticMatchRemainsValid)
{
  const auto dir = std::filesystem::temp_directory_path();
  const auto superpoint_path = dir / "advanced_localization_cpp_superpoint_dummy.onnx";
  const auto lightglue_path = dir / "advanced_localization_cpp_lightglue_dummy.onnx";
  {
    std::ofstream(superpoint_path) << "dummy";
    std::ofstream(lightglue_path) << "dummy";
  }

  SystemConfig config;
  config.lightglue.enabled = true;
  config.lightglue.superpoint_onnx_path = superpoint_path.string();
  config.lightglue.lightglue_onnx_path = lightglue_path.string();
  config.lightglue.min_matches = 4;
  config.lightglue.min_inliers = 3;
  config.lightglue.ransac_reproj_threshold_px = 3.0;

  OnnxTensorRtVisualLocalizer localizer(config);
  ASSERT_TRUE(localizer.is_ready()) << localizer.startup_error();

  cv::Mat map = cv::Mat::zeros(240, 240, CV_8UC1);
  for (int i = 0; i < 24; ++i) {
    const int x = 20 + (i * 37) % 190;
    const int y = 20 + (i * 53) % 190;
    cv::circle(map, cv::Point(x, y), 4 + (i % 4), cv::Scalar(180 + (i % 60)), -1);
    cv::line(map, cv::Point(x, y), cv::Point((x + 17) % 240, (y + 11) % 240), cv::Scalar(120 + (i % 90)), 1);
  }
  const cv::Mat camera = map.clone();

  const auto measurement = localizer.run(map, camera, Vector2::Zero(), 0.2, 0.0, 1.0, 0.2);
  ASSERT_TRUE(measurement.has_value());
  EXPECT_GE(measurement->match_count, config.lightglue.min_matches);
  EXPECT_GE(measurement->inlier_count, config.lightglue.min_inliers);
}
