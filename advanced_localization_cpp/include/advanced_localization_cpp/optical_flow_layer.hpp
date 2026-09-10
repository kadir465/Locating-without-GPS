#pragma once

#include "advanced_localization_cpp/config.hpp"
#include "advanced_localization_cpp/types.hpp"

#include <opencv2/core.hpp>
#include <optional>
#include <vector>

namespace advanced_localization
{

class OpticalFlowLayer
{
public:
  OpticalFlowLayer(const SystemConfig & config, const Matrix3 & camera_matrix);

  std::optional<FlowMeasurement> run(const cv::Mat & frame_gray, double timestamp_s, double altitude_m, double yaw_rad);

private:
  cv::Mat prepare_frame(const cv::Mat & frame_gray, double & scale);
  void bootstrap(const cv::Mat & frame_gray, double timestamp_s, double scale);
  std::vector<cv::Point2f> detect_features(const cv::Mat & frame_gray) const;
  Vector2 pixel_flow_to_velocity_enu(const cv::Point2f & flow_px, double dt, double altitude_m, double yaw_rad) const;

  SystemConfig config_;
  Matrix3 camera_matrix_;
  cv::Mat prev_frame_;
  cv::Mat resized_frame_;
  std::optional<double> prev_timestamp_s_;
  double prev_frame_scale_{1.0};
  std::vector<cv::Point2f> prev_points_;
};

}  // namespace advanced_localization
