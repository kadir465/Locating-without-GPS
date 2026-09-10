#include "advanced_localization_cpp/optical_flow_layer.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cmath>

namespace advanced_localization
{

OpticalFlowLayer::OpticalFlowLayer(const SystemConfig & config, const Matrix3 & camera_matrix)
: config_(config),
  camera_matrix_(camera_matrix)
{
}

std::optional<FlowMeasurement> OpticalFlowLayer::run(const cv::Mat & frame_gray, double timestamp_s, double altitude_m, double yaw_rad)
{
  if (frame_gray.empty()) {
    return std::nullopt;
  }
  double frame_scale = 1.0;
  const cv::Mat frame_work = prepare_frame(frame_gray, frame_scale);
  if (prev_frame_.empty() || !prev_timestamp_s_.has_value()) {
    bootstrap(frame_work, timestamp_s, frame_scale);
    return std::nullopt;
  }

  const double dt = timestamp_s - *prev_timestamp_s_;
  if (dt <= 1e-6) {
    return std::nullopt;
  }

  if (prev_points_.size() < static_cast<std::size_t>(config_.flow.min_features)) {
    prev_points_ = detect_features(prev_frame_);
    if (prev_points_.size() < 4) {
      bootstrap(frame_work, timestamp_s, frame_scale);
      return std::nullopt;
    }
  }

  std::vector<cv::Point2f> next_points;
  std::vector<uchar> status_fw;
  std::vector<float> error_fw;
  cv::calcOpticalFlowPyrLK(prev_frame_, frame_work, prev_points_, next_points, status_fw, error_fw, cv::Size(21, 21), 3);
  if (next_points.empty() || status_fw.empty()) {
    bootstrap(frame_work, timestamp_s, frame_scale);
    return std::nullopt;
  }

  std::vector<cv::Point2f> back_points;
  std::vector<uchar> status_bw;
  std::vector<float> error_bw;
  cv::calcOpticalFlowPyrLK(frame_work, prev_frame_, next_points, back_points, status_bw, error_bw, cv::Size(21, 21), 3);
  if (back_points.empty() || status_bw.empty()) {
    bootstrap(frame_work, timestamp_s, frame_scale);
    return std::nullopt;
  }

  std::vector<cv::Point2f> prev_valid;
  std::vector<cv::Point2f> next_valid;
  const double work_to_original_px = 1.0 / std::max(frame_scale, 1e-6);
  const std::size_t count =
    std::min({prev_points_.size(), next_points.size(), back_points.size(), status_fw.size(), status_bw.size()});
  for (std::size_t i = 0; i < count; ++i) {
    const double dx = back_points[i].x - prev_points_[i].x;
    const double dy = back_points[i].y - prev_points_[i].y;
    const double bidirectional_error = std::hypot(dx, dy) * work_to_original_px;
    if (status_fw[i] == 1 && status_bw[i] == 1 && bidirectional_error < config_.flow.max_bidirectional_error_px) {
      prev_valid.push_back(prev_points_[i]);
      next_valid.push_back(next_points[i]);
    }
  }

  if (prev_valid.size() < static_cast<std::size_t>(config_.flow.min_features)) {
    bootstrap(frame_work, timestamp_s, frame_scale);
    return std::nullopt;
  }

  std::vector<float> dx_values;
  std::vector<float> dy_values;
  dx_values.reserve(prev_valid.size());
  dy_values.reserve(prev_valid.size());
  for (std::size_t i = 0; i < prev_valid.size(); ++i) {
    dx_values.push_back(next_valid[i].x - prev_valid[i].x);
    dy_values.push_back(next_valid[i].y - prev_valid[i].y);
  }
  auto median = [](std::vector<float> values) {
      const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
      std::nth_element(values.begin(), mid, values.end());
      return *mid;
    };
  const cv::Point2f flow_px_work(median(dx_values), median(dy_values));
  const cv::Point2f flow_px(
    static_cast<float>(static_cast<double>(flow_px_work.x) * work_to_original_px),
    static_cast<float>(static_cast<double>(flow_px_work.y) * work_to_original_px));
  const Vector2 velocity_xy_enu_mps = pixel_flow_to_velocity_enu(flow_px, dt, altitude_m, yaw_rad);

  const double feature_scale = std::clamp(config_.flow.min_features / std::max(static_cast<double>(prev_valid.size()), 1.0), 0.5, 4.0);
  const double sigma = config_.flow.noise_std_mps * feature_scale;
  Matrix2 covariance = Matrix2::Zero();
  covariance(0, 0) = sigma * sigma;
  covariance(1, 1) = sigma * sigma;

  frame_work.copyTo(prev_frame_);
  prev_timestamp_s_ = timestamp_s;
  prev_frame_scale_ = frame_scale;
  prev_points_ = next_valid;

  return FlowMeasurement{timestamp_s, velocity_xy_enu_mps, covariance, "optical_flow", static_cast<int>(prev_valid.size())};
}

cv::Mat OpticalFlowLayer::prepare_frame(const cv::Mat & frame_gray, double & scale)
{
  scale = 1.0;
  const int max_side_px = config_.flow.max_side_px;
  if (max_side_px <= 0) {
    return frame_gray;
  }
  const int max_side = std::max(frame_gray.cols, frame_gray.rows);
  if (max_side <= max_side_px) {
    return frame_gray;
  }
  scale = static_cast<double>(max_side_px) / static_cast<double>(max_side);
  const int resized_cols = std::max(1, static_cast<int>(std::round(static_cast<double>(frame_gray.cols) * scale)));
  const int resized_rows = std::max(1, static_cast<int>(std::round(static_cast<double>(frame_gray.rows) * scale)));
  cv::resize(frame_gray, resized_frame_, cv::Size(resized_cols, resized_rows), 0.0, 0.0, cv::INTER_AREA);
  return resized_frame_;
}

void OpticalFlowLayer::bootstrap(const cv::Mat & frame_gray, double timestamp_s, double scale)
{
  frame_gray.copyTo(prev_frame_);
  prev_timestamp_s_ = timestamp_s;
  prev_frame_scale_ = scale;
  prev_points_ = detect_features(frame_gray);
}

std::vector<cv::Point2f> OpticalFlowLayer::detect_features(const cv::Mat & frame_gray) const
{
  std::vector<cv::Point2f> points;
  cv::goodFeaturesToTrack(frame_gray, points, 800, 0.01, 6.0, cv::Mat(), 7, false);
  return points;
}

Vector2 OpticalFlowLayer::pixel_flow_to_velocity_enu(const cv::Point2f & flow_px, double dt, double altitude_m, double yaw_rad) const
{
  const double fx = camera_matrix_(0, 0);
  const double fy = camera_matrix_(1, 1);
  const double alt = std::max(1.0, std::abs(altitude_m));
  const double vx_body = static_cast<double>(flow_px.y) * alt / std::max(fy, 1e-6) / dt;
  const double vy_body = -static_cast<double>(flow_px.x) * alt / std::max(fx, 1e-6) / dt;
  const double c = std::cos(yaw_rad);
  const double s = std::sin(yaw_rad);
  return Vector2(c * vx_body - s * vy_body, s * vx_body + c * vy_body);
}

}  // namespace advanced_localization
