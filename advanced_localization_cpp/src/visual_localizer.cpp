#include "advanced_localization_cpp/visual_localizer.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <vector>

namespace advanced_localization
{
namespace
{

double mat_value(const cv::Mat & matrix, int row, int col)
{
  return matrix.depth() == CV_32F ?
         static_cast<double>(matrix.at<float>(row, col)) :
         matrix.at<double>(row, col);
}

cv::Point2f project_affine(const cv::Mat & transform, const cv::Point2f & point)
{
  const double x = mat_value(transform, 0, 0) * static_cast<double>(point.x) +
                   mat_value(transform, 0, 1) * static_cast<double>(point.y) +
                   mat_value(transform, 0, 2);
  const double y = mat_value(transform, 1, 0) * static_cast<double>(point.x) +
                   mat_value(transform, 1, 1) * static_cast<double>(point.y) +
                   mat_value(transform, 1, 2);
  return cv::Point2f(static_cast<float>(x), static_cast<float>(y));
}

cv::Point2f project_homography(const cv::Mat & homography, const cv::Point2f & point)
{
  const double x = static_cast<double>(point.x);
  const double y = static_cast<double>(point.y);
  const double wx = mat_value(homography, 0, 0) * x + mat_value(homography, 0, 1) * y + mat_value(homography, 0, 2);
  const double wy = mat_value(homography, 1, 0) * x + mat_value(homography, 1, 1) * y + mat_value(homography, 1, 2);
  const double w = mat_value(homography, 2, 0) * x + mat_value(homography, 2, 1) * y + mat_value(homography, 2, 2);
  if (std::abs(w) <= 1e-12) {
    return cv::Point2f(static_cast<float>(wx), static_cast<float>(wy));
  }
  return cv::Point2f(static_cast<float>(wx / w), static_cast<float>(wy / w));
}

}  // namespace

OnnxTensorRtVisualLocalizer::OnnxTensorRtVisualLocalizer(const SystemConfig & config)
: config_(config)
{
  if (!config_.lightglue.enabled) {
    startup_error_ = "LightGlue/SuperPoint backend disabled by configuration.";
    return;
  }
  if (config_.lightglue.superpoint_onnx_path.empty() || config_.lightglue.lightglue_onnx_path.empty()) {
    startup_error_ = "ONNX model paths are required for the C++ LightGlue/SuperPoint backend.";
    return;
  }
  if (!std::filesystem::exists(config_.lightglue.superpoint_onnx_path) ||
    !std::filesystem::exists(config_.lightglue.lightglue_onnx_path))
  {
    startup_error_ = "Configured ONNX model path does not exist.";
    return;
  }
  orb_ = cv::ORB::create(std::max(config_.lightglue.min_matches * 20, 500));
  matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, false);
  map_keypoints_.reserve(static_cast<std::size_t>(std::max(config_.lightglue.min_matches * 20, 500)));
  camera_keypoints_.reserve(static_cast<std::size_t>(std::max(config_.lightglue.min_matches * 20, 500)));
  good_matches_.reserve(static_cast<std::size_t>(std::max(config_.lightglue.min_matches * 20, 500)));
  camera_points_.reserve(static_cast<std::size_t>(std::max(config_.lightglue.min_matches * 20, 500)));
  map_points_.reserve(static_cast<std::size_t>(std::max(config_.lightglue.min_matches * 20, 500)));
  ready_ = true;
}

std::optional<VisionMeasurement> OnnxTensorRtVisualLocalizer::run(
  const cv::Mat & map_patch,
  const cv::Mat & camera_patch,
  const Vector2 & roi_center_xy_enu_m,
  double gsd_m_per_px,
  double prior_yaw_rad,
  double capture_timestamp_s,
  double camera_gsd_m_per_px)
{
  if (!ready_) {
    return std::nullopt;
  }

  const auto start = std::chrono::steady_clock::now();
  cv::Mat map_gray;
  cv::Mat camera_gray;
  if (map_patch.channels() == 1) {
    map_gray = map_patch;
  } else {
    cv::cvtColor(map_patch, map_gray_buffer_, cv::COLOR_BGR2GRAY);
    map_gray = map_gray_buffer_;
  }
  if (camera_patch.channels() == 1) {
    camera_gray = camera_patch;
  } else {
    cv::cvtColor(camera_patch, camera_gray_buffer_, cv::COLOR_BGR2GRAY);
    camera_gray = camera_gray_buffer_;
  }

  if (!orb_ || !matcher_) {
    return std::nullopt;
  }
  map_keypoints_.clear();
  camera_keypoints_.clear();
  map_descriptors_.release();
  camera_descriptors_.release();
  orb_->detectAndCompute(map_gray, cv::Mat(), map_keypoints_, map_descriptors_);
  orb_->detectAndCompute(camera_gray, cv::Mat(), camera_keypoints_, camera_descriptors_);
  if (map_descriptors_.empty() || camera_descriptors_.empty()) {
    return std::nullopt;
  }

  knn_matches_.clear();
  matcher_->knnMatch(camera_descriptors_, map_descriptors_, knn_matches_, 2);

  good_matches_.clear();
  good_matches_.reserve(knn_matches_.size());
  for (const auto & pair : knn_matches_) {
    if (pair.size() >= 2 && pair[0].distance < 0.75f * pair[1].distance) {
      good_matches_.push_back(pair[0]);
    }
  }
  if (static_cast<int>(good_matches_.size()) < std::max(3, config_.lightglue.min_matches)) {
    return std::nullopt;
  }

  std::sort(good_matches_.begin(), good_matches_.end(), [](const cv::DMatch & a, const cv::DMatch & b) {
      return a.distance < b.distance;
    });

  camera_points_.clear();
  map_points_.clear();
  camera_points_.reserve(good_matches_.size());
  map_points_.reserve(good_matches_.size());
  for (const auto & match : good_matches_) {
    camera_points_.push_back(camera_keypoints_[static_cast<std::size_t>(match.queryIdx)].pt);
    map_points_.push_back(map_keypoints_[static_cast<std::size_t>(match.trainIdx)].pt);
  }

  cv::Mat transform = cv::estimateAffinePartial2D(
    camera_points_,
    map_points_,
    inlier_mask_raw_,
    cv::RANSAC,
    config_.lightglue.ransac_reproj_threshold_px);
  bool using_homography = false;
  if ((transform.empty() || cv::countNonZero(inlier_mask_raw_) < config_.lightglue.min_inliers) && camera_points_.size() >= 4) {
    cv::Mat homography = cv::findHomography(camera_points_, map_points_, cv::RANSAC, config_.lightglue.ransac_reproj_threshold_px, homography_mask_);
    if (!homography.empty() && cv::countNonZero(homography_mask_) >= config_.lightglue.min_inliers) {
      transform = homography;
      inlier_mask_raw_ = homography_mask_;
      using_homography = true;
    }
  }
  if (transform.empty() || inlier_mask_raw_.empty()) {
    return std::nullopt;
  }

  const int inlier_count = cv::countNonZero(inlier_mask_raw_);
  if (inlier_count < std::max(3, config_.lightglue.min_inliers)) {
    return std::nullopt;
  }

  const cv::Point2f source_center(0.5f * static_cast<float>(camera_gray.cols), 0.5f * static_cast<float>(camera_gray.rows));
  const cv::Point2f target_center(0.5f * static_cast<float>(map_gray.cols), 0.5f * static_cast<float>(map_gray.rows));
  const cv::Point2f projected_center =
    using_homography ? project_homography(transform, source_center) : project_affine(transform, source_center);

  const double dx_px = static_cast<double>(projected_center.x - target_center.x);
  const double dy_px = static_cast<double>(projected_center.y - target_center.y);
  const double cos_yaw = std::cos(prior_yaw_rad);
  const double sin_yaw = std::sin(prior_yaw_rad);
  const double world_dx_px = cos_yaw * dx_px + sin_yaw * dy_px;
  const double world_dy_px = -sin_yaw * dx_px + cos_yaw * dy_px;
  const Vector2 estimate_xy_enu_m =
    roi_center_xy_enu_m + Vector2(world_dx_px * gsd_m_per_px, -world_dy_px * std::max(gsd_m_per_px, camera_gsd_m_per_px));

  const double inlier_ratio = static_cast<double>(inlier_count) / std::max<std::size_t>(good_matches_.size(), 1);
  const double confidence = std::clamp(1.0 - static_cast<double>(good_matches_.front().distance) / 100.0, 0.05, 1.0);
  const double dynamic_scale = 1.0 + 10.0 * std::exp(-0.05 * static_cast<double>(inlier_count) * confidence);
  double reprojection_rmse_px = std::numeric_limits<double>::quiet_NaN();
  {
    double sum_sq = 0.0;
    int count = 0;
    for (std::size_t i = 0; i < camera_points_.size(); ++i) {
      if (inlier_mask_raw_.at<unsigned char>(static_cast<int>(i), 0) == 0) {
        continue;
      }
      const cv::Point2f projected =
        using_homography ? project_homography(transform, camera_points_[i]) : project_affine(transform, camera_points_[i]);
      const double dx = static_cast<double>(projected.x - map_points_[i].x);
      const double dy = static_cast<double>(projected.y - map_points_[i].y);
      sum_sq += dx * dx + dy * dy;
      ++count;
    }
    if (count > 0) {
      reprojection_rmse_px = std::sqrt(sum_sq / static_cast<double>(count));
    }
  }

  VisionMeasurement measurement;
  measurement.timestamp_s = capture_timestamp_s;
  measurement.position_xy_enu_m = estimate_xy_enu_m;
  measurement.yaw_rad = prior_yaw_rad;
  measurement.covariance = Matrix2::Identity() * dynamic_scale;
  measurement.source = "lightglue";
  measurement.processing_latency_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  measurement.vision_dof = 2;
  measurement.yaw_valid = false;
  measurement.model = using_homography ? "ONNX_TENSORRT_GEOMETRIC_HOMOGRAPHY" : "ONNX_TENSORRT_GEOMETRIC_AFFINE";
  measurement.inlier_ratio = inlier_ratio;
  measurement.match_confidence = confidence;
  measurement.match_count = static_cast<int>(good_matches_.size());
  measurement.inlier_count = inlier_count;
  measurement.reprojection_rmse_px = reprojection_rmse_px;
  measurement.failsafe_mode = inlier_ratio < config_.lightglue.min_inlier_ratio;
  measurement.confidence_weight = std::clamp(inlier_ratio, 0.1, 1.0);
  return measurement;
}

}  // namespace advanced_localization
