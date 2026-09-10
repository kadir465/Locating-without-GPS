#pragma once

#include "advanced_localization_cpp/config.hpp"
#include "advanced_localization_cpp/types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <optional>
#include <vector>

namespace advanced_localization
{

class VisualLocalizer
{
public:
  virtual ~VisualLocalizer() = default;
  virtual bool is_ready() const = 0;
  virtual std::string startup_error() const = 0;
  virtual std::optional<VisionMeasurement> run(
    const cv::Mat & map_patch,
    const cv::Mat & camera_patch,
    const Vector2 & roi_center_xy_enu_m,
    double gsd_m_per_px,
    double prior_yaw_rad,
    double capture_timestamp_s,
    double camera_gsd_m_per_px) = 0;
};

class OnnxTensorRtVisualLocalizer : public VisualLocalizer
{
public:
  explicit OnnxTensorRtVisualLocalizer(const SystemConfig & config);

  bool is_ready() const override { return ready_; }
  std::string startup_error() const override { return startup_error_; }
  std::optional<VisionMeasurement> run(
    const cv::Mat & map_patch,
    const cv::Mat & camera_patch,
    const Vector2 & roi_center_xy_enu_m,
    double gsd_m_per_px,
    double prior_yaw_rad,
    double capture_timestamp_s,
    double camera_gsd_m_per_px) override;

private:
  SystemConfig config_;
  bool ready_{false};
  std::string startup_error_;
  cv::Ptr<cv::ORB> orb_;
  cv::Ptr<cv::BFMatcher> matcher_;
  cv::Mat map_gray_buffer_;
  cv::Mat camera_gray_buffer_;
  cv::Mat map_descriptors_;
  cv::Mat camera_descriptors_;
  cv::Mat inlier_mask_raw_;
  cv::Mat homography_mask_;
  std::vector<cv::KeyPoint> map_keypoints_;
  std::vector<cv::KeyPoint> camera_keypoints_;
  std::vector<std::vector<cv::DMatch>> knn_matches_;
  std::vector<cv::DMatch> good_matches_;
  std::vector<cv::Point2f> camera_points_;
  std::vector<cv::Point2f> map_points_;
};

class TensorRtLightGlueVisualLocalizer : public OnnxTensorRtVisualLocalizer
{
public:
  explicit TensorRtLightGlueVisualLocalizer(const SystemConfig & config)
  : OnnxTensorRtVisualLocalizer(config)
  {
  }
};

}  // namespace advanced_localization
