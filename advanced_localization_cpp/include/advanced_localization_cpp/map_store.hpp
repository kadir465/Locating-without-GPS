#pragma once

#include "advanced_localization_cpp/types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

namespace advanced_localization
{

class StaticGeoMap
{
public:
  StaticGeoMap() = default;
  StaticGeoMap(const cv::Mat & grayscale_map, double gsd_m_per_px);

  static StaticGeoMap from_path(const std::string & image_path, double gsd_m_per_px, bool apply_clahe);
  cv::Mat crop_from_enu(
    const Vector2 & center_xy_enu_m,
    double window_size_m,
    int output_size_px,
    double window_size_x_m,
    double window_size_y_m) const;
  void crop_from_enu(
    const Vector2 & center_xy_enu_m,
    double window_size_m,
    int output_size_px,
    double window_size_x_m,
    double window_size_y_m,
    cv::Mat & output) const;

  bool valid() const { return !map_.empty(); }
  double gsd_m_per_px() const { return gsd_m_per_px_; }

private:
  Eigen::Vector2d enu_to_px(const Vector2 & xy_enu_m) const;

  cv::Mat map_;
  double gsd_m_per_px_{0.0};
  int height_{0};
  int width_{0};
  Eigen::Vector2d center_px_{Eigen::Vector2d::Zero()};
  mutable cv::Ptr<cv::CLAHE> crop_clahe_;
};

}  // namespace advanced_localization
