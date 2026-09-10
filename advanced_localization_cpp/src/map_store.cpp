#include "advanced_localization_cpp/map_store.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <stdexcept>

namespace advanced_localization
{

StaticGeoMap::StaticGeoMap(const cv::Mat & grayscale_map, double gsd_m_per_px)
: gsd_m_per_px_(gsd_m_per_px)
{
  if (grayscale_map.empty() || grayscale_map.channels() != 1) {
    throw std::invalid_argument("grayscale_map must be single-channel");
  }
  if (gsd_m_per_px <= 0.0) {
    throw std::invalid_argument("gsd_m_per_px must be positive");
  }
  grayscale_map.copyTo(map_);
  height_ = map_.rows;
  width_ = map_.cols;
  center_px_ = Eigen::Vector2d(0.5 * static_cast<double>(width_), 0.5 * static_cast<double>(height_));
  crop_clahe_ = cv::createCLAHE(2.0, cv::Size(8, 8));
}

StaticGeoMap StaticGeoMap::from_path(const std::string & image_path, double gsd_m_per_px, bool apply_clahe)
{
  cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
  if (image.empty()) {
    throw std::runtime_error("failed to load map image: " + image_path);
  }
  if (apply_clahe) {
    auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(image, image);
  }
  return StaticGeoMap(image, gsd_m_per_px);
}

cv::Mat StaticGeoMap::crop_from_enu(
  const Vector2 & center_xy_enu_m,
  double window_size_m,
  int output_size_px,
  double window_size_x_m,
  double window_size_y_m) const
{
  cv::Mat output;
  crop_from_enu(center_xy_enu_m, window_size_m, output_size_px, window_size_x_m, window_size_y_m, output);
  return output;
}

void StaticGeoMap::crop_from_enu(
  const Vector2 & center_xy_enu_m,
  double window_size_m,
  int output_size_px,
  double window_size_x_m,
  double window_size_y_m,
  cv::Mat & output) const
{
  if (!valid()) {
    throw std::runtime_error("map is not initialized");
  }
  if (output_size_px < 8) {
    throw std::invalid_argument("output_size_px must be >= 8");
  }
  const Eigen::Vector2d center_px = enu_to_px(center_xy_enu_m);
  const int window_size_x_px = std::max(8, static_cast<int>(std::round((window_size_x_m > 0.0 ? window_size_x_m : window_size_m) / gsd_m_per_px_)));
  const int window_size_y_px = std::max(8, static_cast<int>(std::round((window_size_y_m > 0.0 ? window_size_y_m : window_size_m) / gsd_m_per_px_)));

  cv::getRectSubPix(map_, cv::Size(window_size_x_px, window_size_y_px), cv::Point2f(center_px.x(), center_px.y()), output);
  if (output.rows != output_size_px || output.cols != output_size_px) {
    cv::resize(output, output, cv::Size(output_size_px, output_size_px), 0.0, 0.0, cv::INTER_LINEAR);
  }
  if (crop_clahe_) {
    crop_clahe_->apply(output, output);
  }
}

Eigen::Vector2d StaticGeoMap::enu_to_px(const Vector2 & xy_enu_m) const
{
  return Eigen::Vector2d(center_px_.x() + xy_enu_m.x() / gsd_m_per_px_, center_px_.y() - xy_enu_m.y() / gsd_m_per_px_);
}

}  // namespace advanced_localization
