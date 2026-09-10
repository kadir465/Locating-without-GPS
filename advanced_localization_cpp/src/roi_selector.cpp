#include "advanced_localization_cpp/roi_selector.hpp"

#include <algorithm>
#include <cmath>

namespace advanced_localization
{

ShiftedRoiSelector::ShiftedRoiSelector(
  double base_window_size_m,
  double min_speed_mps,
  double max_window_size_m,
  double padding_sigma)
: base_window_size_m_(base_window_size_m),
  min_speed_mps_(min_speed_mps),
  max_window_size_m_(max_window_size_m),
  padding_sigma_(padding_sigma)
{
}

RoiWindow ShiftedRoiSelector::select(
  const Vector2 & position_xy_enu_m,
  const Vector2 & velocity_xy_mps,
  const Vector2 & acceleration_xy_mps2,
  const Eigen::MatrixXd & covariance_xy_m2,
  double latency_s) const
{
  latency_s = std::max(0.0, latency_s);
  const double speed = velocity_xy_mps.norm();
  const Vector2 center = position_xy_enu_m + velocity_xy_mps * latency_s + 0.5 * acceleration_xy_mps2 * latency_s * latency_s;

  double footprint_width_m = base_window_size_m_;
  double footprint_height_m = base_window_size_m_;
  if (speed >= min_speed_mps_) {
    const Vector2 direction = velocity_xy_mps / speed;
    double forward_shift_m =
      speed * latency_s + 0.5 * std::max(0.0, acceleration_xy_mps2.dot(direction)) * latency_s * latency_s;
    forward_shift_m = std::clamp(forward_shift_m, 0.0, 0.8 * base_window_size_m_);
    const double major_m = base_window_size_m_ + forward_shift_m;
    const double minor_m = base_window_size_m_;
    const Vector2 lateral(-direction.y(), direction.x());
    footprint_width_m = std::abs(direction.x()) * major_m + std::abs(lateral.x()) * minor_m;
    footprint_height_m = std::abs(direction.y()) * major_m + std::abs(lateral.y()) * minor_m;
  }

  double p_xx = base_window_size_m_ * base_window_size_m_;
  double p_yy = p_xx;
  if (covariance_xy_m2.rows() == 2 && covariance_xy_m2.cols() == 2) {
    p_xx = covariance_xy_m2(0, 0);
    p_yy = covariance_xy_m2(1, 1);
  } else if (covariance_xy_m2.size() == 2) {
    p_xx = covariance_xy_m2(0);
    p_yy = covariance_xy_m2(1);
  }

  const double padding_x_m = padding_sigma_ * std::sqrt(std::max(p_xx, 1e-6));
  const double padding_y_m = padding_sigma_ * std::sqrt(std::max(p_yy, 1e-6));
  const double width_m = std::clamp(footprint_width_m + padding_x_m, base_window_size_m_, max_window_size_m_);
  const double height_m = std::clamp(footprint_height_m + padding_y_m, base_window_size_m_, max_window_size_m_);
  return RoiWindow{center, std::max(width_m, height_m), width_m, height_m, latency_s, padding_x_m, padding_y_m};
}

}  // namespace advanced_localization
