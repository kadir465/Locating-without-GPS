#pragma once

#include "advanced_localization_cpp/types.hpp"

namespace advanced_localization
{

struct RoiWindow
{
  Vector2 center_xy_enu_m{Vector2::Zero()};
  double size_m{0.0};
  double width_m{0.0};
  double height_m{0.0};
  double predicted_latency_s{0.0};
  double padding_x_m{0.0};
  double padding_y_m{0.0};
};

class ShiftedRoiSelector
{
public:
  ShiftedRoiSelector(double base_window_size_m, double min_speed_mps, double max_window_size_m, double padding_sigma);

  RoiWindow select(
    const Vector2 & position_xy_enu_m,
    const Vector2 & velocity_xy_mps,
    const Vector2 & acceleration_xy_mps2,
    const Eigen::MatrixXd & covariance_xy_m2,
    double latency_s) const;

private:
  double base_window_size_m_;
  double min_speed_mps_;
  double max_window_size_m_;
  double padding_sigma_;
};

}  // namespace advanced_localization
