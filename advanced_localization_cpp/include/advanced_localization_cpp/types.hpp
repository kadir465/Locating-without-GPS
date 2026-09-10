#pragma once

#include <Eigen/Dense>
#include <limits>
#include <optional>
#include <string>

namespace advanced_localization
{

using Vector2 = Eigen::Vector2d;
using Vector3 = Eigen::Vector3d;
using Vector4 = Eigen::Vector4d;
using Matrix2 = Eigen::Matrix2d;
using Matrix3 = Eigen::Matrix3d;
using Matrix16 = Eigen::Matrix<double, 16, 16>;
using Vector16 = Eigen::Matrix<double, 16, 1>;

struct ImuSample
{
  double timestamp_s{0.0};
  Vector3 accel_mps2{Vector3::Zero()};
  Vector3 gyro_rps{Vector3::Zero()};
};

struct VisionMeasurement
{
  double timestamp_s{0.0};
  Vector2 position_xy_enu_m{Vector2::Zero()};
  double yaw_rad{0.0};
  Eigen::MatrixXd covariance{Matrix2::Identity()};
  std::string source{"vision"};
  double processing_latency_s{0.0};
  double scale{1.0};
  double inlier_ratio{1.0};
  int vision_dof{2};
  bool yaw_valid{false};
  std::string model{"SE2_PLANAR"};
  double match_confidence{1.0};
  int match_count{0};
  int inlier_count{0};
  double reprojection_rmse_px{std::numeric_limits<double>::quiet_NaN()};
  bool failsafe_mode{false};
  double confidence_weight{1.0};
};

struct FlowMeasurement
{
  double timestamp_s{0.0};
  Vector2 velocity_xy_mps{Vector2::Zero()};
  Matrix2 covariance_2x2{Matrix2::Identity()};
  std::string source{"optical_flow"};
  int feature_count{0};
};

struct FilterSnapshot
{
  double timestamp_s{0.0};
  Vector3 position_enu_m{Vector3::Zero()};
  Vector3 velocity_enu_mps{Vector3::Zero()};
  Vector4 quaternion_wxyz{1.0, 0.0, 0.0, 0.0};
  Vector3 accel_bias_mps2{Vector3::Zero()};
  Vector3 gyro_bias_rps{Vector3::Zero()};
  double time_delay_s{0.0};
  Matrix16 covariance_16x16{Matrix16::Identity()};
};

struct UpdateResult
{
  bool accepted{false};
  std::string reason;
  std::string source;
  std::optional<double> mahalanobis_distance;
  std::optional<Eigen::VectorXd> innovation;
};

enum class FailsafeCode
{
  None,
  VisionLoss,
  Inconsistency,
  HighCovariance,
};

inline std::string to_string(FailsafeCode code)
{
  switch (code) {
    case FailsafeCode::None:
      return "NONE";
    case FailsafeCode::VisionLoss:
      return "VISION_LOSS";
    case FailsafeCode::Inconsistency:
      return "INCONSISTENCY";
    case FailsafeCode::HighCovariance:
      return "HIGH_COVARIANCE";
  }
  return "UNKNOWN";
}

}  // namespace advanced_localization
