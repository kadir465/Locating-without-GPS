#pragma once

#include "advanced_localization_cpp/config.hpp"
#include "advanced_localization_cpp/ring_buffer.hpp"
#include "advanced_localization_cpp/types.hpp"

#include <optional>

namespace advanced_localization
{

struct FilterHealth
{
  int consecutive_rejections{0};
  int consecutive_lightglue_rejections{0};
  int consecutive_vision_loss{0};
  double position_trace_xy_m2{0.0};
  double velocity_trace_m2ps2{0.0};
  double pzz_m2{0.0};
  std::string last_update_source{"none"};
};

class CascadedESKF
{
public:
  explicit CascadedESKF(const SystemConfig & config);

  bool is_initialized() const { return is_initialized_; }
  std::optional<double> timestamp_s() const { return timestamp_s_; }
  Vector3 position_enu_m() const { return position_enu_m_; }
  Vector3 velocity_enu_mps() const { return velocity_enu_mps_; }
  Vector4 quaternion_wxyz() const { return quaternion_wxyz_; }
  double yaw_rad() const;
  Matrix16 covariance_16x16() const { return P_; }
  Vector2 position_covariance_xy_m2() const;
  double pzz_m2() const { return P_(2, 2); }
  double velocity_trace_m2ps2() const { return P_(3, 3) + P_(4, 4) + P_(5, 5); }
  double time_delay_s() const { return time_delay_s_; }
  FilterHealth health() const;

  void initialize(const Vector3 & position_enu_m, double yaw_rad, double timestamp_s);
  void predict_from_imu(const ImuSample & imu_sample);
  UpdateResult update_from_vision(const VisionMeasurement & measurement);
  UpdateResult update_from_flow(const FlowMeasurement & measurement);
  UpdateResult apply_historical_vision_update(const VisionMeasurement & measurement);
  void inflate_altitude_uncertainty(double target_variance_m2);
  void set_altitude_estimate(double z_enu_m);
  void register_vision_loss();
  UpdateResult reject_vision_measurement(const std::string & source, const std::string & reason);

private:
  void predict_internal(const ImuSample & imu_sample, bool record_imu, bool record_snapshot);
  void build_continuous_jacobians(const Matrix3 & rotation_bw, const Vector3 & accel_unbiased, const Vector3 & gyro_unbiased);
  UpdateResult update_from_flow_internal(const FlowMeasurement & measurement, bool record_history);
  bool passes_kinematic_gate(const VisionMeasurement & measurement) const;
  Eigen::MatrixXd apply_measurement_covariance_floor(const Eigen::MatrixXd & covariance) const;
  void inject_error_state(const Vector16 & delta_x);
  FilterSnapshot snapshot() const;
  void restore_snapshot(const FilterSnapshot & snapshot);
  UpdateResult reject(
    const std::string & source,
    const std::string & reason,
    std::optional<double> mahalanobis_distance = std::nullopt,
    std::optional<Eigen::VectorXd> innovation = std::nullopt);

  SystemConfig config_;
  Vector3 position_enu_m_{Vector3::Zero()};
  Vector3 velocity_enu_mps_{Vector3::Zero()};
  Vector4 quaternion_wxyz_{1.0, 0.0, 0.0, 0.0};
  Vector3 accel_bias_mps2_{Vector3::Zero()};
  Vector3 gyro_bias_rps_{Vector3::Zero()};
  double time_delay_s_{0.0};
  Matrix16 P_{Matrix16::Identity()};
  bool is_initialized_{false};
  std::optional<double> timestamp_s_;
  std::optional<double> last_accepted_measurement_time_s_;
  std::optional<Vector2> last_accepted_position_xy_m_;
  int consecutive_rejections_{0};
  int consecutive_lightglue_rejections_{0};
  int consecutive_vision_loss_{0};
  std::string last_update_source_{"none"};

  TimeIndexedRingBuffer<FilterSnapshot> state_history_;
  TimeIndexedRingBuffer<ImuSample> imu_history_;
  TimeIndexedRingBuffer<FlowMeasurement> flow_history_;

  Matrix16 F_{Matrix16::Zero()};
  Matrix16 Phi_{Matrix16::Identity()};
  Eigen::Matrix<double, 16, 12> G_{Eigen::Matrix<double, 16, 12>::Zero()};
  Eigen::Matrix<double, 12, 12> Qd_{Eigen::Matrix<double, 12, 12>::Zero()};
  Eigen::Matrix<double, 3, 16> H3_{Eigen::Matrix<double, 3, 16>::Zero()};
  Eigen::Matrix<double, 2, 16> H2_{Eigen::Matrix<double, 2, 16>::Zero()};
  Eigen::Matrix<double, 2, 16> H_flow_{Eigen::Matrix<double, 2, 16>::Zero()};
};

}  // namespace advanced_localization
