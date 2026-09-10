#include "advanced_localization_cpp/eskf.hpp"

#include "advanced_localization_cpp/math_utils.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace advanced_localization
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

bool is_lightglue_source(const std::string & source)
{
  std::string lowered = source;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return lowered == "lightglue";
}

}  // namespace

CascadedESKF::CascadedESKF(const SystemConfig & config)
: config_(config),
  state_history_(std::max(static_cast<int>(config.timing.history_buffer_seconds * config.imu_rate_hz * 4.0), 128)),
  imu_history_(std::max(static_cast<int>(config.timing.history_buffer_seconds * config.imu_rate_hz * 4.0), 128)),
  flow_history_(std::max(static_cast<int>(config.timing.history_buffer_seconds * config.imu_rate_hz * 4.0), 128))
{
  P_.setIdentity();
  P_ *= 1e-3;
  P_.block<3, 3>(0, 0) = Matrix3::Identity() * 2.0;
  P_.block<3, 3>(3, 3) = Matrix3::Identity() * 1.0;
  P_.block<3, 3>(6, 6) = Matrix3::Identity() * (kPi / 18.0);
  P_.block<3, 3>(9, 9) = Matrix3::Identity() * 0.2;
  P_.block<3, 3>(12, 12) = Matrix3::Identity() * 0.05;
  P_(15, 15) = 0.02;

  H3_(0, 0) = 1.0;
  H3_(1, 1) = 1.0;
  H3_(2, 8) = 1.0;
  H2_(0, 0) = 1.0;
  H2_(1, 1) = 1.0;
  H_flow_(0, 3) = 1.0;
  H_flow_(1, 4) = 1.0;
}

double CascadedESKF::yaw_rad() const
{
  return yaw_from_quaternion(quaternion_wxyz_);
}

Vector2 CascadedESKF::position_covariance_xy_m2() const
{
  return Vector2(P_(0, 0), P_(1, 1));
}

FilterHealth CascadedESKF::health() const
{
  return FilterHealth{
    consecutive_rejections_,
    consecutive_lightglue_rejections_,
    consecutive_vision_loss_,
    P_(0, 0) + P_(1, 1),
    velocity_trace_m2ps2(),
    pzz_m2(),
    last_update_source_};
}

void CascadedESKF::initialize(const Vector3 & position_enu_m, double yaw_rad_value, double timestamp_s_value)
{
  position_enu_m_ = position_enu_m;
  velocity_enu_mps_.setZero();
  quaternion_wxyz_ = quaternion_from_yaw(yaw_rad_value);
  accel_bias_mps2_.setZero();
  gyro_bias_rps_.setZero();
  time_delay_s_ = std::clamp(config_.timing.initial_time_delay_s, 0.0, config_.timing.max_time_delay_s);
  timestamp_s_ = timestamp_s_value;
  is_initialized_ = true;

  consecutive_rejections_ = 0;
  consecutive_lightglue_rejections_ = 0;
  consecutive_vision_loss_ = 0;
  last_update_source_ = "init";
  last_accepted_measurement_time_s_ = timestamp_s_value;
  last_accepted_position_xy_m_ = position_enu_m_.head<2>();

  state_history_.clear();
  imu_history_.clear();
  flow_history_.clear();
  state_history_.append(timestamp_s_value, snapshot());
}

void CascadedESKF::predict_from_imu(const ImuSample & imu_sample)
{
  predict_internal(imu_sample, true, true);
}

void CascadedESKF::predict_internal(const ImuSample & imu_sample, bool record_imu, bool record_snapshot)
{
  if (!is_initialized_) {
    return;
  }
  if (record_imu) {
    imu_history_.append(imu_sample.timestamp_s, imu_sample);
  }
  if (!timestamp_s_.has_value()) {
    timestamp_s_ = imu_sample.timestamp_s;
    if (record_snapshot) {
      state_history_.append(*timestamp_s_, snapshot());
    }
    return;
  }

  const double dt = imu_sample.timestamp_s - *timestamp_s_;
  if (dt <= 0.0) {
    return;
  }

  const Vector3 accel_unbiased = imu_sample.accel_mps2 - accel_bias_mps2_;
  const Vector3 gyro_unbiased = imu_sample.gyro_rps - gyro_bias_rps_;
  const Matrix3 rotation_bw = rotation_matrix_from_quaternion(quaternion_wxyz_);
  const Vector3 accel_world = rotation_bw * accel_unbiased + config_.gravity_enu_mps2;

  position_enu_m_ += velocity_enu_mps_ * dt + 0.5 * accel_world * dt * dt;
  velocity_enu_mps_ += accel_world * dt;
  quaternion_wxyz_ = quaternion_normalize(quaternion_multiply(quaternion_wxyz_, quaternion_from_small_angle(gyro_unbiased * dt)));

  build_continuous_jacobians(rotation_bw, accel_unbiased, gyro_unbiased);
  Phi_ = Matrix16::Identity() + F_ * dt;
  Qd_.setZero();
  Qd_.block<3, 3>(0, 0) = Matrix3::Identity() * config_.noise.accel_noise_std_mps2 * config_.noise.accel_noise_std_mps2 * dt;
  Qd_.block<3, 3>(3, 3) = Matrix3::Identity() * config_.noise.gyro_noise_std_rps * config_.noise.gyro_noise_std_rps * dt;
  Qd_.block<3, 3>(6, 6) = Matrix3::Identity() * config_.noise.accel_bias_rw_std_mps2 * config_.noise.accel_bias_rw_std_mps2 * dt;
  Qd_.block<3, 3>(9, 9) = Matrix3::Identity() * config_.noise.gyro_bias_rw_std_rps * config_.noise.gyro_bias_rw_std_rps * dt;

  P_ = force_symmetric(Phi_ * P_ * Phi_.transpose() + G_ * Qd_ * G_.transpose());
  timestamp_s_ = imu_sample.timestamp_s;
  if (record_snapshot) {
    state_history_.append(*timestamp_s_, snapshot());
  }
}

void CascadedESKF::build_continuous_jacobians(const Matrix3 & rotation_bw, const Vector3 & accel_unbiased, const Vector3 & gyro_unbiased)
{
  F_.setZero();
  F_.block<3, 3>(0, 3) = Matrix3::Identity();
  F_.block<3, 3>(3, 6) = -rotation_bw * skew_symmetric(accel_unbiased);
  F_.block<3, 3>(3, 9) = -rotation_bw;
  F_.block<3, 3>(6, 6) = -skew_symmetric(gyro_unbiased);
  F_.block<3, 3>(6, 12) = -Matrix3::Identity();

  G_.setZero();
  G_.block<3, 3>(3, 0) = rotation_bw;
  G_.block<3, 3>(6, 3) = Matrix3::Identity();
  G_.block<3, 3>(9, 6) = Matrix3::Identity();
  G_.block<3, 3>(12, 9) = Matrix3::Identity();
}

UpdateResult CascadedESKF::update_from_vision(const VisionMeasurement & measurement)
{
  if (!is_initialized_) {
    return reject(measurement.source, "filter_not_initialized");
  }
  if (!passes_kinematic_gate(measurement)) {
    return reject(measurement.source, "kinematic_limit_exceeded");
  }

  Eigen::MatrixXd H;
  Eigen::VectorXd predicted;
  Eigen::VectorXd innovation_raw;
  double chi2_threshold = config_.gate.chi2_threshold_2dof;

  if (measurement.vision_dof == 3) {
    H = H3_;
    predicted = Eigen::Vector3d(position_enu_m_.x(), position_enu_m_.y(), yaw_rad());
    innovation_raw = Eigen::Vector3d(
      measurement.position_xy_enu_m.x() - predicted(0),
      measurement.position_xy_enu_m.y() - predicted(1),
      normalize_angle(measurement.yaw_rad - predicted(2)));
    chi2_threshold = config_.gate.chi2_threshold_3dof;
  } else {
    H = H2_;
    predicted = Vector2(position_enu_m_.x(), position_enu_m_.y());
    innovation_raw = Vector2(measurement.position_xy_enu_m.x() - predicted(0), measurement.position_xy_enu_m.y() - predicted(1));
  }

  const Eigen::MatrixXd covariance = apply_measurement_covariance_floor(measurement.covariance);
  Eigen::MatrixXd S = force_symmetric_dynamic(H * P_ * H.transpose() + covariance);
  const Eigen::MatrixXd S_inv = S.ldlt().solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));
  const double mahalanobis_distance = innovation_raw.transpose() * S_inv * innovation_raw;
  if (mahalanobis_distance > chi2_threshold) {
    return reject(measurement.source, "mahalanobis_gate_rejected", mahalanobis_distance, innovation_raw);
  }

  Eigen::VectorXd innovation = innovation_raw;
  const Eigen::VectorXd clipped_innovation_xy = clip_vector_norm(innovation.head(2), config_.gate.max_position_correction_m);
  innovation.head(2) = clipped_innovation_xy;
  if (measurement.vision_dof == 3) {
    innovation(2) = std::clamp(
      innovation(2),
      -config_.gate.max_yaw_correction_deg * kPi / 180.0,
      config_.gate.max_yaw_correction_deg * kPi / 180.0);
  }

  const Eigen::MatrixXd K = P_ * H.transpose() * S_inv;
  Vector16 delta_x = K * innovation;
  const Eigen::VectorXd clipped_delta_xy = clip_vector_norm(delta_x.head(2), config_.gate.max_position_correction_m);
  delta_x.head(2) = clipped_delta_xy;
  if (measurement.vision_dof == 3) {
    delta_x(8) = std::clamp(
      delta_x(8),
      -config_.gate.max_yaw_correction_deg * kPi / 180.0,
      config_.gate.max_yaw_correction_deg * kPi / 180.0);
  }

  inject_error_state(delta_x);
  const Eigen::MatrixXd I_KH = Matrix16::Identity() - K * H;
  P_ = force_symmetric_dynamic(I_KH * P_ * I_KH.transpose() + K * covariance * K.transpose());

  last_accepted_measurement_time_s_ = measurement.timestamp_s;
  last_accepted_position_xy_m_ = measurement.position_xy_enu_m;
  consecutive_rejections_ = 0;
  consecutive_vision_loss_ = 0;
  if (is_lightglue_source(measurement.source)) {
    consecutive_lightglue_rejections_ = 0;
  }
  last_update_source_ = measurement.source;
  if (timestamp_s_.has_value()) {
    state_history_.append(*timestamp_s_, snapshot());
  }
  return UpdateResult{true, "accepted", measurement.source, mahalanobis_distance, innovation};
}

Eigen::MatrixXd CascadedESKF::apply_measurement_covariance_floor(const Eigen::MatrixXd & covariance) const
{
  Eigen::MatrixXd out = covariance;
  const double min_var = config_.gate.min_measurement_variance_m2;
  for (Eigen::Index i = 0; i < out.rows() && i < out.cols(); ++i) {
    out(i, i) = std::max(out(i, i), min_var);
  }
  return out;
}

UpdateResult CascadedESKF::update_from_flow(const FlowMeasurement & measurement)
{
  return update_from_flow_internal(measurement, true);
}

UpdateResult CascadedESKF::update_from_flow_internal(const FlowMeasurement & measurement, bool record_history)
{
  if (!is_initialized_) {
    return reject(measurement.source, "filter_not_initialized");
  }
  if (pzz_m2() > config_.flow.pzz_reject_threshold_m2) {
    return reject(measurement.source, "flow_rejected_pzz_high");
  }

  const Vector2 predicted = velocity_enu_mps_.head<2>();
  const Vector2 innovation_xy = measurement.velocity_xy_mps - predicted;
  Eigen::Matrix2d S = H_flow_ * P_ * H_flow_.transpose() + measurement.covariance_2x2;
  S = force_symmetric_dynamic(S);
  const Matrix2 S_inv = S.inverse();
  const double mahalanobis_distance = innovation_xy.transpose() * S_inv * innovation_xy;
  if (mahalanobis_distance > config_.flow.chi2_threshold_2dof) {
    return reject(measurement.source, "flow_mahalanobis_rejected", mahalanobis_distance, innovation_xy);
  }

  const Eigen::Matrix<double, 16, 2> K = P_ * H_flow_.transpose() * S_inv;
  const Vector16 delta_x = K * innovation_xy;
  inject_error_state(delta_x);
  const Matrix16 I_KH = Matrix16::Identity() - K * H_flow_;
  P_ = force_symmetric(I_KH * P_ * I_KH.transpose() + K * measurement.covariance_2x2 * K.transpose());
  last_update_source_ = measurement.source;

  if (record_history) {
    FlowMeasurement latest_flow;
    double latest_ts = measurement.timestamp_s;
    double latest_history_ts = 0.0;
    if (flow_history_.latest(latest_history_ts, latest_flow)) {
      latest_ts = std::max(latest_ts, latest_history_ts);
    }
    flow_history_.append(latest_ts, measurement);
  }
  if (timestamp_s_.has_value()) {
    state_history_.append(*timestamp_s_, snapshot());
  }
  return UpdateResult{true, "accepted", measurement.source, mahalanobis_distance, innovation_xy};
}

UpdateResult CascadedESKF::apply_historical_vision_update(const VisionMeasurement & measurement)
{
  if (!is_initialized_ || !timestamp_s_.has_value()) {
    return reject(measurement.source, "filter_not_initialized");
  }
  const double corrected_timestamp_s = measurement.timestamp_s;
  if (corrected_timestamp_s > *timestamp_s_ + 1e-3) {
    return reject(measurement.source, "measurement_from_future");
  }

  FilterSnapshot snapshot_at_time;
  double snapshot_time_s = 0.0;
  if (!state_history_.at_nearest(corrected_timestamp_s, snapshot_time_s, snapshot_at_time)) {
    return reject(measurement.source, "state_history_too_short");
  }
  if (snapshot_time_s > corrected_timestamp_s) {
    FilterSnapshot before_snapshot;
    double before_time_s = 0.0;
    if (state_history_.at_or_before(corrected_timestamp_s, before_time_s, before_snapshot)) {
      snapshot_time_s = before_time_s;
      snapshot_at_time = before_snapshot;
    }
  }

  const auto imu_future = imu_history_.after(snapshot_time_s);
  const auto flow_future = flow_history_.after(snapshot_time_s);
  restore_snapshot(snapshot_at_time);
  state_history_.clear_after(snapshot_time_s);
  imu_history_.clear_after(snapshot_time_s);
  flow_history_.clear_after(snapshot_time_s);

  struct ReplayEvent
  {
    double timestamp_s;
    int kind;
    ImuSample imu;
    FlowMeasurement flow;
  };
  std::vector<ReplayEvent> events;
  for (const auto & item : imu_future) {
    events.push_back(ReplayEvent{item.first, 0, item.second, FlowMeasurement{}});
  }
  for (const auto & item : flow_future) {
    events.push_back(ReplayEvent{item.first, 1, ImuSample{}, item.second});
  }
  std::sort(events.begin(), events.end(), [](const ReplayEvent & a, const ReplayEvent & b) {
      if (a.timestamp_s == b.timestamp_s) {
        return a.kind < b.kind;
      }
      return a.timestamp_s < b.timestamp_s;
    });

  for (const auto & event : events) {
    if (event.timestamp_s > corrected_timestamp_s) {
      continue;
    }
    if (event.kind == 0) {
      predict_internal(event.imu, true, true);
    } else {
      update_from_flow_internal(event.flow, true);
    }
  }

  VisionMeasurement corrected = measurement;
  corrected.timestamp_s = corrected_timestamp_s;
  UpdateResult result = update_from_vision(corrected);
  if (result.accepted) {
    time_delay_s_ = std::clamp(measurement.processing_latency_s, 0.0, config_.timing.max_time_delay_s);
  }

  for (const auto & event : events) {
    if (event.timestamp_s <= corrected_timestamp_s) {
      continue;
    }
    if (event.kind == 0) {
      predict_internal(event.imu, true, true);
    } else {
      update_from_flow_internal(event.flow, true);
    }
  }

  return result;
}

bool CascadedESKF::passes_kinematic_gate(const VisionMeasurement & measurement) const
{
  if (!last_accepted_measurement_time_s_.has_value() || !last_accepted_position_xy_m_.has_value()) {
    return true;
  }
  if (measurement.timestamp_s <= *last_accepted_measurement_time_s_) {
    return true;
  }
  const double dt = measurement.timestamp_s - *last_accepted_measurement_time_s_;
  const double max_distance = config_.gate.vmax_mps * dt + config_.gate.kinematic_tolerance_m;
  const double distance = (measurement.position_xy_enu_m - *last_accepted_position_xy_m_).norm();
  return distance <= max_distance && distance <= config_.gate.inconsistency_threshold_m;
}

void CascadedESKF::inject_error_state(const Vector16 & delta_x)
{
  position_enu_m_ += delta_x.segment<3>(0);
  velocity_enu_mps_ += delta_x.segment<3>(3);
  quaternion_wxyz_ = quaternion_normalize(quaternion_multiply(quaternion_wxyz_, quaternion_from_small_angle(delta_x.segment<3>(6))));
  accel_bias_mps2_ += delta_x.segment<3>(9);
  gyro_bias_rps_ += delta_x.segment<3>(12);
  time_delay_s_ = std::clamp(time_delay_s_ + delta_x(15), 0.0, config_.timing.max_time_delay_s);
}

void CascadedESKF::inflate_altitude_uncertainty(double target_variance_m2)
{
  P_(2, 2) = std::max(target_variance_m2, P_(2, 2));
}

void CascadedESKF::set_altitude_estimate(double z_enu_m)
{
  position_enu_m_.z() = z_enu_m;
}

void CascadedESKF::register_vision_loss()
{
  ++consecutive_vision_loss_;
}

UpdateResult CascadedESKF::reject_vision_measurement(const std::string & source, const std::string & reason)
{
  return reject(source, reason);
}

FilterSnapshot CascadedESKF::snapshot() const
{
  return FilterSnapshot{*timestamp_s_, position_enu_m_, velocity_enu_mps_, quaternion_wxyz_, accel_bias_mps2_, gyro_bias_rps_, time_delay_s_, P_};
}

void CascadedESKF::restore_snapshot(const FilterSnapshot & snapshot)
{
  timestamp_s_ = snapshot.timestamp_s;
  position_enu_m_ = snapshot.position_enu_m;
  velocity_enu_mps_ = snapshot.velocity_enu_mps;
  quaternion_wxyz_ = snapshot.quaternion_wxyz;
  accel_bias_mps2_ = snapshot.accel_bias_mps2;
  gyro_bias_rps_ = snapshot.gyro_bias_rps;
  time_delay_s_ = snapshot.time_delay_s;
  P_ = snapshot.covariance_16x16;
}

UpdateResult CascadedESKF::reject(
  const std::string & source,
  const std::string & reason,
  std::optional<double> mahalanobis_distance,
  std::optional<Eigen::VectorXd> innovation)
{
  ++consecutive_rejections_;
  if (is_lightglue_source(source)) {
    ++consecutive_lightglue_rejections_;
  }
  last_update_source_ = "rejected:" + source;
  return UpdateResult{false, reason, source, mahalanobis_distance, innovation};
}

}  // namespace advanced_localization
