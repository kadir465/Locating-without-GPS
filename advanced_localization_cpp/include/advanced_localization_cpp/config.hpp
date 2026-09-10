#pragma once

#include "advanced_localization_cpp/types.hpp"

#include <string>

namespace advanced_localization
{

struct NoiseConfig
{
  double accel_noise_std_mps2{0.35};
  double gyro_noise_std_rps{0.03};
  double accel_bias_rw_std_mps2{0.01};
  double gyro_bias_rw_std_rps{0.0015};
};

struct GateConfig
{
  double chi2_threshold_3dof{9.0};
  double chi2_threshold_2dof{5.99};
  double max_position_correction_m{3.0};
  double max_yaw_correction_deg{5.0};
  double vmax_mps{15.0};
  double kinematic_tolerance_m{5.0};
  int max_consecutive_rejections{5};
  int max_consecutive_vision_loss{8};
  double max_position_trace_m2{100.0};
  double min_measurement_variance_m2{0.05};
  double inconsistency_threshold_m{25.0};
};

struct TimingConfig
{
  double lightglue_budget_ms{80.0};
  double total_vision_budget_ms{500.0};
  double history_buffer_seconds{0.5};
  double initial_time_delay_s{0.0};
  double max_time_delay_s{0.5};
};

struct FlowConfig
{
  double rate_hz{10.0};
  double chi2_threshold_2dof{5.99};
  double pzz_reject_threshold_m2{16.0};
  double noise_std_mps{0.8};
  int min_features{40};
  int max_side_px{1280};
  double max_bidirectional_error_px{1.5};
  double reference_altitude_m{30.0};
};

struct LightGlueConfig
{
  bool enabled{false};
  bool require_cuda{true};
  std::string device{"auto"};
  int image_size_px{256};
  bool use_amp{true};
  int min_matches{10};
  int min_inliers{6};
  double min_confidence{0.35};
  double min_inlier_ratio{0.30};
  double ransac_reproj_threshold_px{3.0};
  double filter_threshold{0.10};
  double depth_confidence{0.95};
  double width_confidence{0.99};
  double roi_base_window_m{20.0};
  double roi_prediction_latency_s{0.20};
  double roi_max_window_m{80.0};
  double roi_padding_sigma{3.0};
  double residual_gate_m{25.0};
  double scale_consistency_min{0.50};
  double scale_consistency_max{1.50};
  double scale_consistency_min_distance_m{1.0};
  double trust_gain_min{0.05};
  double trust_gain_max{0.70};
  std::string superpoint_onnx_path;
  std::string lightglue_onnx_path;
};

struct ObservabilityConfig
{
  double velocity_trace_limit_m2ps2{25.0};
  bool trigger_once{true};
  int command_id{31000};
};

struct CameraCalibrationConfig
{
  std::string info_yaml_path{"config/camera_info_dummy.yaml"};
  int image_width{640};
  int image_height{480};
  Matrix3 camera_matrix{(Matrix3() << 500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0).finished()};
  Eigen::VectorXd distortion_coeffs{Eigen::VectorXd::Zero(5)};
};

struct SystemConfig
{
  double imu_rate_hz{100.0};
  Vector3 gravity_enu_mps2{0.0, 0.0, -9.80665};
  NoiseConfig noise;
  GateConfig gate;
  TimingConfig timing;
  FlowConfig flow;
  LightGlueConfig lightglue;
  ObservabilityConfig observability;
  CameraCalibrationConfig camera;

  static SystemConfig from_yaml(const std::string & yaml_path);
};

}  // namespace advanced_localization
