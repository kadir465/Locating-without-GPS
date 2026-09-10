#include "advanced_localization_cpp/config.hpp"

#include <yaml-cpp/yaml.h>

#include <filesystem>

namespace advanced_localization
{
namespace
{

template<typename T>
void assign_if_present(const YAML::Node & node, const char * key, T & target)
{
  if (node && node[key]) {
    target = node[key].as<T>();
  }
}

Vector3 vector3_from_yaml(const YAML::Node & node, const Vector3 & fallback)
{
  if (!node || !node.IsSequence() || node.size() != 3) {
    return fallback;
  }
  return Vector3(node[0].as<double>(), node[1].as<double>(), node[2].as<double>());
}

Matrix3 matrix3_from_yaml(const YAML::Node & node, const Matrix3 & fallback)
{
  if (!node || !node.IsSequence() || node.size() != 3) {
    return fallback;
  }
  Matrix3 matrix;
  for (int r = 0; r < 3; ++r) {
    if (!node[r].IsSequence() || node[r].size() != 3) {
      return fallback;
    }
    for (int c = 0; c < 3; ++c) {
      matrix(r, c) = node[r][c].as<double>();
    }
  }
  return matrix;
}

Eigen::VectorXd vector_from_yaml(const YAML::Node & node, const Eigen::VectorXd & fallback)
{
  if (!node || !node.IsSequence()) {
    return fallback;
  }
  Eigen::VectorXd values(static_cast<Eigen::Index>(node.size()));
  for (std::size_t i = 0; i < node.size(); ++i) {
    values(static_cast<Eigen::Index>(i)) = node[i].as<double>();
  }
  return values;
}

}  // namespace

SystemConfig SystemConfig::from_yaml(const std::string & yaml_path)
{
  SystemConfig config;
  if (yaml_path.empty() || !std::filesystem::exists(yaml_path)) {
    return config;
  }

  const YAML::Node root = YAML::LoadFile(yaml_path);
  assign_if_present(root, "imu_rate_hz", config.imu_rate_hz);
  config.gravity_enu_mps2 = vector3_from_yaml(root["gravity_enu_mps2"], config.gravity_enu_mps2);

  const auto noise = root["noise"];
  assign_if_present(noise, "accel_noise_std_mps2", config.noise.accel_noise_std_mps2);
  assign_if_present(noise, "gyro_noise_std_rps", config.noise.gyro_noise_std_rps);
  assign_if_present(noise, "accel_bias_rw_std_mps2", config.noise.accel_bias_rw_std_mps2);
  assign_if_present(noise, "gyro_bias_rw_std_rps", config.noise.gyro_bias_rw_std_rps);

  const auto gate = root["gate"];
  assign_if_present(gate, "chi2_threshold_3dof", config.gate.chi2_threshold_3dof);
  assign_if_present(gate, "chi2_threshold_2dof", config.gate.chi2_threshold_2dof);
  assign_if_present(gate, "max_position_correction_m", config.gate.max_position_correction_m);
  assign_if_present(gate, "max_yaw_correction_deg", config.gate.max_yaw_correction_deg);
  assign_if_present(gate, "vmax_mps", config.gate.vmax_mps);
  assign_if_present(gate, "kinematic_tolerance_m", config.gate.kinematic_tolerance_m);
  assign_if_present(gate, "max_consecutive_rejections", config.gate.max_consecutive_rejections);
  assign_if_present(gate, "max_consecutive_vision_loss", config.gate.max_consecutive_vision_loss);
  assign_if_present(gate, "max_position_trace_m2", config.gate.max_position_trace_m2);
  assign_if_present(gate, "min_measurement_variance_m2", config.gate.min_measurement_variance_m2);
  assign_if_present(gate, "inconsistency_threshold_m", config.gate.inconsistency_threshold_m);

  const auto timing = root["timing"];
  assign_if_present(timing, "lightglue_budget_ms", config.timing.lightglue_budget_ms);
  assign_if_present(timing, "total_vision_budget_ms", config.timing.total_vision_budget_ms);
  assign_if_present(timing, "history_buffer_seconds", config.timing.history_buffer_seconds);
  assign_if_present(timing, "initial_time_delay_s", config.timing.initial_time_delay_s);
  assign_if_present(timing, "max_time_delay_s", config.timing.max_time_delay_s);

  const auto flow = root["flow"];
  assign_if_present(flow, "rate_hz", config.flow.rate_hz);
  assign_if_present(flow, "chi2_threshold_2dof", config.flow.chi2_threshold_2dof);
  assign_if_present(flow, "pzz_reject_threshold_m2", config.flow.pzz_reject_threshold_m2);
  assign_if_present(flow, "noise_std_mps", config.flow.noise_std_mps);
  assign_if_present(flow, "min_features", config.flow.min_features);
  assign_if_present(flow, "max_side_px", config.flow.max_side_px);
  assign_if_present(flow, "max_bidirectional_error_px", config.flow.max_bidirectional_error_px);
  assign_if_present(flow, "reference_altitude_m", config.flow.reference_altitude_m);

  const auto lightglue = root["lightglue"];
  assign_if_present(lightglue, "enabled", config.lightglue.enabled);
  assign_if_present(lightglue, "require_cuda", config.lightglue.require_cuda);
  assign_if_present(lightglue, "device", config.lightglue.device);
  assign_if_present(lightglue, "image_size_px", config.lightglue.image_size_px);
  assign_if_present(lightglue, "use_amp", config.lightglue.use_amp);
  assign_if_present(lightglue, "min_matches", config.lightglue.min_matches);
  assign_if_present(lightglue, "min_inliers", config.lightglue.min_inliers);
  assign_if_present(lightglue, "min_confidence", config.lightglue.min_confidence);
  assign_if_present(lightglue, "min_inlier_ratio", config.lightglue.min_inlier_ratio);
  assign_if_present(lightglue, "ransac_reproj_threshold_px", config.lightglue.ransac_reproj_threshold_px);
  assign_if_present(lightglue, "roi_base_window_m", config.lightglue.roi_base_window_m);
  assign_if_present(lightglue, "roi_prediction_latency_s", config.lightglue.roi_prediction_latency_s);
  assign_if_present(lightglue, "roi_max_window_m", config.lightglue.roi_max_window_m);
  assign_if_present(lightglue, "roi_padding_sigma", config.lightglue.roi_padding_sigma);
  assign_if_present(lightglue, "residual_gate_m", config.lightglue.residual_gate_m);
  assign_if_present(lightglue, "scale_consistency_min", config.lightglue.scale_consistency_min);
  assign_if_present(lightglue, "scale_consistency_max", config.lightglue.scale_consistency_max);
  assign_if_present(lightglue, "scale_consistency_min_distance_m", config.lightglue.scale_consistency_min_distance_m);
  assign_if_present(lightglue, "trust_gain_min", config.lightglue.trust_gain_min);
  assign_if_present(lightglue, "trust_gain_max", config.lightglue.trust_gain_max);
  assign_if_present(lightglue, "superpoint_onnx_path", config.lightglue.superpoint_onnx_path);
  assign_if_present(lightglue, "lightglue_onnx_path", config.lightglue.lightglue_onnx_path);

  const auto observability = root["observability"];
  assign_if_present(observability, "velocity_trace_limit_m2ps2", config.observability.velocity_trace_limit_m2ps2);
  assign_if_present(observability, "trigger_once", config.observability.trigger_once);
  assign_if_present(observability, "command_id", config.observability.command_id);

  const auto camera = root["camera"];
  assign_if_present(camera, "info_yaml_path", config.camera.info_yaml_path);
  assign_if_present(camera, "image_width", config.camera.image_width);
  assign_if_present(camera, "image_height", config.camera.image_height);
  config.camera.camera_matrix = matrix3_from_yaml(camera["camera_matrix"], config.camera.camera_matrix);
  config.camera.distortion_coeffs = vector_from_yaml(camera["distortion_coeffs"], config.camera.distortion_coeffs);

  return config;
}

}  // namespace advanced_localization
