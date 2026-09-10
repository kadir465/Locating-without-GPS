#pragma once

#include "advanced_localization_cpp/config.hpp"
#include "advanced_localization_cpp/types.hpp"
#include "advanced_localization_cpp/visual_localizer.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace advanced_localization
{

constexpr double kVideoTestEarthRadiusM = 6378137.0;
constexpr double kFixedVideoTestAltitudeM = 100.0;

struct GroundTruthSample
{
  double timestamp_s{0.0};
  double latitude_deg{0.0};
  double longitude_deg{0.0};
  std::optional<double> heading_deg;
  std::optional<double> rel_alt_m;
};

struct Frame
{
  cv::Mat image;
  int index{0};
  double timestamp_s{0.0};
};

struct FrameSourceMetadata
{
  std::string backend{"unknown"};
  std::string codec{"unknown"};
  int width{0};
  int height{0};
  double fps{0.0};
  int frame_count{0};
};

struct FrameSourceProfile
{
  std::string backend{"unknown"};
  std::string codec{"unknown"};
  int width{0};
  int height{0};
  double fps{0.0};
  int frames_read{0};
  int dropped_frames{0};
  double decode_ms{0.0};
  double wait_ms{0.0};
  double queue_fill_mean{0.0};
  double decode_fps{0.0};
};

class FrameSource
{
public:
  virtual ~FrameSource() = default;
  virtual void open() = 0;
  virtual bool read(Frame & frame) = 0;
  virtual FrameSourceMetadata metadata() const = 0;
  virtual FrameSourceProfile profile() const = 0;
};

struct VideoTestResult
{
  int frame_count{0};
  int correction_count{0};
  double runtime_s{0.0};
  double processed_fps{0.0};
  double rmse_m{0.0};
  double mae_m{0.0};
  double cep95_m{0.0};
  double max_error_m{0.0};
  double mean_klt_ms{0.0};
  double mean_visual_ms{0.0};
  std::vector<double> timestamps_s;
  std::vector<double> error_m;
  struct Profile
  {
    bool enabled{false};
    int frames_read{0};
    int klt_updates{0};
    int correction_jobs{0};
    int visual_results{0};
    int csv_rows{0};
    double decode_read_ms{0.0};
    double decode_wait_ms{0.0};
    double gt_interpolation_ms{0.0};
    double klt_total_ms{0.0};
    double klt_grayscale_ms{0.0};
    double klt_resize_ms{0.0};
    double klt_clahe_ms{0.0};
    double klt_feature_ms{0.0};
    double klt_lk_ms{0.0};
    double klt_stats_ms{0.0};
    double correction_crop_ms{0.0};
    double visual_search_ms{0.0};
    double controller_ms{0.0};
    double csv_ms{0.0};
    double map_render_ms{0.0};
    FrameSourceProfile frame_source;
  } profile;
};

struct FlowSignConfig
{
  bool swap{true};
  int sx{1};
  int sy{1};
};

struct FlowStepDiagnostics
{
  bool valid{false};
  double du_px{0.0};
  double dv_px{0.0};
  double flow_std_px{0.0};
  double flow_body_x_m{0.0};
  double flow_body_y_m{0.0};
  double flow_enu_x_m{0.0};
  double flow_enu_y_m{0.0};
  double heading_used_rad{0.0};
  bool zupt_applied{false};
  double learned_scale{1.0};
  std::string reject_reason;
};

struct LightGlueSearchResult
{
  double capture_timestamp_s{0.0};
  Vector2 capture_estimate_xy_enu_m{Vector2::Zero()};
  Vector2 capture_flow_odom_xy_enu_m{Vector2::Zero()};
  double heading_seed_rad{0.0};
  std::optional<VisionMeasurement> measurement;
  int evaluated_hypotheses{0};
  double wall_time_s{0.0};
  std::string debug_summary;
};

struct VideoTestingOptions
{
  std::string video_path;
  std::string srt_path;
  std::string lightglue_config_yaml{"advanced_localization_cpp/config/system_config.yaml"};
  std::string csv_output_path;
  std::string frame_source{"opencv"};
  std::string gst_pipeline;
  std::string benchmark_preset;
  std::string map_path;
  double camera_fov_deg{84.0};
  double test_altitude_m{kFixedVideoTestAltitudeM};
  double correction_interval_s{1.0};
  double correction_noise_std_m{0.2};
  bool enable_correction{true};
  bool verbose{false};
  int random_seed{7};
  double lightglue_window_size_m{60.0};
  int lightglue_map_roi_size_px{500};
  std::optional<int> lightglue_patch_size_px;
  std::optional<double> lightglue_confidence_gate;
  double lightglue_inlier_gate{0.30};
  double lightglue_jump_gate_m{50.0};
  double lightglue_residual_gate_m{25.0};
  double lightglue_meas_scale_min{0.5};
  double lightglue_meas_scale_max{1.5};
  double lightglue_blend_gain_min{0.05};
  double lightglue_blend_gain_max{0.70};
  double lightglue_mahalanobis_gate{9.21};
  int lightglue_max_yaw_hypotheses{5};
  int lightglue_hypotheses_per_job{1};
  double lightglue_gpu_target_duty_cycle{0.35};
  double lightglue_gpu_min_interval_s{0.35};
  double lightglue_gpu_max_interval_s{5.0};
  double lightglue_max_result_age_s{3.5};
  bool enable_srt_scale_assist{true};
  double srt_scale_assist_alpha{0.20};
  bool use_gt_heading{true};
  bool force_srt_direction_lock{true};
  double vio_rate_hz{10.0};
  std::optional<int> max_frames;
  double map_gsd_m_per_px{0.2};
  bool show_map_window{false};
  std::string map_window_name{"Map Point&Trail"};
  int map_max_display_size_px{1000};
  std::string map_window_video_output_path;
  std::optional<double> map_window_video_fps;
  bool use_clahe{true};
  bool enable_async_lightglue{true};
  bool enable_async_decode{true};
  int async_decode_queue_size{4};
  int flow_max_side_px{1280};
  bool decode_only{false};
  bool klt_only{false};
  bool csv_enabled{true};
  double image_sequence_fps{30.0};
  int raw_width_px{0};
  int raw_height_px{0};
  double raw_fps{0.0};
  bool profile{false};
  std::string profile_output_path;
  std::string realtime_profile_output_path;
};

std::unique_ptr<FrameSource> make_video_test_frame_source(const VideoTestingOptions & options);

std::vector<GroundTruthSample> parse_dji_srt(const std::string & srt_path);
Vector2 latlon_to_xy(double latitude_deg, double longitude_deg, double latitude_ref_deg, double longitude_ref_deg);
double wrap_angle_rad(double angle_rad);
double north_cw_heading_deg_to_math_rad(double heading_deg);
std::pair<Vector2, Vector2> flow_to_enu(
  double du_px,
  double dv_px,
  double heading_rad,
  double scale_m_per_px,
  const FlowSignConfig & sign_cfg,
  bool heading_is_north_cw = false);
Vector2 project_xy_onto_direction(const Vector2 & vector_xy, const Vector2 & direction_xy);
Vector2 project_xy_forward_onto_direction(const Vector2 & vector_xy, const Vector2 & direction_xy);
cv::Mat center_crop_square(const cv::Mat & frame_bgr, int output_size_px);
void center_crop_square(const cv::Mat & frame_bgr, int output_size_px, cv::Mat & output);

class VisionOnlyTracker
{
public:
  explicit VisionOnlyTracker(
    double focal_length_px,
    double test_altitude_m = kFixedVideoTestAltitudeM,
    int max_features = 200,
    int min_features = 12,
    bool use_clahe = true,
    double scale_min = 0.60,
    double scale_max = 0.95,
    int flow_max_side_px = 0);

  double meters_per_pixel() const;
  std::tuple<bool, double, double, double> estimate_flow_step(const cv::Mat & frame_bgr);
  struct FlowTiming
  {
    double total_ms{0.0};
    double grayscale_ms{0.0};
    double resize_ms{0.0};
    double clahe_ms{0.0};
    double feature_ms{0.0};
    double lk_ms{0.0};
    double stats_ms{0.0};
  };
  void set_timing_enabled(bool enabled) { timing_enabled_ = enabled; }
  const FlowTiming & last_timing_ms() const { return last_timing_ms_; }
  std::tuple<double, double, FlowStepDiagnostics> integrate_flow_step(
    double du_px,
    double dv_px,
    double flow_std_px,
    double frame_dt_s,
    double heading_rad,
    const FlowSignConfig & sign_cfg,
    bool heading_is_north_cw = false);
  std::optional<double> update_scale_from_lightglue(
    const Vector2 & ai_true_xy,
    const Vector2 & saved_est_xy,
    std::optional<Vector2> saved_flow_odom_xy = std::nullopt,
    bool allow_update = true,
    double alpha = 0.15,
    double max_update_delta = 0.025);
  double update_scale_from_measurement(
    double scale_measurement,
    std::optional<double> alpha = std::nullopt,
    std::optional<double> scale_min = std::nullopt,
    std::optional<double> scale_max = std::nullopt,
    int history_size = 5,
    std::optional<double> max_update_delta = std::nullopt);
  void set_absolute_position(double x_m, double y_m);
  void inflate_position_covariance(double process_noise_m2);
  void contract_position_covariance(double gain);

  double pos_x_m{0.0};
  double pos_y_m{0.0};
  double altitude_m{kFixedVideoTestAltitudeM};
  double focal_length_px{1.0};
  double base_meters_per_pixel{1.0};
  double learned_scale{0.725};
  double pos_cov_xx_m2{4.0};
  double pos_cov_yy_m2{4.0};

private:
  double scale_min_{0.60};
  double scale_max_{0.95};
  std::vector<double> scale_history_;
  std::optional<double> prev_ai_x_;
  std::optional<double> prev_ai_y_;
  std::optional<double> prev_saved_est_x_;
  std::optional<double> prev_saved_est_y_;
  int max_features_{200};
  int min_features_{12};
  bool use_clahe_{true};
  int flow_max_side_px_{0};
  bool timing_enabled_{false};
  FlowTiming last_timing_ms_;
  cv::Mat prev_gray_;
  cv::Mat gray_buffer_;
  cv::Mat flow_gray_buffer_;
  cv::Mat gray_proc_buffer_;
  std::vector<cv::Point2f> prev_points_;
  std::vector<cv::Point2f> curr_points_;
  std::vector<cv::Point2f> filtered_points_;
  std::vector<unsigned char> status_buffer_;
  std::vector<float> error_buffer_;
  std::vector<double> dx_buffer_;
  std::vector<double> dy_buffer_;
  cv::Ptr<cv::CLAHE> clahe_;
  cv::Size lk_win_size_{21, 21};
  int lk_max_level_{3};
  cv::TermCriteria lk_criteria_{cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01};
};

VideoTestResult run_vision_only_video_test(
  const VideoTestingOptions & options,
  std::unique_ptr<VisualLocalizer> visual_localizer = nullptr);

}  // namespace advanced_localization
