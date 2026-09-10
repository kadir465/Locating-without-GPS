#include "advanced_localization_cpp/config.hpp"
#include "advanced_localization_cpp/eskf.hpp"
#include "advanced_localization_cpp/failsafe.hpp"
#include "advanced_localization_cpp/map_store.hpp"
#include "advanced_localization_cpp/math_utils.hpp"
#include "advanced_localization_cpp/optical_flow_layer.hpp"
#include "advanced_localization_cpp/roi_selector.hpp"
#include "advanced_localization_cpp/visual_localizer.hpp"

#include <px4_msgs/msg/sensor_combined.hpp>
#include <px4_msgs/msg/vehicle_visual_odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace advanced_localization
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

double px4_timestamp_to_seconds(uint64_t timestamp_us)
{
  return static_cast<double>(timestamp_us) * 1e-6;
}

uint64_t seconds_to_px4_timestamp_us(double timestamp_s)
{
  return static_cast<uint64_t>(std::max(0.0, timestamp_s) * 1e6);
}

Vector3 frd_to_estimator_body(const Vector3 & frd)
{
  return Vector3(frd.x(), -frd.y(), -frd.z());
}

cv::Mat image_to_grayscale(const sensor_msgs::msg::Image & msg)
{
  const std::string encoding = msg.encoding;
  cv::Mat raw(static_cast<int>(msg.height), static_cast<int>(msg.step), CV_8UC1, const_cast<unsigned char *>(msg.data.data()));
  if (encoding == "mono8" || encoding == "8UC1") {
    return raw(cv::Rect(0, 0, static_cast<int>(msg.width), static_cast<int>(msg.height))).clone();
  }
  if (encoding == "rgb8" || encoding == "bgr8") {
    const int channels = 3;
    const int width_from_step = static_cast<int>(msg.step) / channels;
    cv::Mat color(static_cast<int>(msg.height), width_from_step, CV_8UC3, const_cast<unsigned char *>(msg.data.data()));
    cv::Mat cropped = color(cv::Rect(0, 0, static_cast<int>(msg.width), static_cast<int>(msg.height)));
    cv::Mat gray;
    cv::cvtColor(cropped, gray, encoding == "rgb8" ? cv::COLOR_RGB2GRAY : cv::COLOR_BGR2GRAY);
    return gray;
  }
  throw std::runtime_error("unsupported camera encoding: " + encoding);
}

}  // namespace

class LocalizationNode : public rclcpp::Node
{
public:
  LocalizationNode()
  : Node("advanced_localization_cpp")
  {
    declare_parameter<std::string>("config_yaml", "");
    declare_parameter<std::string>("map_path", "");
    declare_parameter<double>("map_gsd_m_per_px", 0.2);
    declare_parameter<std::string>("camera_topic", "/camera/image_mono");
    declare_parameter<std::string>("px4_imu_topic", "/fmu/out/sensor_combined");
    declare_parameter<int>("flow_max_side_px", 1280);
    declare_parameter<double>("start_x_enu_m", 0.0);
    declare_parameter<double>("start_y_enu_m", 0.0);
    declare_parameter<double>("start_z_enu_m", 0.0);
    declare_parameter<double>("start_yaw_deg", 0.0);

    const auto config_yaml = get_parameter("config_yaml").as_string();
    config_ = SystemConfig::from_yaml(config_yaml);
    config_.flow.max_side_px = static_cast<int>(get_parameter("flow_max_side_px").as_int());
    filter_ = std::make_unique<CascadedESKF>(config_);
    failsafe_ = std::make_unique<FailsafeManager>(config_.gate.max_consecutive_rejections, config_.gate.max_position_trace_m2);
    flow_layer_ = std::make_unique<OpticalFlowLayer>(config_, config_.camera.camera_matrix);
    roi_selector_ = std::make_unique<ShiftedRoiSelector>(
      config_.lightglue.roi_base_window_m,
      1.0,
      config_.lightglue.roi_max_window_m,
      config_.lightglue.roi_padding_sigma);
    visual_localizer_ = std::make_unique<OnnxTensorRtVisualLocalizer>(config_);

    const auto map_path = get_parameter("map_path").as_string();
    if (!map_path.empty()) {
      try {
        map_ = StaticGeoMap::from_path(map_path, get_parameter("map_gsd_m_per_px").as_double(), true);
        RCLCPP_INFO(get_logger(), "Loaded static map: %s", map_path.c_str());
      } catch (const std::exception & exc) {
        RCLCPP_ERROR(get_logger(), "Failed to load map '%s': %s", map_path.c_str(), exc.what());
      }
    }

    start_position_enu_m_ = Vector3(
      get_parameter("start_x_enu_m").as_double(),
      get_parameter("start_y_enu_m").as_double(),
      get_parameter("start_z_enu_m").as_double());
    start_yaw_rad_ = get_parameter("start_yaw_deg").as_double() * kPi / 180.0;

    auto sensor_qos = rclcpp::SensorDataQoS();
    imu_sub_ = create_subscription<px4_msgs::msg::SensorCombined>(
      get_parameter("px4_imu_topic").as_string(),
      sensor_qos,
      std::bind(&LocalizationNode::on_imu, this, std::placeholders::_1));
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      get_parameter("camera_topic").as_string(),
      sensor_qos,
      std::bind(&LocalizationNode::on_image, this, std::placeholders::_1));

    odom_pub_ = create_publisher<px4_msgs::msg::VehicleVisualOdometry>("/fmu/in/vehicle_visual_odometry", 10);
    status_pub_ = create_publisher<std_msgs::msg::String>("/advanced_localization/status", 10);

    using namespace std::chrono_literals;
    const auto flow_period = std::chrono::duration<double>(1.0 / std::max(config_.flow.rate_hz, 1.0));
    flow_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(flow_period),
      std::bind(&LocalizationNode::on_flow_timer, this));
    vision_timer_ = create_wall_timer(250ms, std::bind(&LocalizationNode::on_vision_timer, this));

    RCLCPP_INFO(get_logger(), "PX4 uXRCE-DDS state-estimate-only localization node initialized");
  }

private:
  void on_imu(const px4_msgs::msg::SensorCombined::SharedPtr msg)
  {
    const double timestamp_s = px4_timestamp_to_seconds(msg->timestamp);
    if (!filter_->is_initialized()) {
      filter_->initialize(start_position_enu_m_, start_yaw_rad_, timestamp_s);
    }

    const Vector3 accel_frd(msg->accelerometer_m_s2[0], msg->accelerometer_m_s2[1], msg->accelerometer_m_s2[2]);
    const Vector3 gyro_frd(msg->gyro_rad[0], msg->gyro_rad[1], msg->gyro_rad[2]);
    const ImuSample sample{timestamp_s, frd_to_estimator_body(accel_frd), frd_to_estimator_body(gyro_frd)};
    filter_->predict_from_imu(sample);
    publish_odometry(timestamp_s);
    publish_status(std::nullopt);
  }

  void on_image(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try {
      cv::Mat gray = image_to_grayscale(*msg);
      std::lock_guard<std::mutex> lock(frame_mutex_);
      latest_frame_ = gray;
      latest_frame_timestamp_s_ = static_cast<double>(msg->header.stamp.sec) + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
    } catch (const std::exception & exc) {
      RCLCPP_WARN(get_logger(), "%s", exc.what());
    }
  }

  void on_flow_timer()
  {
    if (!filter_->is_initialized()) {
      return;
    }
    auto frame = latest_frame();
    if (!frame.has_value()) {
      return;
    }
    const double altitude_m = std::max(std::abs(filter_->position_enu_m().z()), 1.0);
    auto measurement = flow_layer_->run(frame->first, frame->second, altitude_m, filter_->yaw_rad());
    if (!measurement.has_value()) {
      return;
    }
    latest_flow_velocity_xy_mps_ = measurement->velocity_xy_mps;
    UpdateResult result = filter_->update_from_flow(*measurement);
    publish_status(result);
  }

  void on_vision_timer()
  {
    if (!filter_->is_initialized() || !map_.valid() || !visual_localizer_->is_ready()) {
      return;
    }
    auto frame = latest_frame();
    if (!frame.has_value()) {
      return;
    }
    Eigen::MatrixXd covariance_xy(2, 1);
    covariance_xy(0, 0) = filter_->position_covariance_xy_m2().x();
    covariance_xy(1, 0) = filter_->position_covariance_xy_m2().y();
    RoiWindow roi = roi_selector_->select(
      filter_->position_enu_m().head<2>(),
      latest_flow_velocity_xy_mps_,
      Vector2::Zero(),
      covariance_xy,
      std::max(config_.lightglue.roi_prediction_latency_s, config_.timing.lightglue_budget_ms * 1e-3));

    const double altitude_m = std::max(std::abs(filter_->position_enu_m().z()), 1.0);
    const double focal_px = std::max(config_.camera.camera_matrix(0, 0), 1e-6);
    const double camera_gsd_m_per_px = altitude_m / focal_px;
    const double camera_footprint_m = camera_gsd_m_per_px * static_cast<double>(config_.lightglue.image_size_px);
    const cv::Mat map_patch = map_.crop_from_enu(roi.center_xy_enu_m, camera_footprint_m, config_.lightglue.image_size_px, camera_footprint_m, camera_footprint_m);
    const cv::Mat camera_patch = center_crop(frame->first, config_.lightglue.image_size_px);

    auto measurement = visual_localizer_->run(
      map_patch,
      camera_patch,
      roi.center_xy_enu_m,
      camera_gsd_m_per_px,
      filter_->yaw_rad(),
      frame->second,
      camera_gsd_m_per_px);
    if (!measurement.has_value()) {
      filter_->register_vision_loss();
      publish_status(std::nullopt);
      return;
    }

    UpdateResult result = filter_->apply_historical_vision_update(*measurement);
    auto decision = failsafe_->evaluate(filter_->health(), result);
    last_failsafe_code_ = decision.code;
    publish_status(result);
  }

  std::optional<std::pair<cv::Mat, double>> latest_frame() const
  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (latest_frame_.empty() || !latest_frame_timestamp_s_.has_value()) {
      return std::nullopt;
    }
    return std::make_pair(latest_frame_.clone(), *latest_frame_timestamp_s_);
  }

  static cv::Mat center_crop(const cv::Mat & image, int output_size_px)
  {
    cv::Mat padded = image;
    const int pad_y = std::max(0, output_size_px - padded.rows);
    const int pad_x = std::max(0, output_size_px - padded.cols);
    if (pad_y > 0 || pad_x > 0) {
      cv::copyMakeBorder(padded, padded, pad_y / 2, pad_y - pad_y / 2, pad_x / 2, pad_x - pad_x / 2, cv::BORDER_REFLECT101);
    }
    const int x0 = std::max(0, (padded.cols - output_size_px) / 2);
    const int y0 = std::max(0, (padded.rows - output_size_px) / 2);
    return padded(cv::Rect(x0, y0, output_size_px, output_size_px)).clone();
  }

  void publish_odometry(double timestamp_s)
  {
    const Vector3 position_ned = enu_to_ned_position(filter_->position_enu_m());
    const Vector3 velocity_ned = enu_to_ned_position(filter_->velocity_enu_mps());
    const Vector4 q_ned = enu_quaternion_to_ned(filter_->quaternion_wxyz());

    px4_msgs::msg::VehicleVisualOdometry msg{};
    msg.timestamp = seconds_to_px4_timestamp_us(timestamp_s);
    msg.timestamp_sample = msg.timestamp;
    msg.pose_frame = px4_msgs::msg::VehicleVisualOdometry::POSE_FRAME_NED;
    msg.velocity_frame = px4_msgs::msg::VehicleVisualOdometry::VELOCITY_FRAME_NED;
    msg.position = {static_cast<float>(position_ned.x()), static_cast<float>(position_ned.y()), static_cast<float>(position_ned.z())};
    msg.q = {static_cast<float>(q_ned(0)), static_cast<float>(q_ned(1)), static_cast<float>(q_ned(2)), static_cast<float>(q_ned(3))};
    msg.velocity = {static_cast<float>(velocity_ned.x()), static_cast<float>(velocity_ned.y()), static_cast<float>(velocity_ned.z())};
    msg.angular_velocity = {0.0f, 0.0f, 0.0f};
    const Matrix16 P = filter_->covariance_16x16();
    msg.position_variance = {static_cast<float>(P(1, 1)), static_cast<float>(P(0, 0)), static_cast<float>(P(2, 2))};
    msg.orientation_variance = {static_cast<float>(P(6, 6)), static_cast<float>(P(7, 7)), static_cast<float>(P(8, 8))};
    msg.velocity_variance = {static_cast<float>(P(4, 4)), static_cast<float>(P(3, 3)), static_cast<float>(P(5, 5))};
    msg.quality = 0;
    msg.reset_counter = 0;
    odom_pub_->publish(msg);
  }

  void publish_status(const std::optional<UpdateResult> & latest_update)
  {
    const auto health = filter_->health();
    std::ostringstream stream;
    stream << "mode=" << failsafe_->mode()
           << ";failsafe_code=" << to_string(last_failsafe_code_)
           << ";rej_all=" << health.consecutive_rejections
           << ";rej_lightglue=" << health.consecutive_lightglue_rejections
           << ";vision_loss=" << health.consecutive_vision_loss
           << ";trace_xy=" << health.position_trace_xy_m2
           << ";trace_v=" << health.velocity_trace_m2ps2
           << ";pzz=" << health.pzz_m2
           << ";last_source=" << health.last_update_source
           << ";update=" << (latest_update.has_value() ? latest_update->reason : "predict_only");
    std_msgs::msg::String msg;
    msg.data = stream.str();
    status_pub_->publish(msg);
  }

  SystemConfig config_;
  std::unique_ptr<CascadedESKF> filter_;
  std::unique_ptr<FailsafeManager> failsafe_;
  std::unique_ptr<OpticalFlowLayer> flow_layer_;
  std::unique_ptr<ShiftedRoiSelector> roi_selector_;
  std::unique_ptr<VisualLocalizer> visual_localizer_;
  StaticGeoMap map_;
  Vector3 start_position_enu_m_{Vector3::Zero()};
  double start_yaw_rad_{0.0};
  Vector2 latest_flow_velocity_xy_mps_{Vector2::Zero()};
  FailsafeCode last_failsafe_code_{FailsafeCode::None};

  mutable std::mutex frame_mutex_;
  cv::Mat latest_frame_;
  std::optional<double> latest_frame_timestamp_s_;

  rclcpp::Subscription<px4_msgs::msg::SensorCombined>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<px4_msgs::msg::VehicleVisualOdometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::TimerBase::SharedPtr flow_timer_;
  rclcpp::TimerBase::SharedPtr vision_timer_;
};

}  // namespace advanced_localization

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<advanced_localization::LocalizationNode>());
  rclcpp::shutdown();
  return 0;
}
