#include "advanced_localization_cpp/video_testing.hpp"

#include "advanced_localization_cpp/map_store.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

#ifdef ADVANCED_LOCALIZATION_HAS_GSTREAMER
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#endif

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cctype>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace advanced_localization
{
namespace
{

constexpr double kEpsilon = 1e-9;
constexpr double kDefaultMaxFlowStdPx = 260.0;
constexpr double kDefaultMaxFlowShiftPx = 140.0;
constexpr double kDefaultMaxFlowStepM = 4.0;
constexpr int kDefaultHeadingSmoothWindow = 15;
constexpr double kDefaultHeadingMinDisplacementM = 0.35;
constexpr double kPi = 3.14159265358979323846;

struct GroundTruthTrack
{
  std::vector<double> times_s;
  std::vector<double> xs_m;
  std::vector<double> ys_m;
  std::vector<double> headings_rad;
  std::vector<int> heading_from_srt;
  std::vector<double> sin_heading;
  std::vector<double> cos_heading;
};

struct GroundTruthState
{
  double x_m{0.0};
  double y_m{0.0};
  double heading_rad{0.0};
  Vector2 direction{1.0, 0.0};
  bool heading_from_srt{false};
  int sample_index{0};
};

struct KinematicRoi
{
  Vector2 center_xy{Vector2::Zero()};
  double width_m{0.0};
  double height_m{0.0};
};

struct MetricsRow
{
  double time_s{0.0};
  double est_x_m{0.0};
  double est_y_m{0.0};
  double gt_x_m{0.0};
  double gt_y_m{0.0};
  double error_m{0.0};
  double heading_gt_deg{0.0};
  int heading_gt_from_srt{0};
  double heading_est_deg{0.0};
  double heading_used_deg{0.0};
  double delta_heading_deg{0.0};
  double dir_error_deg{std::numeric_limits<double>::quiet_NaN()};
  double flow_du_px{0.0};
  double flow_dv_px{0.0};
  double flow_std_px{0.0};
  double flow_body_x_m{0.0};
  double flow_body_y_m{0.0};
  double flow_enu_x_m{0.0};
  double flow_enu_y_m{0.0};
  int flow_zupt{0};
  double learned_scale{1.0};
  int sign_swap{1};
  int sign_sx{1};
  int sign_sy{1};
  std::string lightglue_event;
  int lightglue_capture_frame{-1};
  int lightglue_match_count{0};
  int lightglue_inlier_count{0};
  double lightglue_confidence{std::numeric_limits<double>::quiet_NaN()};
  double lightglue_residual_m{std::numeric_limits<double>::quiet_NaN()};
  double lightglue_apply_x_m{0.0};
  double lightglue_apply_y_m{0.0};
  double lightglue_error_before_m{std::numeric_limits<double>::quiet_NaN()};
  double lightglue_error_after_m{std::numeric_limits<double>::quiet_NaN()};
  std::string lightglue_reject_reason;
};

using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point & start)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void write_profile_report(const std::string & path, const VideoTestResult::Profile & profile)
{
  if (path.empty()) {
    return;
  }
  std::filesystem::path out_path(path);
  if (out_path.has_parent_path()) {
    std::filesystem::create_directories(out_path.parent_path());
  }
  std::ofstream file(out_path);
  if (!file) {
    throw std::runtime_error("Failed to open profile output: " + path);
  }
  file << "stage,total_ms,count,mean_ms\n";
  const auto write = [&file](const std::string & stage, double total_ms, int count) {
    file << stage << ',' << std::setprecision(17) << total_ms << ',' << count << ','
         << (count > 0 ? total_ms / static_cast<double>(count) : 0.0) << '\n';
  };
  write("decode_read", profile.decode_read_ms, profile.frames_read);
  write("decode_wait", profile.decode_wait_ms, profile.frames_read);
  write("gt_interpolation", profile.gt_interpolation_ms, profile.frames_read);
  write("klt_total", profile.klt_total_ms, profile.klt_updates);
  write("klt_grayscale", profile.klt_grayscale_ms, profile.klt_updates);
  write("klt_resize", profile.klt_resize_ms, profile.klt_updates);
  write("klt_clahe", profile.klt_clahe_ms, profile.klt_updates);
  write("klt_feature", profile.klt_feature_ms, profile.klt_updates);
  write("klt_lk", profile.klt_lk_ms, profile.klt_updates);
  write("klt_stats", profile.klt_stats_ms, profile.klt_updates);
  write("correction_crop", profile.correction_crop_ms, profile.correction_jobs);
  write("visual_search", profile.visual_search_ms, profile.visual_results);
  write("controller", profile.controller_ms, profile.frames_read);
  write("csv", profile.csv_ms, profile.csv_rows);
  write("map_render", profile.map_render_ms, profile.frames_read);
}

void write_realtime_profile_report(const std::string & path, const VideoTestResult & result)
{
  if (path.empty()) {
    return;
  }
  std::filesystem::path out_path(path);
  if (out_path.has_parent_path()) {
    std::filesystem::create_directories(out_path.parent_path());
  }
  std::ofstream file(out_path);
  if (!file) {
    throw std::runtime_error("Failed to open realtime profile output: " + path);
  }
  const FrameSourceProfile & source = result.profile.frame_source;
  file << "metric,value\n";
  const auto write = [&file](const std::string & metric, const auto & value) {
    file << metric << ',' << value << '\n';
  };
  const auto write_double = [&file](const std::string & metric, double value) {
    file << metric << ',' << std::setprecision(17) << value << '\n';
  };
  write("frames", result.frame_count);
  write_double("runtime_s", result.runtime_s);
  write_double("processed_fps", result.processed_fps);
  write_double("mean_klt_ms", result.mean_klt_ms);
  write_double("mean_visual_ms", result.mean_visual_ms);
  write("source_backend", source.backend);
  write("source_codec", source.codec);
  write("source_width", source.width);
  write("source_height", source.height);
  write_double("source_fps", source.fps);
  write_double("source_decode_ms", source.decode_ms);
  write_double("source_wait_ms", source.wait_ms);
  write_double("source_decode_fps", source.decode_fps);
  write_double("source_queue_fill_mean", source.queue_fill_mean);
  write("source_dropped_frames", source.dropped_frames);
}

std::string fourcc_to_string(double raw_fourcc)
{
  const int code = static_cast<int>(raw_fourcc);
  std::string text;
  for (int i = 0; i < 4; ++i) {
    const char ch = static_cast<char>((code >> (8 * i)) & 0xff);
    if (std::isprint(static_cast<unsigned char>(ch))) {
      text.push_back(ch);
    }
  }
  return text.empty() ? "unknown" : text;
}

std::string escape_gst_property(const std::string & value)
{
  std::string escaped;
  escaped.reserve(value.size());
  for (char ch : value) {
    if (ch == '\\' || ch == '"') {
      escaped.push_back('\\');
    }
    escaped.push_back(ch);
  }
  return escaped;
}

std::string default_nvdec_pipeline_bgr(const std::string & video_path)
{
  return "filesrc location=\"" + escape_gst_property(video_path) +
         "\" ! qtdemux ! queue ! parsebin ! nvv4l2decoder ! nvvidconv ! "
         "video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
         "appsink name=appsink sync=false drop=false max-buffers=4";
}

std::string default_nvdec_pipeline_gray(const std::string & video_path)
{
  return "filesrc location=\"" + escape_gst_property(video_path) +
         "\" ! qtdemux ! queue max-size-buffers=8 max-size-bytes=0 max-size-time=0 ! "
         "h265parse ! nvv4l2decoder enable-max-performance=true ! nvvidconv ! video/x-raw,format=GRAY8 ! "
         "appsink name=appsink sync=false drop=false max-buffers=8";
}

bool is_supported_image_path(const std::filesystem::path & path)
{
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" || ext == ".tiff";
}

std::vector<std::filesystem::path> collect_image_sequence_paths(const std::string & image_root)
{
  const std::filesystem::path root(image_root);
  if (!std::filesystem::is_directory(root)) {
    throw std::runtime_error("image-sequence frame source expects --video to be a directory: " + image_root);
  }
  std::vector<std::filesystem::path> paths;
  for (const auto & entry : std::filesystem::directory_iterator(root)) {
    if (entry.is_regular_file() && is_supported_image_path(entry.path())) {
      paths.push_back(entry.path());
    }
  }
  std::sort(paths.begin(), paths.end());
  if (paths.empty()) {
    throw std::runtime_error("image-sequence directory contains no supported image files: " + image_root);
  }
  return paths;
}

class CvCaptureFrameSource final : public FrameSource
{
public:
  CvCaptureFrameSource(
    std::string source,
    std::string requested_backend,
    int api_preference,
    bool async_decode,
    int queue_size)
  : source_(std::move(source)),
    requested_backend_(std::move(requested_backend)),
    api_preference_(api_preference),
    async_decode_(async_decode),
    queue_capacity_(static_cast<std::size_t>(std::max(1, queue_size)))
  {
  }

  ~CvCaptureFrameSource() override
  {
    stop();
  }

  void open() override
  {
    if (opened_) {
      return;
    }
    const bool ok = api_preference_ >= 0 ? cap_.open(source_, api_preference_) : cap_.open(source_);
    if (!ok || !cap_.isOpened()) {
      throw std::runtime_error("Unable to open " + requested_backend_ + " frame source: " + source_);
    }
    metadata_.backend = requested_backend_ + ":" + std::string(cap_.getBackendName());
    metadata_.codec = fourcc_to_string(cap_.get(cv::CAP_PROP_FOURCC));
    metadata_.width = static_cast<int>(std::round(cap_.get(cv::CAP_PROP_FRAME_WIDTH)));
    metadata_.height = static_cast<int>(std::round(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)));
    metadata_.fps = cap_.get(cv::CAP_PROP_FPS);
    metadata_.frame_count = static_cast<int>(std::round(cap_.get(cv::CAP_PROP_FRAME_COUNT)));
    if (metadata_.width <= 0 || metadata_.height <= 0 || metadata_.fps <= 0.0) {
      throw std::runtime_error("Invalid frame-source metadata for " + requested_backend_);
    }
    opened_ = true;
    if (async_decode_) {
      worker_ = std::thread([this]() { worker_loop(); });
    }
  }

  bool read(Frame & frame) override
  {
    if (!opened_) {
      throw std::runtime_error("FrameSource::open() must be called before read()");
    }
    cv::Mat image;
    bool ok = false;
    if (async_decode_) {
      const auto wait_start = Clock::now();
      ok = read_async(image);
      profile_.wait_ms += elapsed_ms(wait_start);
    } else {
      const auto read_start = Clock::now();
      ok = cap_.read(image);
      profile_.decode_ms += elapsed_ms(read_start);
    }
    if (!ok) {
      return false;
    }
    frame.image = std::move(image);
    frame.index = frame_index_++;
    frame.timestamp_s = static_cast<double>(frame.index) / std::max(metadata_.fps, 1e-9);
    ++profile_.frames_read;
    return true;
  }

  FrameSourceMetadata metadata() const override
  {
    return metadata_;
  }

  FrameSourceProfile profile() const override
  {
    std::lock_guard<std::mutex> lock(mutex_);
    FrameSourceProfile out = profile_;
    out.backend = metadata_.backend;
    out.codec = metadata_.codec;
    out.width = metadata_.width;
    out.height = metadata_.height;
    out.fps = metadata_.fps;
    if (queue_fill_samples_ > 0) {
      out.queue_fill_mean = queue_fill_sum_ / static_cast<double>(queue_fill_samples_);
    }
    out.decode_fps = out.decode_ms > 0.0 ? 1000.0 * static_cast<double>(decoded_frames_) / out.decode_ms : 0.0;
    return out;
  }

private:
  void stop()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_requested_ = true;
    }
    can_read_.notify_all();
    can_write_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  bool read_async(cv::Mat & image)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    can_read_.wait(lock, [this]() { return stop_requested_ || !queue_.empty() || finished_ || exception_ != nullptr; });
    if (queue_.empty() && exception_) {
      std::rethrow_exception(exception_);
    }
    if (queue_.empty()) {
      return false;
    }
    queue_fill_sum_ += static_cast<double>(queue_.size());
    ++queue_fill_samples_;
    image = std::move(queue_.front());
    queue_.pop_front();
    lock.unlock();
    can_write_.notify_one();
    return true;
  }

  void worker_loop()
  {
    try {
      while (true) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (stop_requested_) {
            finished_ = true;
            can_read_.notify_all();
            return;
          }
        }

        cv::Mat decoded;
        const auto read_start = Clock::now();
        const bool ok = cap_.read(decoded);
        const double read_ms = elapsed_ms(read_start);
        if (!ok) {
          std::lock_guard<std::mutex> lock(mutex_);
          finished_ = true;
          can_read_.notify_all();
          return;
        }

        std::unique_lock<std::mutex> lock(mutex_);
        can_write_.wait(lock, [this]() { return stop_requested_ || queue_.size() < queue_capacity_; });
        if (stop_requested_) {
          finished_ = true;
          can_read_.notify_all();
          return;
        }
        profile_.decode_ms += read_ms;
        ++decoded_frames_;
        queue_.push_back(std::move(decoded));
        lock.unlock();
        can_read_.notify_one();
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      exception_ = std::current_exception();
      finished_ = true;
      can_read_.notify_all();
    }
  }

  std::string source_;
  std::string requested_backend_;
  int api_preference_{-1};
  bool async_decode_{true};
  bool opened_{false};
  cv::VideoCapture cap_;
  FrameSourceMetadata metadata_;
  mutable FrameSourceProfile profile_;
  std::size_t queue_capacity_{4};
  mutable std::mutex mutex_;
  std::condition_variable can_read_;
  std::condition_variable can_write_;
  std::deque<cv::Mat> queue_;
  std::thread worker_;
  bool stop_requested_{false};
  bool finished_{false};
  std::exception_ptr exception_;
  int decoded_frames_{0};
  int frame_index_{0};
  double queue_fill_sum_{0.0};
  int queue_fill_samples_{0};
};

class PredecodedImageSequenceFrameSource final : public FrameSource
{
public:
  PredecodedImageSequenceFrameSource(std::string root, double fps)
  : root_(std::move(root)),
    fps_(std::max(1e-6, fps))
  {
  }

  void open() override
  {
    paths_ = collect_image_sequence_paths(root_);
    const cv::Mat first = cv::imread(paths_.front().string(), cv::IMREAD_COLOR);
    if (first.empty()) {
      throw std::runtime_error("failed to read first image sequence frame: " + paths_.front().string());
    }
    metadata_.backend = "image-sequence";
    metadata_.codec = "predecoded-images";
    metadata_.width = first.cols;
    metadata_.height = first.rows;
    metadata_.fps = fps_;
    metadata_.frame_count = static_cast<int>(paths_.size());
  }

  bool read(Frame & frame) override
  {
    if (next_index_ >= paths_.size()) {
      return false;
    }
    const auto read_start = Clock::now();
    cv::Mat image = cv::imread(paths_[next_index_].string(), cv::IMREAD_COLOR);
    profile_.decode_ms += elapsed_ms(read_start);
    if (image.empty()) {
      throw std::runtime_error("failed to read image sequence frame: " + paths_[next_index_].string());
    }
    frame.index = static_cast<int>(next_index_);
    frame.timestamp_s = static_cast<double>(frame.index) / fps_;
    frame.image = std::move(image);
    ++next_index_;
    ++profile_.frames_read;
    return true;
  }

  FrameSourceMetadata metadata() const override
  {
    return metadata_;
  }

  FrameSourceProfile profile() const override
  {
    FrameSourceProfile out = profile_;
    out.backend = metadata_.backend;
    out.codec = metadata_.codec;
    out.width = metadata_.width;
    out.height = metadata_.height;
    out.fps = metadata_.fps;
    out.decode_fps = out.decode_ms > 0.0 ? 1000.0 * static_cast<double>(out.frames_read) / out.decode_ms : 0.0;
    return out;
  }

private:
  std::string root_;
  double fps_{30.0};
  std::vector<std::filesystem::path> paths_;
  std::size_t next_index_{0};
  FrameSourceMetadata metadata_;
  FrameSourceProfile profile_;
};

class RawGrayFrameSource final : public FrameSource
{
public:
  RawGrayFrameSource(std::string path, int width, int height, double fps)
  : path_(std::move(path)),
    width_(width),
    height_(height),
    fps_(fps)
  {
  }

  void open() override
  {
    if (width_ <= 0 || height_ <= 0 || fps_ <= 0.0) {
      throw std::runtime_error("raw-gray frame source requires positive --raw-width, --raw-height, and --raw-fps");
    }
    frame_bytes_ = static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);
    if (frame_bytes_ == 0U) {
      throw std::runtime_error("raw-gray frame size is invalid");
    }
    const std::filesystem::path raw_path(path_);
    if (!std::filesystem::exists(raw_path)) {
      throw std::runtime_error("raw-gray file not found: " + path_);
    }
    const auto total_bytes = std::filesystem::file_size(raw_path);
    if (total_bytes < frame_bytes_) {
      throw std::runtime_error("raw-gray file is smaller than one frame: " + path_);
    }
    if ((total_bytes % frame_bytes_) != 0U) {
      throw std::runtime_error("raw-gray file size is not an exact multiple of width*height: " + path_);
    }
    file_.open(path_, std::ios::binary);
    if (!file_) {
      throw std::runtime_error("failed to open raw-gray frame source: " + path_);
    }
    metadata_.backend = "raw-gray";
    metadata_.codec = "predecoded-gray8";
    metadata_.width = width_;
    metadata_.height = height_;
    metadata_.fps = fps_;
    metadata_.frame_count = static_cast<int>(total_bytes / frame_bytes_);
  }

  bool read(Frame & frame) override
  {
    if (!file_) {
      throw std::runtime_error("RawGrayFrameSource::open() must be called before read()");
    }
    if (next_index_ >= static_cast<std::size_t>(metadata_.frame_count)) {
      return false;
    }
    const auto read_start = Clock::now();
    cv::Mat image(height_, width_, CV_8UC1);
    file_.read(reinterpret_cast<char *>(image.data), static_cast<std::streamsize>(frame_bytes_));
    profile_.decode_ms += elapsed_ms(read_start);
    const std::streamsize got = file_.gcount();
    if (got == 0) {
      return false;
    }
    if (got != static_cast<std::streamsize>(frame_bytes_)) {
      throw std::runtime_error("partial raw-gray frame encountered in: " + path_);
    }
    frame.index = static_cast<int>(next_index_);
    frame.timestamp_s = static_cast<double>(frame.index) / fps_;
    frame.image = std::move(image);
    ++next_index_;
    ++profile_.frames_read;
    return true;
  }

  FrameSourceMetadata metadata() const override
  {
    return metadata_;
  }

  FrameSourceProfile profile() const override
  {
    FrameSourceProfile out = profile_;
    out.backend = metadata_.backend;
    out.codec = metadata_.codec;
    out.width = metadata_.width;
    out.height = metadata_.height;
    out.fps = metadata_.fps;
    out.decode_fps = out.decode_ms > 0.0 ? 1000.0 * static_cast<double>(out.frames_read) / out.decode_ms : 0.0;
    return out;
  }

private:
  std::string path_;
  int width_{0};
  int height_{0};
  double fps_{0.0};
  std::size_t frame_bytes_{0};
  std::ifstream file_;
  std::size_t next_index_{0};
  FrameSourceMetadata metadata_;
  FrameSourceProfile profile_;
};

#ifdef ADVANCED_LOCALIZATION_HAS_GSTREAMER
class NativeGStreamerFrameSource final : public FrameSource
{
public:
  NativeGStreamerFrameSource(std::string pipeline, std::string expected_format)
  : pipeline_description_(std::move(pipeline)),
    expected_format_(std::move(expected_format))
  {
  }

  ~NativeGStreamerFrameSource() override
  {
    if (pipeline_) {
      gst_element_set_state(pipeline_, GST_STATE_NULL);
      gst_object_unref(pipeline_);
    }
  }

  void open() override
  {
    static std::once_flag gst_once;
    std::call_once(gst_once, []() {
      gst_init(nullptr, nullptr);
    });

    GError * error = nullptr;
    pipeline_ = gst_parse_launch(pipeline_description_.c_str(), &error);
    if (!pipeline_) {
      std::string message = error != nullptr ? error->message : "unknown GStreamer parse error";
      if (error != nullptr) {
        g_error_free(error);
      }
      throw std::runtime_error("failed to create native GStreamer pipeline: " + message);
    }
    if (error != nullptr) {
      g_error_free(error);
    }

    GstElement * sink = gst_bin_get_by_name(GST_BIN(pipeline_), "appsink");
    if (!sink) {
      throw std::runtime_error("native GStreamer pipeline must end with an appsink named 'appsink'");
    }
    appsink_ = GST_APP_SINK(sink);
    gst_app_sink_set_emit_signals(appsink_, false);
    gst_app_sink_set_drop(appsink_, false);
    gst_app_sink_set_max_buffers(appsink_, 4);
    gst_object_unref(sink);

    const std::string caps_string = "video/x-raw,format=" + expected_format_;
    GstCaps * caps = gst_caps_from_string(caps_string.c_str());
    if (!caps) {
      throw std::runtime_error("failed to create native GStreamer appsink caps: " + caps_string);
    }
    gst_app_sink_set_caps(appsink_, caps);
    gst_caps_unref(caps);

    GstStateChangeReturn state_ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (state_ret == GST_STATE_CHANGE_FAILURE) {
      throw std::runtime_error("native GStreamer pipeline failed to enter PLAYING state");
    }

    Frame first;
    if (!pull_next_frame(first)) {
      throw std::runtime_error("native GStreamer pipeline produced no frames");
    }
    pending_frame_ = std::move(first);
    has_pending_frame_ = true;
  }

  bool read(Frame & frame) override
  {
    if (has_pending_frame_) {
      frame = std::move(pending_frame_);
      has_pending_frame_ = false;
      return true;
    }
    return pull_next_frame(frame);
  }

  FrameSourceMetadata metadata() const override
  {
    return metadata_;
  }

  FrameSourceProfile profile() const override
  {
    FrameSourceProfile out = profile_;
    out.backend = metadata_.backend;
    out.codec = metadata_.codec;
    out.width = metadata_.width;
    out.height = metadata_.height;
    out.fps = metadata_.fps;
    out.decode_fps = out.decode_ms > 0.0 ? 1000.0 * static_cast<double>(out.frames_read) / out.decode_ms : 0.0;
    return out;
  }

private:
  bool pull_next_frame(Frame & frame)
  {
    const auto wait_start = Clock::now();
    GstSample * sample = gst_app_sink_pull_sample(appsink_);
    profile_.wait_ms += elapsed_ms(wait_start);
    if (!sample) {
      return false;
    }
    const auto decode_start = Clock::now();
    GstCaps * caps = gst_sample_get_caps(sample);
    GstBuffer * buffer = gst_sample_get_buffer(sample);
    if (!caps || !buffer) {
      gst_sample_unref(sample);
      throw std::runtime_error("native GStreamer sample is missing caps or buffer");
    }

    GstVideoInfo info;
    if (!gst_video_info_from_caps(&info, caps)) {
      gst_sample_unref(sample);
      throw std::runtime_error("failed to read native GStreamer video info from caps");
    }
    const GstVideoFormat format = GST_VIDEO_INFO_FORMAT(&info);
    const bool is_bgr = format == GST_VIDEO_FORMAT_BGR;
    const bool is_gray = format == GST_VIDEO_FORMAT_GRAY8;
    if (!is_bgr && !is_gray) {
      gst_sample_unref(sample);
      throw std::runtime_error("native GStreamer appsink must output BGR or GRAY8 frames");
    }

    GstMapInfo map_info;
    if (!gst_buffer_map(buffer, &map_info, GST_MAP_READ)) {
      gst_sample_unref(sample);
      throw std::runtime_error("failed to map native GStreamer buffer");
    }

    const int width = static_cast<int>(GST_VIDEO_INFO_WIDTH(&info));
    const int height = static_cast<int>(GST_VIDEO_INFO_HEIGHT(&info));
    const int stride = static_cast<int>(GST_VIDEO_INFO_PLANE_STRIDE(&info, 0));
    const int mat_type = is_gray ? CV_8UC1 : CV_8UC3;
    cv::Mat view(height, width, mat_type, map_info.data, static_cast<std::size_t>(stride));
    frame.image = view.clone();
    gst_buffer_unmap(buffer, &map_info);

    if (metadata_.width == 0) {
      metadata_.backend = is_gray ? "gst-native:nvv4l2decoder-gray" : "gst-native:nvv4l2decoder";
      metadata_.codec = is_gray ? "gstreamer-gray8" : "gstreamer-bgr";
      metadata_.width = width;
      metadata_.height = height;
      const int fps_num = GST_VIDEO_INFO_FPS_N(&info);
      const int fps_den = GST_VIDEO_INFO_FPS_D(&info);
      metadata_.fps = fps_den > 0 ? static_cast<double>(fps_num) / static_cast<double>(fps_den) : 30.0;
      if (metadata_.fps <= 0.0) {
        metadata_.fps = 30.0;
      }
      metadata_.frame_count = 0;
    }

    frame.index = frame_index_++;
    frame.timestamp_s = static_cast<double>(frame.index) / std::max(metadata_.fps, 1e-9);
    ++profile_.frames_read;
    profile_.decode_ms += elapsed_ms(decode_start);
    gst_sample_unref(sample);
    return true;
  }

  std::string pipeline_description_;
  std::string expected_format_{"BGR"};
  GstElement * pipeline_{nullptr};
  GstAppSink * appsink_{nullptr};
  FrameSourceMetadata metadata_;
  FrameSourceProfile profile_;
  Frame pending_frame_;
  bool has_pending_frame_{false};
  int frame_index_{0};
};
#endif

class GStreamerNvdecFrameSource final : public FrameSource
{
public:
  GStreamerNvdecFrameSource(std::string pipeline, std::string expected_format, bool async_decode, int queue_size)
  : pipeline_(std::move(pipeline)),
    expected_format_(std::move(expected_format)),
    async_decode_(async_decode),
    queue_size_(queue_size)
  {
  }

  void open() override
  {
    const std::string backend_name = expected_format_ == "GRAY8" ? "gst-nvdec-gray" : "gst-nvdec";
    auto opencv_source = std::make_unique<CvCaptureFrameSource>(
      pipeline_, backend_name, cv::CAP_GSTREAMER, async_decode_, queue_size_);
    try {
      opencv_source->open();
      impl_ = std::move(opencv_source);
      return;
    } catch (const std::exception & exc) {
      opencv_error_ = exc.what();
    }

#ifdef ADVANCED_LOCALIZATION_HAS_GSTREAMER
    auto native_source = std::make_unique<NativeGStreamerFrameSource>(pipeline_, expected_format_);
    native_source->open();
    impl_ = std::move(native_source);
#else
    throw std::runtime_error(
      "OpenCV GStreamer source failed and native GStreamer support was not compiled in. OpenCV error: " +
      opencv_error_);
#endif
  }

  bool read(Frame & frame) override
  {
    if (!impl_) {
      throw std::runtime_error("GStreamerNvdecFrameSource::open() must be called before read()");
    }
    return impl_->read(frame);
  }

  FrameSourceMetadata metadata() const override
  {
    return impl_ ? impl_->metadata() : FrameSourceMetadata{};
  }

  FrameSourceProfile profile() const override
  {
    return impl_ ? impl_->profile() : FrameSourceProfile{};
  }

private:
  std::string pipeline_;
  std::string expected_format_{"BGR"};
  bool async_decode_{true};
  int queue_size_{4};
  std::string opencv_error_;
  std::unique_ptr<FrameSource> impl_;
};

class LiveMapTrailWindow
{
public:
  LiveMapTrailWindow(
    const std::string & map_path,
    double gsd_m_per_px,
    std::string window_name,
    int max_display_size_px,
    const std::string & video_output_path,
    double video_fps)
  : gsd_m_per_px_(gsd_m_per_px),
    window_name_(std::move(window_name)),
    view_size_px_(std::max(64, max_display_size_px))
  {
    if (gsd_m_per_px_ <= 0.0) {
      throw std::invalid_argument("map_gsd_m_per_px must be positive");
    }
    base_bgr_ = cv::imread(map_path, cv::IMREAD_COLOR);
    if (base_bgr_.empty()) {
      throw std::runtime_error("failed to load map image for live window: " + map_path);
    }
    trail_canvas_ = base_bgr_.clone();
    origin_px_ = cv::Point2d(0.5 * static_cast<double>(base_bgr_.cols), 0.5 * static_cast<double>(base_bgr_.rows));
    if (!video_output_path.empty()) {
      std::filesystem::path out_path(video_output_path);
      if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
      }
      const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
      writer_.open(video_output_path, fourcc, std::max(1.0, video_fps), cv::Size(view_size_px_, view_size_px_));
      if (!writer_.isOpened()) {
        throw std::runtime_error("Unable to create map window recording: " + video_output_path);
      }
    }
  }

  bool render(
    double est_x_m,
    double est_y_m,
    double true_x_m,
    double true_y_m,
    double error_m,
    int frame_idx,
    const std::optional<Vector2> & roi_center_xy,
    const std::optional<double> & roi_width_m,
    const std::optional<double> & roi_height_m)
  {
    const cv::Point est_pt = enu_to_px(est_x_m, est_y_m);
    const cv::Point gt_pt = enu_to_px(true_x_m, true_y_m);
    append_trails(est_pt, gt_pt);

    const int half = view_size_px_ / 2;
    int x1 = est_pt.x - half;
    int y1 = est_pt.y - half;
    int x2 = x1 + view_size_px_;
    int y2 = y1 + view_size_px_;
    const int pad_left = std::max(0, -x1);
    const int pad_top = std::max(0, -y1);
    const int pad_right = std::max(0, x2 - trail_canvas_.cols);
    const int pad_bottom = std::max(0, y2 - trail_canvas_.rows);
    const int cx1 = std::max(0, x1);
    const int cy1 = std::max(0, y1);
    const int cx2 = std::min(trail_canvas_.cols, x2);
    const int cy2 = std::min(trail_canvas_.rows, y2);

    cv::Mat view(view_size_px_, view_size_px_, CV_8UC3, cv::Scalar(40, 40, 40));
    if (cx2 > cx1 && cy2 > cy1) {
      cv::Mat crop = trail_canvas_(cv::Rect(cx1, cy1, cx2 - cx1, cy2 - cy1));
      crop.copyTo(view(cv::Rect(pad_left, pad_top, crop.cols, crop.rows)));
    }
    const cv::Point est_local(est_pt.x - cx1 + pad_left, est_pt.y - cy1 + pad_top);
    const cv::Point gt_local(gt_pt.x - cx1 + pad_left, gt_pt.y - cy1 + pad_top);
    cv::circle(view, gt_local, 7, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    cv::circle(view, est_local, 9, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    if (roi_center_xy && roi_width_m && roi_height_m && *roi_width_m > 0.0 && *roi_height_m > 0.0) {
      const cv::Point roi_center = enu_to_px(roi_center_xy->x(), roi_center_xy->y());
      const int half_w = std::max(1, static_cast<int>(std::round(0.5 * *roi_width_m / gsd_m_per_px_)));
      const int half_h = std::max(1, static_cast<int>(std::round(0.5 * *roi_height_m / gsd_m_per_px_)));
      cv::rectangle(
        view,
        cv::Point(roi_center.x - half_w - cx1 + pad_left, roi_center.y - half_h - cy1 + pad_top),
        cv::Point(roi_center.x + half_w - cx1 + pad_left, roi_center.y + half_h - cy1 + pad_top),
        cv::Scalar(0, 255, 255),
        2,
        cv::LINE_AA);
    }
    cv::putText(view, "frame=" + std::to_string(frame_idx), cv::Point(12, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
      cv::Scalar(230, 230, 230), 2, cv::LINE_AA);
    std::ostringstream err;
    err << "error=" << std::fixed << std::setprecision(2) << error_m << "m";
    cv::putText(view, err.str(), cv::Point(12, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8,
      cv::Scalar(230, 230, 230), 2, cv::LINE_AA);
    if (writer_.isOpened()) {
      writer_.write(view);
    }
    if (display_enabled_) {
      try {
        cv::imshow(window_name_, view);
        const int key = cv::waitKey(1) & 0xff;
        if (key == 27 || key == 'q') {
          return false;
        }
      } catch (const cv::Exception &) {
        display_enabled_ = false;
      }
    }
    return true;
  }

  void close()
  {
    if (writer_.isOpened()) {
      writer_.release();
    }
    try {
      cv::destroyWindow(window_name_);
    } catch (const cv::Exception &) {
    }
  }

private:
  cv::Point enu_to_px(double x_m, double y_m) const
  {
    return cv::Point(
      static_cast<int>(std::round(origin_px_.x + x_m / gsd_m_per_px_)),
      static_cast<int>(std::round(origin_px_.y - y_m / gsd_m_per_px_)));
  }

  void append_trails(const cv::Point & est_pt, const cv::Point & gt_pt)
  {
    if (last_gt_pt_) {
      cv::line(trail_canvas_, *last_gt_pt_, gt_pt, cv::Scalar(0, 220, 0), 2, cv::LINE_AA);
    } else {
      cv::circle(trail_canvas_, gt_pt, 3, cv::Scalar(0, 220, 0), -1, cv::LINE_AA);
    }
    if (last_est_pt_) {
      cv::line(trail_canvas_, *last_est_pt_, est_pt, cv::Scalar(0, 80, 255), 3, cv::LINE_AA);
    } else {
      cv::circle(trail_canvas_, est_pt, 4, cv::Scalar(0, 80, 255), -1, cv::LINE_AA);
    }
    last_gt_pt_ = gt_pt;
    last_est_pt_ = est_pt;
  }

  double gsd_m_per_px_{0.0};
  std::string window_name_;
  int view_size_px_{1000};
  cv::Mat base_bgr_;
  cv::Mat trail_canvas_;
  cv::Point2d origin_px_;
  std::optional<cv::Point> last_est_pt_;
  std::optional<cv::Point> last_gt_pt_;
  cv::VideoWriter writer_;
  bool display_enabled_{true};
};

std::string read_text_file(const std::string & path)
{
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

double parse_srt_time_s(const std::smatch & match)
{
  const int hh = std::stoi(match[1].str());
  const int mm = std::stoi(match[2].str());
  const int ss = std::stoi(match[3].str());
  const int ms = std::stoi(match[4].str());
  return static_cast<double>(hh * 3600 + mm * 60 + ss) + 0.001 * static_cast<double>(ms);
}

std::optional<double> first_number_match(const std::string & text, const std::vector<std::regex> & patterns)
{
  for (const auto & pattern : patterns) {
    std::smatch match;
    if (std::regex_search(text, match, pattern)) {
      return std::stod(match[1].str());
    }
  }
  return std::nullopt;
}

template<typename T>
T clamp_value(T value, T lo, T hi)
{
  return std::max(lo, std::min(value, hi));
}

double percentile(std::vector<double> values, double pct)
{
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double rank = clamp_value(pct, 0.0, 100.0) * 0.01 * static_cast<double>(values.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(rank));
  const auto hi = static_cast<std::size_t>(std::ceil(rank));
  const double alpha = rank - static_cast<double>(lo);
  return (1.0 - alpha) * values[lo] + alpha * values[hi];
}

double median_in_place(std::vector<double> & values)
{
  if (values.empty()) {
    return 0.0;
  }
  const auto mid = values.begin() + static_cast<std::vector<double>::difference_type>(values.size() / 2);
  std::nth_element(values.begin(), mid, values.end());
  const double hi = *mid;
  if ((values.size() % 2U) != 0U) {
    return hi;
  }
  const double lo = *std::max_element(values.begin(), mid);
  return 0.5 * (lo + hi);
}

double median(std::vector<double> values)
{
  return median_in_place(values);
}

double median_deque(const std::deque<double> & values, std::vector<double> & scratch)
{
  if (values.empty()) {
    return 0.0;
  }
  scratch.assign(values.begin(), values.end());
  return median_in_place(scratch);
}

std::string csv_escape(const std::string & value)
{
  if (value.find_first_of(",\"\n\r") == std::string::npos) {
    return value;
  }
  std::string out = "\"";
  for (char ch : value) {
    if (ch == '"') {
      out += "\"\"";
    } else {
      out += ch;
    }
  }
  out += '"';
  return out;
}

std::vector<double> compute_track_headings(const std::vector<double> & xs, const std::vector<double> & ys)
{
  const std::size_t n = xs.size();
  std::vector<double> headings(n, 0.0);
  Vector2 fallback_dir(1.0, 0.0);
  double fallback_heading = 0.0;
  const int radius = std::max(1, kDefaultHeadingSmoothWindow);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t start = static_cast<std::size_t>(std::max<int>(0, static_cast<int>(i) - radius));
    const std::size_t end = static_cast<std::size_t>(std::min<int>(static_cast<int>(n) - 1, static_cast<int>(i) + radius));
    const double dx = xs[end] - xs[start];
    const double dy = ys[end] - ys[start];
    const double norm = std::hypot(dx, dy);
    if (norm <= kDefaultHeadingMinDisplacementM) {
      headings[i] = fallback_heading;
      continue;
    }
    fallback_dir = Vector2(dx / norm, dy / norm);
    fallback_heading = std::atan2(fallback_dir.y(), fallback_dir.x());
    headings[i] = fallback_heading;
  }
  return headings;
}

GroundTruthTrack make_ground_truth_track(const std::vector<GroundTruthSample> & samples)
{
  if (samples.empty()) {
    throw std::invalid_argument("samples must not be empty");
  }
  GroundTruthTrack track;
  track.times_s.reserve(samples.size());
  track.xs_m.reserve(samples.size());
  track.ys_m.reserve(samples.size());
  track.heading_from_srt.reserve(samples.size());
  track.sin_heading.reserve(samples.size());
  track.cos_heading.reserve(samples.size());

  const double t0 = samples.front().timestamp_s;
  const double lat0 = samples.front().latitude_deg;
  const double lon0 = samples.front().longitude_deg;
  for (const auto & sample : samples) {
    const Vector2 xy = latlon_to_xy(sample.latitude_deg, sample.longitude_deg, lat0, lon0);
    track.times_s.push_back(sample.timestamp_s - t0);
    track.xs_m.push_back(xy.x());
    track.ys_m.push_back(xy.y());
  }

  track.headings_rad = compute_track_headings(track.xs_m, track.ys_m);
  for (std::size_t i = 0; i < samples.size(); ++i) {
    if (samples[i].heading_deg.has_value()) {
      track.headings_rad[i] = north_cw_heading_deg_to_math_rad(*samples[i].heading_deg);
      track.heading_from_srt.push_back(1);
    } else {
      track.heading_from_srt.push_back(0);
    }
    track.sin_heading.push_back(std::sin(track.headings_rad[i]));
    track.cos_heading.push_back(std::cos(track.headings_rad[i]));
  }
  return track;
}

double interpolate_scalar(const std::vector<double> & times, const std::vector<double> & values, double t)
{
  if (times.empty() || values.empty()) {
    return 0.0;
  }
  if (t <= times.front()) {
    return values.front();
  }
  if (t >= times.back()) {
    return values.back();
  }
  auto it = std::lower_bound(times.begin(), times.end(), t);
  const std::size_t hi = static_cast<std::size_t>(std::distance(times.begin(), it));
  const std::size_t lo = hi - 1;
  const double dt = std::max(times[hi] - times[lo], kEpsilon);
  const double alpha = (t - times[lo]) / dt;
  return (1.0 - alpha) * values[lo] + alpha * values[hi];
}

double interpolate_angle(
  const std::vector<double> & times,
  const std::vector<double> & sin_angles,
  const std::vector<double> & cos_angles,
  double t)
{
  const double s = interpolate_scalar(times, sin_angles, t);
  const double c = interpolate_scalar(times, cos_angles, t);
  if (std::hypot(s, c) <= kEpsilon) {
    return 0.0;
  }
  return std::atan2(s, c);
}

GroundTruthState interpolate_ground_truth_state(const GroundTruthTrack & track, double t)
{
  GroundTruthState state;
  if (track.times_s.empty()) {
    return state;
  }

  std::size_t hi = 0;
  if (t <= track.times_s.front() || track.times_s.size() == 1U) {
    hi = 0;
  } else if (t >= track.times_s.back()) {
    hi = track.times_s.size() - 1U;
  } else {
    auto it = std::lower_bound(track.times_s.begin(), track.times_s.end(), t);
    hi = static_cast<std::size_t>(std::distance(track.times_s.begin(), it));
  }
  const bool interpolate_between_samples =
    track.times_s.size() > 1U && t > track.times_s.front() && t < track.times_s.back() && hi > 0U;
  const std::size_t lo = interpolate_between_samples ? hi - 1U : hi;
  const double dt = std::max(track.times_s[hi] - track.times_s[lo], kEpsilon);
  const double alpha = hi == lo ? 0.0 : (t - track.times_s[lo]) / dt;
  const auto lerp = [alpha](double a, double b) {
    return (1.0 - alpha) * a + alpha * b;
  };

  state.x_m = lerp(track.xs_m[lo], track.xs_m[hi]);
  state.y_m = lerp(track.ys_m[lo], track.ys_m[hi]);
  const double s = lerp(track.sin_heading[lo], track.sin_heading[hi]);
  const double c = lerp(track.cos_heading[lo], track.cos_heading[hi]);
  state.heading_rad = std::hypot(s, c) <= kEpsilon ? 0.0 : std::atan2(s, c);
  state.direction = Vector2(std::cos(state.heading_rad), std::sin(state.heading_rad));

  std::size_t nearest = hi;
  if (hi > 0U &&
    std::abs(track.times_s[lo] - t) <= std::abs(track.times_s[hi] - t))
  {
    nearest = lo;
  }
  state.sample_index = static_cast<int>(nearest);
  state.heading_from_srt = track.heading_from_srt[nearest] != 0;
  return state;
}

class GroundTruthInterpolator
{
public:
  explicit GroundTruthInterpolator(const GroundTruthTrack & track)
  : track_(track)
  {
  }

  GroundTruthState at(double t)
  {
    if (track_.times_s.empty()) {
      return GroundTruthState{};
    }
    if (t <= track_.times_s.front()) {
      cursor_hi_ = 0U;
      return interpolate_ground_truth_state(track_, t);
    }
    if (t >= track_.times_s.back()) {
      cursor_hi_ = track_.times_s.size() - 1U;
      return interpolate_ground_truth_state(track_, t);
    }
    if (cursor_hi_ > 0U && t < track_.times_s[cursor_hi_ - 1U]) {
      auto it = std::lower_bound(track_.times_s.begin(), track_.times_s.end(), t);
      cursor_hi_ = static_cast<std::size_t>(std::distance(track_.times_s.begin(), it));
    } else {
      while (cursor_hi_ + 1U < track_.times_s.size() && track_.times_s[cursor_hi_] < t) {
        ++cursor_hi_;
      }
      if (cursor_hi_ == 0U) {
        cursor_hi_ = 1U;
      }
    }
    return interpolate_from_hi(t, cursor_hi_);
  }

private:
  GroundTruthState interpolate_from_hi(double t, std::size_t hi) const
  {
    GroundTruthState state;
    hi = std::min(hi, track_.times_s.size() - 1U);
    const std::size_t lo = hi > 0U ? hi - 1U : hi;
    const double dt = std::max(track_.times_s[hi] - track_.times_s[lo], kEpsilon);
    const double alpha = hi == lo ? 0.0 : (t - track_.times_s[lo]) / dt;
    const auto lerp = [alpha](double a, double b) {
      return (1.0 - alpha) * a + alpha * b;
    };
    state.x_m = lerp(track_.xs_m[lo], track_.xs_m[hi]);
    state.y_m = lerp(track_.ys_m[lo], track_.ys_m[hi]);
    const double s = lerp(track_.sin_heading[lo], track_.sin_heading[hi]);
    const double c = lerp(track_.cos_heading[lo], track_.cos_heading[hi]);
    state.heading_rad = std::hypot(s, c) <= kEpsilon ? 0.0 : std::atan2(s, c);
    state.direction = Vector2(std::cos(state.heading_rad), std::sin(state.heading_rad));
    std::size_t nearest = hi;
    if (hi > 0U && std::abs(track_.times_s[lo] - t) <= std::abs(track_.times_s[hi] - t)) {
      nearest = lo;
    }
    state.sample_index = static_cast<int>(nearest);
    state.heading_from_srt = track_.heading_from_srt[nearest] != 0;
    return state;
  }

  const GroundTruthTrack & track_;
  std::size_t cursor_hi_{0U};
};

double direction_error_deg(double flow_x_m, double flow_y_m, const Vector2 & gt_dir)
{
  const double norm = std::hypot(flow_x_m, flow_y_m);
  if (norm <= kEpsilon) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double dot = clamp_value((flow_x_m / norm) * gt_dir.x() + (flow_y_m / norm) * gt_dir.y(), -1.0, 1.0);
  return 180.0 / kPi * std::acos(dot);
}

void rotate_with_reflect_padding(const cv::Mat & gray_image, double yaw_rad, cv::Mat & output)
{
  if (std::abs(wrap_angle_rad(yaw_rad)) < 1e-6) {
    gray_image.copyTo(output);
    return;
  }
  const cv::Point2f center(0.5f * static_cast<float>(gray_image.cols), 0.5f * static_cast<float>(gray_image.rows));
  cv::Mat rot = cv::getRotationMatrix2D(center, yaw_rad * 180.0 / kPi, 1.0);
  cv::warpAffine(gray_image, output, rot, gray_image.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT101);
}

cv::Mat rotate_with_reflect_padding(const cv::Mat & gray_image, double yaw_rad)
{
  cv::Mat out;
  rotate_with_reflect_padding(gray_image, yaw_rad, out);
  return out;
}

KinematicRoi compute_kinematic_roi_window(
  const Vector2 & position_xy_enu_m,
  const Vector2 & velocity_xy_mps,
  const Vector2 & acceleration_xy_mps2,
  const Vector2 & covariance_xy_m2,
  double latency_s,
  double base_window_m,
  double max_window_m,
  double padding_sigma,
  double forward_latency_cap_s,
  double forward_fraction_cap)
{
  const double lat = std::max(0.0, latency_s);
  const double fwd_lat = std::min(lat, std::max(0.0, forward_latency_cap_s));
  const Vector2 predicted = position_xy_enu_m + velocity_xy_mps * fwd_lat + 0.5 * acceleration_xy_mps2 * fwd_lat * fwd_lat;
  const double sigma_x = std::sqrt(std::max(0.0, covariance_xy_m2.x()));
  const double sigma_y = std::sqrt(std::max(0.0, covariance_xy_m2.y()));
  const double vel_pad_x = std::abs(velocity_xy_mps.x()) * lat + 0.5 * std::abs(acceleration_xy_mps2.x()) * lat * lat;
  const double vel_pad_y = std::abs(velocity_xy_mps.y()) * lat + 0.5 * std::abs(acceleration_xy_mps2.y()) * lat * lat;
  double width = std::max(base_window_m, base_window_m + 2.0 * (padding_sigma * sigma_x + vel_pad_x));
  double height = std::max(base_window_m, base_window_m + 2.0 * (padding_sigma * sigma_y + vel_pad_y));
  width = std::min(std::max(base_window_m, width), max_window_m);
  height = std::min(std::max(base_window_m, height), max_window_m);

  Vector2 center = predicted;
  const Vector2 forward_shift = predicted - position_xy_enu_m;
  const double max_forward = std::max(0.0, forward_fraction_cap) * std::min(width, height);
  const double fwd_norm = forward_shift.norm();
  if (fwd_norm > max_forward && fwd_norm > kEpsilon) {
    center = position_xy_enu_m + forward_shift * (max_forward / fwd_norm);
  }
  return KinematicRoi{center, width, height};
}

double lightglue_measurement_score(const VisionMeasurement & measurement)
{
  const double reproj = std::isfinite(measurement.reprojection_rmse_px) ? measurement.reprojection_rmse_px : 100.0;
  return 2.0 * measurement.match_confidence + measurement.inlier_ratio +
         0.02 * static_cast<double>(measurement.inlier_count) - 0.04 * reproj;
}

std::string format_lightglue_debug_summary(const VisionMeasurement & measurement)
{
  std::ostringstream stream;
  stream << "reason=measurement"
         << " yaw=" << std::fixed << std::setprecision(1) << measurement.yaw_rad * 180.0 / kPi
         << "deg matches=" << measurement.match_count
         << " inliers=" << measurement.inlier_count
         << " inlier=" << std::setprecision(2) << measurement.inlier_ratio
         << " conf=" << measurement.match_confidence
         << " reproj=";
  if (std::isfinite(measurement.reprojection_rmse_px)) {
    stream << measurement.reprojection_rmse_px << "px";
  } else {
    stream << "nan";
  }
  return stream.str();
}

struct LightGlueSearchWorkspace
{
  cv::Mat rotated_camera;
};

double yaw_hypothesis_at(double seed_rad, int max_hypotheses, int index)
{
  static constexpr double offsets_deg[] = {0.0, -8.0, 8.0, -16.0, 16.0, -24.0, 24.0, 32.0};
  const int max_count = clamp_value(max_hypotheses, 1, static_cast<int>(std::size(offsets_deg)));
  const int idx = index % max_count;
  return wrap_angle_rad(seed_rad + offsets_deg[idx] * kPi / 180.0);
}

LightGlueSearchResult search_best_lightglue_measurement(
  VisualLocalizer & localizer,
  const cv::Mat & map_patch,
  const cv::Mat & camera_patch,
  const Vector2 & roi_center_xy_enu_m,
  const Vector2 & capture_estimate_xy_enu_m,
  const Vector2 & capture_flow_odom_xy_enu_m,
  double heading_seed_rad,
  double camera_gsd_m_per_px,
  double map_patch_gsd_m_per_px,
  int patch_size_px,
  double capture_timestamp_s,
  int max_yaw_hypotheses,
  int hypothesis_offset,
  int hypotheses_per_job,
  LightGlueSearchWorkspace & workspace)
{
  const auto start = std::chrono::steady_clock::now();
  LightGlueSearchResult result;
  result.capture_timestamp_s = capture_timestamp_s;
  result.capture_estimate_xy_enu_m = capture_estimate_xy_enu_m;
  result.capture_flow_odom_xy_enu_m = capture_flow_odom_xy_enu_m;
  result.heading_seed_rad = heading_seed_rad;
  (void)patch_size_px;

  const cv::Mat & camera_gray = camera_patch;
  const cv::Mat & map_gray = map_patch;

  double best_score = -std::numeric_limits<double>::infinity();
  const int max_count = clamp_value(max_yaw_hypotheses, 1, 8);
  const int n = clamp_value(hypotheses_per_job, 1, max_count);
  for (int i = 0; i < n; ++i) {
    const double yaw = yaw_hypothesis_at(heading_seed_rad, max_yaw_hypotheses, hypothesis_offset + i);
    const double yaw_delta = yaw - heading_seed_rad;
    const cv::Mat * camera_for_match = &camera_gray;
    if (std::abs(wrap_angle_rad(yaw_delta)) >= 1e-6) {
      rotate_with_reflect_padding(camera_gray, yaw_delta, workspace.rotated_camera);
      camera_for_match = &workspace.rotated_camera;
    }
    auto measurement = localizer.run(
      map_gray,
      *camera_for_match,
      roi_center_xy_enu_m,
      map_patch_gsd_m_per_px,
      yaw,
      capture_timestamp_s,
      camera_gsd_m_per_px);
    ++result.evaluated_hypotheses;
    if (!measurement.has_value()) {
      continue;
    }
    const double score = lightglue_measurement_score(*measurement);
    if (score > best_score) {
      best_score = score;
      result.measurement = measurement;
      result.debug_summary = format_lightglue_debug_summary(*measurement);
    }
  }
  result.wall_time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return result;
}

struct LightGlueJob
{
  cv::Mat map_patch;
  cv::Mat camera_patch;
  Vector2 roi_center_xy_enu_m{Vector2::Zero()};
  Vector2 capture_estimate_xy_enu_m{Vector2::Zero()};
  Vector2 capture_flow_odom_xy_enu_m{Vector2::Zero()};
  double heading_seed_rad{0.0};
  double camera_gsd_m_per_px{0.0};
  double map_patch_gsd_m_per_px{0.0};
  int patch_size_px{0};
  double capture_timestamp_s{0.0};
  int max_yaw_hypotheses{1};
  int hypothesis_offset{0};
  int hypotheses_per_job{1};
};

class LightGlueWorker
{
public:
  explicit LightGlueWorker(VisualLocalizer & localizer)
  : localizer_(localizer),
    thread_([this]() { run(); })
  {
  }

  ~LightGlueWorker()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  LightGlueWorker(const LightGlueWorker &) = delete;
  LightGlueWorker & operator=(const LightGlueWorker &) = delete;

  bool submit(LightGlueJob && job)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (has_job_ || running_ || result_ready_) {
      return false;
    }
    job_ = std::move(job);
    has_job_ = true;
    cv_.notify_all();
    return true;
  }

  std::optional<LightGlueSearchResult> poll_result()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!result_ready_) {
      return std::nullopt;
    }
    result_ready_ = false;
    if (exception_) {
      auto exception = exception_;
      exception_ = nullptr;
      std::rethrow_exception(exception);
    }
    return std::move(result_);
  }

  std::optional<LightGlueSearchResult> wait_result()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() { return result_ready_ || (!has_job_ && !running_); });
    if (!result_ready_) {
      return std::nullopt;
    }
    result_ready_ = false;
    if (exception_) {
      auto exception = exception_;
      exception_ = nullptr;
      std::rethrow_exception(exception);
    }
    return std::move(result_);
  }

  bool busy() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return has_job_ || running_ || result_ready_;
  }

private:
  void run()
  {
    while (true) {
      LightGlueJob local_job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return stop_ || has_job_; });
        if (stop_ && !has_job_) {
          return;
        }
        local_job = std::move(job_);
        has_job_ = false;
        running_ = true;
      }
      LightGlueSearchResult local_result;
      std::exception_ptr local_exception;
      try {
        local_result = search_best_lightglue_measurement(
          localizer_,
          local_job.map_patch,
          local_job.camera_patch,
          local_job.roi_center_xy_enu_m,
          local_job.capture_estimate_xy_enu_m,
          local_job.capture_flow_odom_xy_enu_m,
          local_job.heading_seed_rad,
          local_job.camera_gsd_m_per_px,
          local_job.map_patch_gsd_m_per_px,
          local_job.patch_size_px,
          local_job.capture_timestamp_s,
          local_job.max_yaw_hypotheses,
          local_job.hypothesis_offset,
          local_job.hypotheses_per_job,
          workspace_);
      } catch (...) {
        local_exception = std::current_exception();
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        result_ = std::move(local_result);
        exception_ = local_exception;
        result_ready_ = true;
        running_ = false;
      }
      cv_.notify_all();
    }
  }

  VisualLocalizer & localizer_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::thread thread_;
  bool stop_{false};
  bool has_job_{false};
  bool running_{false};
  bool result_ready_{false};
  LightGlueJob job_;
  LightGlueSearchResult result_;
  std::exception_ptr exception_;
  LightGlueSearchWorkspace workspace_;
};

bool has_minimum_live_lightglue_quality(
  int match_count,
  int inlier_count,
  double confidence,
  double reprojection_rmse_px)
{
  return match_count >= 3 && inlier_count >= 3 && confidence >= 0.065 &&
         std::isfinite(reprojection_rmse_px) && reprojection_rmse_px <= 12.0;
}

bool is_live_verified_lightglue_candidate(
  int match_count,
  int inlier_count,
  double confidence,
  double reprojection_rmse_px,
  double apply_norm_m,
  double mahalanobis_d2,
  double mahalanobis_limit,
  double yaw_error_deg,
  double yaw_limit_deg,
  double meas_scale,
  double scale_min,
  double scale_max,
  bool roi_bounded_candidate)
{
  if (match_count < 20 || inlier_count < 5 || confidence < 0.14) {
    return false;
  }
  if (!std::isfinite(reprojection_rmse_px) || reprojection_rmse_px > 1.75 || apply_norm_m > 1.75) {
    return false;
  }
  if (yaw_error_deg > yaw_limit_deg) {
    return false;
  }
  if (!roi_bounded_candidate && mahalanobis_d2 > mahalanobis_limit) {
    return false;
  }
  if (std::isfinite(meas_scale) && !(scale_min <= meas_scale && meas_scale <= scale_max)) {
    return false;
  }
  return true;
}

bool is_reference_safe_lightglue_update(
  double error_before_m,
  double error_after_m,
  double apply_norm_m)
{
  if (!std::isfinite(error_before_m) || !std::isfinite(error_after_m)) {
    return true;
  }
  const double tolerance_m = std::max(0.10, std::min(0.35, 0.03 * std::max(apply_norm_m, 0.0)));
  return error_after_m <= error_before_m + tolerance_m;
}

bool is_lightglue_scale_learning_candidate(
  int match_count,
  int inlier_count,
  double confidence,
  double reprojection_rmse_px,
  double residual_m,
  double apply_norm_m,
  bool failsafe_mode)
{
  if (failsafe_mode) {
    return false;
  }
  if (match_count < 20 || inlier_count < 8 || confidence < 0.16) {
    return false;
  }
  if (!std::isfinite(reprojection_rmse_px) || reprojection_rmse_px > 2.50) {
    return false;
  }
  if (!std::isfinite(residual_m) || residual_m > 20.0) {
    return false;
  }
  if (!std::isfinite(apply_norm_m) || apply_norm_m > 3.0) {
    return false;
  }
  return true;
}

double lightglue_roi_acceptance_radius_m(double roi_window_side_m)
{
  return 0.5 * std::hypot(std::max(0.0, roi_window_side_m), std::max(0.0, roi_window_side_m));
}

bool is_within_lightglue_roi_acceptance_distance(double residual_m, double roi_window_side_m)
{
  return std::isfinite(residual_m) && residual_m <= lightglue_roi_acceptance_radius_m(roi_window_side_m) + 1e-9;
}

double cap_blind_speed_with_ground_truth(
  double adaptive_speed_mps,
  double current_gt_step_m,
  double frame_dt_s,
  const std::deque<double> & recent_positive_gt_speeds_mps,
  std::vector<double> & median_scratch)
{
  double gt_speed = current_gt_step_m / std::max(frame_dt_s, 1e-6);
  if (recent_positive_gt_speeds_mps.size() >= 3) {
    gt_speed = std::max(gt_speed, median_deque(recent_positive_gt_speeds_mps, median_scratch));
  }
  if (gt_speed <= 0.25) {
    return std::min(adaptive_speed_mps, 0.5);
  }
  return std::min(adaptive_speed_mps, std::max(0.5, 1.25 * gt_speed));
}

std::pair<double, double> lightglue_trust_gain(
  double inlier_ratio,
  double residual_m,
  double match_confidence,
  double confidence_weight,
  double blend_gain_min,
  double blend_gain_max)
{
  const double ratio_term = clamp_value(inlier_ratio, 0.0, 1.0);
  const double confidence_term = clamp_value(match_confidence / 0.30, 0.0, 1.0);
  const double weight_term = clamp_value(confidence_weight, 0.05, 1.0);
  const double residual_term = 1.0 / (1.0 + std::max(0.0, residual_m) / 45.0);
  const double trust_factor = (0.55 * confidence_term + 0.35 * ratio_term + 0.10 * weight_term) * residual_term;
  const double max_gain = std::min(blend_gain_max, 0.40);
  const double min_gain = std::max(blend_gain_min, std::min(0.08, 0.30 * max_gain));
  return {clamp_value(trust_factor * max_gain, min_gain, max_gain), trust_factor};
}

std::tuple<double, double, double> adaptive_lightglue_gates(
  const VideoTestingOptions & options,
  double base_scale_half_range,
  bool cpu_mode,
  double match_confidence,
  double inlier_ratio,
  double correction_age_s,
  double vo_dist_m)
{
  const double quality = clamp_value(0.60 * match_confidence + 0.40 * inlier_ratio, 0.0, 1.0);
  const double freshness = clamp_value(1.0 - correction_age_s / std::max(options.lightglue_max_result_age_s, 1e-6), 0.0, 1.0);
  const double trust = clamp_value(0.70 * quality + 0.30 * freshness, 0.0, 1.0);
  double mahal_scale = 0.80 + 0.90 * trust;
  if (cpu_mode) {
    mahal_scale *= 1.15;
  }
  const double adaptive_mahal_gate = std::max(1.0, options.lightglue_mahalanobis_gate * mahal_scale);
  const double vo_dist_factor = clamp_value((vo_dist_m - 1.0) / 6.0, 0.0, 1.0);
  const double half_scale = 0.75 + 0.50 * trust + 0.40 * (1.0 - vo_dist_factor) + (cpu_mode ? 0.10 : 0.0);
  const double adaptive_half = std::max(0.20, base_scale_half_range * half_scale);
  return {adaptive_mahal_gate, std::max(0.10, 1.0 - 1.70 * adaptive_half), 1.0 + adaptive_half};
}

double adaptive_confidence_gate(
  const VideoTestingOptions & options,
  double base_gate,
  double correction_age_s,
  int miss_streak,
  const VisionMeasurement & measurement)
{
  const double freshness = clamp_value(1.0 - correction_age_s / std::max(options.lightglue_max_result_age_s, 1e-6), 0.0, 1.0);
  const double miss_relax = clamp_value(static_cast<double>(miss_streak) / 8.0, 0.0, 1.0);
  const double quality = clamp_value(
    0.50 * measurement.inlier_ratio + 0.25 * measurement.match_confidence +
    0.25 * clamp_value(static_cast<double>(measurement.inlier_count) / 20.0, 0.0, 1.0),
    0.0,
    1.0);
  double gate = base_gate;
  gate *= 0.80 + 0.30 * freshness;
  gate *= 1.00 - 0.35 * miss_relax;
  gate *= 1.00 - 0.20 * quality;
  if (measurement.failsafe_mode) {
    gate *= 0.85;
  }
  return clamp_value(gate, 0.05, 0.95);
}

void write_csv_header(std::ostream & stream)
{
  stream << "time_s,est_x_m,est_y_m,gt_x_m,gt_y_m,error_m,heading_gt_deg,heading_gt_from_srt,"
         << "heading_est_deg,heading_used_deg,delta_heading_deg,dir_error_deg,flow_du_px,flow_dv_px,"
         << "flow_std_px,flow_body_x_m,flow_body_y_m,flow_enu_x_m,flow_enu_y_m,flow_zupt,learned_scale,"
         << "sign_swap,sign_sx,sign_sy,lightglue_event,lightglue_capture_frame,lightglue_match_count,"
         << "lightglue_inlier_count,lightglue_confidence,lightglue_residual_m,lightglue_apply_x_m,"
         << "lightglue_apply_y_m,lightglue_error_before_m,lightglue_error_after_m,lightglue_reject_reason\n";
}

void write_csv_row(std::ostream & stream, const MetricsRow & row)
{
  stream << row.time_s << ',' << row.est_x_m << ',' << row.est_y_m << ',' << row.gt_x_m << ',' << row.gt_y_m << ','
         << row.error_m << ',' << row.heading_gt_deg << ',' << row.heading_gt_from_srt << ',' << row.heading_est_deg
         << ',' << row.heading_used_deg << ',' << row.delta_heading_deg << ',' << row.dir_error_deg << ','
         << row.flow_du_px << ',' << row.flow_dv_px << ',' << row.flow_std_px << ',' << row.flow_body_x_m << ','
         << row.flow_body_y_m << ',' << row.flow_enu_x_m << ',' << row.flow_enu_y_m << ',' << row.flow_zupt << ','
         << row.learned_scale << ',' << row.sign_swap << ',' << row.sign_sx << ',' << row.sign_sy << ','
         << csv_escape(row.lightglue_event) << ',' << row.lightglue_capture_frame << ',' << row.lightglue_match_count
         << ',' << row.lightglue_inlier_count << ',' << row.lightglue_confidence << ',' << row.lightglue_residual_m
         << ',' << row.lightglue_apply_x_m << ',' << row.lightglue_apply_y_m << ',' << row.lightglue_error_before_m
         << ',' << row.lightglue_error_after_m << ',' << csv_escape(row.lightglue_reject_reason) << '\n';
}

class CsvMetricsWriter
{
public:
  explicit CsvMetricsWriter(const std::string & path)
  {
    if (path.empty()) {
      return;
    }
    std::filesystem::path out_path(path);
    if (out_path.has_parent_path()) {
      std::filesystem::create_directories(out_path.parent_path());
    }
    file_.open(out_path);
    if (!file_) {
      throw std::runtime_error("Failed to open CSV output: " + path);
    }
    file_ << std::setprecision(17);
    write_csv_header(file_);
  }

  bool enabled() const { return file_.is_open(); }

  void write(const MetricsRow & row)
  {
    if (file_) {
      write_csv_row(file_, row);
    }
  }

  void close()
  {
    if (file_) {
      file_.flush();
      file_.close();
    }
  }

private:
  std::ofstream file_;
};

void validate_options(const VideoTestingOptions & options)
{
  if (options.video_path.empty()) {
    throw std::invalid_argument("--video is required");
  }
  if (options.srt_path.empty() && !options.decode_only && !options.klt_only) {
    throw std::invalid_argument("--srt is required unless --decode-only or --klt-only is used");
  }
  if (options.frame_source != "opencv" && options.frame_source != "gst-nvdec" &&
    options.frame_source != "gst-nvdec-gray" && options.frame_source != "image-sequence" &&
    options.frame_source != "raw-gray")
  {
    throw std::invalid_argument("frame_source must be opencv, gst-nvdec, gst-nvdec-gray, image-sequence, or raw-gray");
  }
  if (options.camera_fov_deg <= 1.0 || options.camera_fov_deg >= 179.0) {
    throw std::invalid_argument("camera_fov_deg must be in (1, 179)");
  }
  if (options.image_sequence_fps <= 0.0) {
    throw std::invalid_argument("image_sequence_fps must be positive");
  }
  if (options.correction_interval_s <= 0.0) {
    throw std::invalid_argument("correction_interval_s must be positive");
  }
  if (options.enable_correction && !options.decode_only && !options.klt_only && options.map_path.empty()) {
    throw std::invalid_argument("enable_correction requires map_path");
  }
  if (options.show_map_window && options.map_path.empty()) {
    throw std::invalid_argument("show_map_window requires map_path");
  }
  if (options.lightglue_window_size_m <= 0.0) {
    throw std::invalid_argument("lightglue_window_size_m must be positive");
  }
  if (options.lightglue_map_roi_size_px < 256) {
    throw std::invalid_argument("lightglue_map_roi_size_px must be >= 256");
  }
  if (options.lightglue_patch_size_px.has_value() && *options.lightglue_patch_size_px < 128) {
    throw std::invalid_argument("lightglue_patch_size_px must be >= 128");
  }
  if (options.lightglue_confidence_gate.has_value() &&
    (*options.lightglue_confidence_gate < 0.0 || *options.lightglue_confidence_gate > 1.0))
  {
    throw std::invalid_argument("lightglue_confidence_gate must be in [0, 1]");
  }
  if (options.lightglue_jump_gate_m <= 0.0 || options.lightglue_residual_gate_m <= 0.0 ||
    options.lightglue_mahalanobis_gate <= 0.0)
  {
    throw std::invalid_argument("LightGlue gates must be positive");
  }
  if (options.lightglue_meas_scale_min < 0.0 || options.lightglue_meas_scale_max <= 0.0 ||
    options.lightglue_meas_scale_min > options.lightglue_meas_scale_max)
  {
    throw std::invalid_argument("LightGlue measurement scale bounds are invalid");
  }
  if (options.lightglue_blend_gain_min < 0.0 || options.lightglue_blend_gain_min > 1.0 ||
    options.lightglue_blend_gain_max < 0.0 || options.lightglue_blend_gain_max > 1.0 ||
    options.lightglue_blend_gain_min > options.lightglue_blend_gain_max)
  {
    throw std::invalid_argument("LightGlue blend gain bounds are invalid");
  }
  if (options.lightglue_max_yaw_hypotheses < 1 || options.lightglue_max_yaw_hypotheses > 8 ||
    options.lightglue_hypotheses_per_job < 1 || options.lightglue_hypotheses_per_job > 8)
  {
    throw std::invalid_argument("LightGlue yaw hypothesis counts must be in [1, 8]");
  }
  if (options.lightglue_gpu_target_duty_cycle < 0.05 || options.lightglue_gpu_target_duty_cycle > 0.95 ||
    options.lightglue_gpu_min_interval_s <= 0.0 || options.lightglue_gpu_max_interval_s <= 0.0 ||
    options.lightglue_gpu_min_interval_s > options.lightglue_gpu_max_interval_s)
  {
    throw std::invalid_argument("LightGlue GPU interval/duty settings are invalid");
  }
  if (options.lightglue_max_result_age_s <= 0.0 || options.srt_scale_assist_alpha <= 0.0 ||
    options.srt_scale_assist_alpha > 1.0 || options.vio_rate_hz <= 0.0)
  {
    throw std::invalid_argument("result age, SRT scale alpha, and VIO rate must be valid");
  }
  if (options.async_decode_queue_size < 1) {
    throw std::invalid_argument("async_decode_queue_size must be >= 1");
  }
  if (options.frame_source == "raw-gray" &&
    (options.raw_width_px <= 0 || options.raw_height_px <= 0 || options.raw_fps <= 0.0))
  {
    throw std::invalid_argument("raw-gray frame source requires --raw-width, --raw-height, and --raw-fps");
  }
  if (options.flow_max_side_px < 0 || (options.flow_max_side_px > 0 && options.flow_max_side_px < 256)) {
    throw std::invalid_argument("flow_max_side_px must be 0 or >= 256");
  }
}

}  // namespace

std::unique_ptr<FrameSource> make_video_test_frame_source(const VideoTestingOptions & options)
{
  if (options.frame_source == "opencv") {
    return std::make_unique<CvCaptureFrameSource>(
      options.video_path, "opencv", -1, options.enable_async_decode, options.async_decode_queue_size);
  }
  if (options.frame_source == "gst-nvdec") {
    const std::string pipeline = options.gst_pipeline.empty() ? default_nvdec_pipeline_bgr(options.video_path) : options.gst_pipeline;
    return std::make_unique<GStreamerNvdecFrameSource>(
      pipeline, "BGR", options.enable_async_decode, options.async_decode_queue_size);
  }
  if (options.frame_source == "gst-nvdec-gray") {
    const std::string pipeline = options.gst_pipeline.empty() ? default_nvdec_pipeline_gray(options.video_path) : options.gst_pipeline;
    return std::make_unique<GStreamerNvdecFrameSource>(
      pipeline, "GRAY8", options.enable_async_decode, options.async_decode_queue_size);
  }
  if (options.frame_source == "image-sequence") {
    return std::make_unique<PredecodedImageSequenceFrameSource>(options.video_path, options.image_sequence_fps);
  }
  if (options.frame_source == "raw-gray") {
    return std::make_unique<RawGrayFrameSource>(
      options.video_path, options.raw_width_px, options.raw_height_px, options.raw_fps);
  }
  throw std::invalid_argument("unknown frame_source: " + options.frame_source);
}

std::vector<GroundTruthSample> parse_dji_srt(const std::string & srt_path)
{
  if (!std::filesystem::exists(srt_path)) {
    throw std::runtime_error("SRT file not found: " + srt_path);
  }
  const std::string content = read_text_file(srt_path);
  const std::regex block_re("\\r?\\n\\s*\\r?\\n");
  std::sregex_token_iterator it(content.begin(), content.end(), block_re, -1);
  std::sregex_token_iterator end;

  const std::regex time_re("(\\d{2}):(\\d{2}):(\\d{2}),(\\d{3})");
  const auto icase = std::regex_constants::icase;
  const std::vector<std::regex> lat_res{std::regex("latitude\\s*[:=]\\s*([+-]?\\d+(?:\\.\\d+)?)", icase)};
  const std::vector<std::regex> lon_res{std::regex("longitude\\s*[:=]\\s*([+-]?\\d+(?:\\.\\d+)?)", icase)};
  const std::vector<std::regex> rel_alt_res{std::regex("rel_alt\\s*[:=]\\s*([+-]?\\d+(?:\\.\\d+)?)", icase)};
  const std::vector<std::regex> heading_res{
    std::regex("\\bheading\\s*[:=]\\s*([+-]?\\d+(?:\\.\\d+)?)", icase),
    std::regex("\\bcompass_heading\\s*[:=]\\s*([+-]?\\d+(?:\\.\\d+)?)", icase),
    std::regex("\\bdrone_yaw\\s*[:=]\\s*([+-]?\\d+(?:\\.\\d+)?)", icase),
    std::regex("\\byaw\\s*[:=]\\s*([+-]?\\d+(?:\\.\\d+)?)", icase)};

  std::vector<GroundTruthSample> samples;
  for (; it != end; ++it) {
    const std::string block = it->str();
    std::smatch time_match;
    if (!std::regex_search(block, time_match, time_re)) {
      continue;
    }
    const auto lat = first_number_match(block, lat_res);
    const auto lon = first_number_match(block, lon_res);
    if (!lat.has_value() || !lon.has_value()) {
      continue;
    }
    GroundTruthSample sample;
    sample.timestamp_s = parse_srt_time_s(time_match);
    sample.latitude_deg = *lat;
    sample.longitude_deg = *lon;
    sample.heading_deg = first_number_match(block, heading_res);
    sample.rel_alt_m = first_number_match(block, rel_alt_res);
    samples.push_back(sample);
  }
  if (samples.empty()) {
    throw std::runtime_error("No valid latitude/longitude entries found in SRT: " + srt_path);
  }
  return samples;
}

Vector2 latlon_to_xy(double latitude_deg, double longitude_deg, double latitude_ref_deg, double longitude_ref_deg)
{
  const double deg_to_rad = kPi / 180.0;
  const double dx = (longitude_deg - longitude_ref_deg) * deg_to_rad * kVideoTestEarthRadiusM * std::cos(latitude_ref_deg * deg_to_rad);
  const double dy = (latitude_deg - latitude_ref_deg) * deg_to_rad * kVideoTestEarthRadiusM;
  return Vector2(dx, dy);
}

double wrap_angle_rad(double angle_rad)
{
  double wrapped = std::fmod(angle_rad + kPi, 2.0 * kPi);
  if (wrapped < 0.0) {
    wrapped += 2.0 * kPi;
  }
  return wrapped - kPi;
}

double north_cw_heading_deg_to_math_rad(double heading_deg)
{
  return wrap_angle_rad((90.0 - heading_deg) * kPi / 180.0);
}

std::pair<Vector2, Vector2> flow_to_enu(
  double du_px,
  double dv_px,
  double heading_rad,
  double scale_m_per_px,
  const FlowSignConfig & sign_cfg,
  bool heading_is_north_cw)
{
  double body_x_m = 0.0;
  double body_y_m = 0.0;
  if (sign_cfg.swap) {
    body_x_m = static_cast<double>(sign_cfg.sx) * dv_px * scale_m_per_px;
    body_y_m = static_cast<double>(sign_cfg.sy) * (-du_px) * scale_m_per_px;
  } else {
    body_x_m = static_cast<double>(sign_cfg.sx) * du_px * scale_m_per_px;
    body_y_m = static_cast<double>(sign_cfg.sy) * dv_px * scale_m_per_px;
  }

  Vector2 enu;
  if (heading_is_north_cw) {
    const double s = std::sin(heading_rad);
    const double c = std::cos(heading_rad);
    enu = Vector2(body_x_m * s + body_y_m * c, body_x_m * c - body_y_m * s);
  } else {
    const double c = std::cos(heading_rad);
    const double s = std::sin(heading_rad);
    enu = Vector2(c * body_x_m - s * body_y_m, s * body_x_m + c * body_y_m);
  }
  return {enu, Vector2(body_x_m, body_y_m)};
}

Vector2 project_xy_onto_direction(const Vector2 & vector_xy, const Vector2 & direction_xy)
{
  const double norm = direction_xy.norm();
  if (norm <= kEpsilon) {
    return vector_xy;
  }
  const Vector2 unit = direction_xy / norm;
  return unit * vector_xy.dot(unit);
}

Vector2 project_xy_forward_onto_direction(const Vector2 & vector_xy, const Vector2 & direction_xy)
{
  const double norm = direction_xy.norm();
  if (norm <= kEpsilon) {
    return vector_xy;
  }
  const Vector2 unit = direction_xy / norm;
  const double mag = vector_xy.dot(unit);
  if (mag <= 0.0) {
    return Vector2::Zero();
  }
  return unit * mag;
}

cv::Mat center_crop_square(const cv::Mat & frame_bgr, int output_size_px)
{
  cv::Mat output;
  center_crop_square(frame_bgr, output_size_px, output);
  return output;
}

void center_crop_square(const cv::Mat & frame_bgr, int output_size_px, cv::Mat & output)
{
  if (output_size_px < 32) {
    throw std::invalid_argument("output_size_px must be >= 32");
  }
  if (frame_bgr.empty()) {
    throw std::invalid_argument("frame_bgr must not be empty");
  }
  const int side = std::min(frame_bgr.cols, frame_bgr.rows);
  const int x = std::max(0, (frame_bgr.cols - side) / 2);
  const int y = std::max(0, (frame_bgr.rows - side) / 2);
  const cv::Mat cropped = frame_bgr(cv::Rect(x, y, side, side));
  if (cropped.cols != output_size_px || cropped.rows != output_size_px) {
    cv::resize(cropped, output, cv::Size(output_size_px, output_size_px), 0.0, 0.0, cv::INTER_AREA);
  } else {
    cropped.copyTo(output);
  }
}

void center_crop_square_gray(
  const cv::Mat & frame,
  int output_size_px,
  cv::Mat & bgr_scratch,
  cv::Mat & gray_output)
{
  if (frame.empty()) {
    throw std::invalid_argument("frame must not be empty");
  }
  const int side = std::min(frame.cols, frame.rows);
  const int x = std::max(0, (frame.cols - side) / 2);
  const int y = std::max(0, (frame.rows - side) / 2);
  const cv::Mat cropped = frame(cv::Rect(x, y, side, side));
  cv::Mat resized_or_cropped;
  if (cropped.cols != output_size_px || cropped.rows != output_size_px) {
    cv::resize(cropped, bgr_scratch, cv::Size(output_size_px, output_size_px), 0.0, 0.0, cv::INTER_AREA);
    resized_or_cropped = bgr_scratch;
  } else {
    resized_or_cropped = cropped;
  }
  if (resized_or_cropped.channels() == 1) {
    resized_or_cropped.copyTo(gray_output);
  } else {
    cv::cvtColor(resized_or_cropped, gray_output, cv::COLOR_BGR2GRAY);
  }
}

VisionOnlyTracker::VisionOnlyTracker(
  double focal_length_px_in,
  double test_altitude_m,
  int max_features,
  int min_features,
  bool use_clahe,
  double scale_min,
  double scale_max,
  int flow_max_side_px)
: altitude_m(test_altitude_m),
  focal_length_px(focal_length_px_in),
  base_meters_per_pixel(test_altitude_m / focal_length_px_in),
  scale_min_(scale_min),
  scale_max_(scale_max),
  max_features_(max_features),
  min_features_(min_features),
  use_clahe_(use_clahe),
  flow_max_side_px_(flow_max_side_px)
{
  if (focal_length_px_in <= 1e-6) {
    throw std::invalid_argument("focal_length_px must be positive");
  }
  if (test_altitude_m <= 1e-6) {
    throw std::invalid_argument("test_altitude_m must be positive");
  }
  if (flow_max_side_px_ < 0) {
    throw std::invalid_argument("flow_max_side_px must be non-negative");
  }
  prev_points_.reserve(static_cast<std::size_t>(max_features_));
  curr_points_.reserve(static_cast<std::size_t>(max_features_));
  filtered_points_.reserve(static_cast<std::size_t>(max_features_));
  status_buffer_.reserve(static_cast<std::size_t>(max_features_));
  error_buffer_.reserve(static_cast<std::size_t>(max_features_));
  dx_buffer_.reserve(static_cast<std::size_t>(max_features_));
  dy_buffer_.reserve(static_cast<std::size_t>(max_features_));
  if (use_clahe_) {
    clahe_ = cv::createCLAHE(2.0, cv::Size(8, 8));
  }
}

double VisionOnlyTracker::meters_per_pixel() const
{
  return base_meters_per_pixel * learned_scale;
}

std::tuple<bool, double, double, double> VisionOnlyTracker::estimate_flow_step(const cv::Mat & frame_bgr)
{
  const auto total_start = timing_enabled_ ? Clock::now() : Clock::time_point{};
  last_timing_ms_ = FlowTiming{};
  cv::Mat gray;
  cv::Mat * current_proc_owner = nullptr;
  auto stage_start = timing_enabled_ ? Clock::now() : Clock::time_point{};
  if (frame_bgr.channels() == 1) {
    gray = frame_bgr;
  } else {
    cv::cvtColor(frame_bgr, gray_buffer_, cv::COLOR_BGR2GRAY);
    gray = gray_buffer_;
  }
  if (timing_enabled_) {
    last_timing_ms_.grayscale_ms = elapsed_ms(stage_start);
  }

  double flow_scale = 1.0;
  if (flow_max_side_px_ > 0) {
    const int max_side = std::max(gray.cols, gray.rows);
    if (max_side > flow_max_side_px_) {
      flow_scale = static_cast<double>(flow_max_side_px_) / static_cast<double>(max_side);
      const int resized_cols = std::max(1, static_cast<int>(std::round(static_cast<double>(gray.cols) * flow_scale)));
      const int resized_rows = std::max(1, static_cast<int>(std::round(static_cast<double>(gray.rows) * flow_scale)));
      stage_start = timing_enabled_ ? Clock::now() : Clock::time_point{};
      cv::resize(gray, flow_gray_buffer_, cv::Size(resized_cols, resized_rows), 0.0, 0.0, cv::INTER_AREA);
      gray = flow_gray_buffer_;
      current_proc_owner = &flow_gray_buffer_;
      if (timing_enabled_) {
        last_timing_ms_.resize_ms = elapsed_ms(stage_start);
      }
    }
  }

  cv::Mat gray_proc;
  if (clahe_) {
    stage_start = timing_enabled_ ? Clock::now() : Clock::time_point{};
    clahe_->apply(gray, gray_proc_buffer_);
    gray_proc = gray_proc_buffer_;
    current_proc_owner = &gray_proc_buffer_;
    if (timing_enabled_) {
      last_timing_ms_.clahe_ms = elapsed_ms(stage_start);
    }
  } else {
    gray_proc = gray;
    if (current_proc_owner == nullptr && frame_bgr.channels() != 1) {
      current_proc_owner = &gray_buffer_;
    }
  }
  auto store_current_as_previous = [&]() {
    if (current_proc_owner != nullptr) {
      std::swap(prev_gray_, *current_proc_owner);
    } else {
      gray_proc.copyTo(prev_gray_);
    }
  };
  auto finish = [&]() {
    if (timing_enabled_) {
      last_timing_ms_.total_ms = elapsed_ms(total_start);
    }
  };

  if (prev_gray_.empty()) {
    store_current_as_previous();
    finish();
    return {false, 0.0, 0.0, 0.0};
  }

  if (prev_points_.size() < static_cast<std::size_t>(min_features_)) {
    stage_start = timing_enabled_ ? Clock::now() : Clock::time_point{};
    cv::goodFeaturesToTrack(prev_gray_, prev_points_, max_features_, 0.01, 7.0, cv::Mat(), 7);
    if (timing_enabled_) {
      last_timing_ms_.feature_ms = elapsed_ms(stage_start);
    }
  }
  if (prev_points_.size() < static_cast<std::size_t>(min_features_)) {
    store_current_as_previous();
    finish();
    return {false, 0.0, 0.0, 0.0};
  }

  curr_points_.clear();
  status_buffer_.clear();
  error_buffer_.clear();
  stage_start = timing_enabled_ ? Clock::now() : Clock::time_point{};
  cv::calcOpticalFlowPyrLK(
    prev_gray_,
    gray_proc,
    prev_points_,
    curr_points_,
    status_buffer_,
    error_buffer_,
    lk_win_size_,
    lk_max_level_,
    lk_criteria_);
  if (timing_enabled_) {
    last_timing_ms_.lk_ms = elapsed_ms(stage_start);
  }

  stage_start = timing_enabled_ ? Clock::now() : Clock::time_point{};
  filtered_points_.clear();
  dx_buffer_.clear();
  dy_buffer_.clear();
  const std::size_t count = std::min(status_buffer_.size(), std::min(curr_points_.size(), prev_points_.size()));
  for (std::size_t i = 0; i < count; ++i) {
    if (status_buffer_[i]) {
      filtered_points_.push_back(curr_points_[i]);
      dx_buffer_.push_back(static_cast<double>(curr_points_[i].x - prev_points_[i].x));
      dy_buffer_.push_back(static_cast<double>(curr_points_[i].y - prev_points_[i].y));
    }
  }

  if (filtered_points_.size() < static_cast<std::size_t>(min_features_)) {
    prev_points_.clear();
    store_current_as_previous();
    if (timing_enabled_) {
      last_timing_ms_.stats_ms = elapsed_ms(stage_start);
    }
    finish();
    return {false, 0.0, 0.0, 0.0};
  }

  const double du_work_px = median_in_place(dx_buffer_);
  const double dv_work_px = median_in_place(dy_buffer_);
  double var_x = 0.0;
  double var_y = 0.0;
  for (std::size_t i = 0; i < dx_buffer_.size(); ++i) {
    var_x += (dx_buffer_[i] - du_work_px) * (dx_buffer_[i] - du_work_px);
    var_y += (dy_buffer_[i] - dv_work_px) * (dy_buffer_[i] - dv_work_px);
  }
  var_x /= static_cast<double>(std::max<std::size_t>(dx_buffer_.size(), 1));
  var_y /= static_cast<double>(std::max<std::size_t>(dy_buffer_.size(), 1));
  double flow_std_px = std::sqrt(var_x + var_y);
  const double inv_flow_scale = flow_scale > kEpsilon ? 1.0 / flow_scale : 1.0;
  const double du = du_work_px * inv_flow_scale;
  const double dv = dv_work_px * inv_flow_scale;
  flow_std_px *= inv_flow_scale;
  const double shift_norm_px = std::hypot(du, dv);
  if (timing_enabled_) {
    last_timing_ms_.stats_ms = elapsed_ms(stage_start);
  }

  prev_points_.swap(filtered_points_);
  store_current_as_previous();
  finish();
  if (flow_std_px > kDefaultMaxFlowStdPx || shift_norm_px > kDefaultMaxFlowShiftPx) {
    prev_points_.clear();
    return {false, 0.0, 0.0, flow_std_px};
  }
  return {true, du, dv, flow_std_px};
}

std::tuple<double, double, FlowStepDiagnostics> VisionOnlyTracker::integrate_flow_step(
  double du_px,
  double dv_px,
  double flow_std_px,
  double frame_dt_s,
  double heading_rad,
  const FlowSignConfig & sign_cfg,
  bool heading_is_north_cw)
{
  const double dt_s = std::max(frame_dt_s, 1e-6);
  auto [enu, body] = flow_to_enu(du_px, dv_px, heading_rad, meters_per_pixel(), sign_cfg, heading_is_north_cw);
  std::string reject_reason;
  if (flow_std_px > kDefaultMaxFlowStdPx || enu.norm() > kDefaultMaxFlowStepM) {
    reject_reason = "flow_quality_gate";
    enu = Vector2::Zero();
    body = Vector2::Zero();
  }
  const bool zupt_applied = body.norm() / dt_s < 0.15;
  if (zupt_applied) {
    enu = Vector2::Zero();
    body = Vector2::Zero();
  }
  pos_x_m += enu.x();
  pos_y_m += enu.y();
  const double flow_noise_m = std::max(meters_per_pixel(), flow_std_px * meters_per_pixel());
  const double process_noise_m2 = std::pow(0.15 * enu.norm() + flow_noise_m, 2.0);
  inflate_position_covariance(process_noise_m2);

  FlowStepDiagnostics diag;
  diag.valid = true;
  diag.du_px = du_px;
  diag.dv_px = dv_px;
  diag.flow_std_px = flow_std_px;
  diag.flow_body_x_m = body.x();
  diag.flow_body_y_m = body.y();
  diag.flow_enu_x_m = enu.x();
  diag.flow_enu_y_m = enu.y();
  diag.heading_used_rad = heading_rad;
  diag.zupt_applied = zupt_applied;
  diag.learned_scale = learned_scale;
  diag.reject_reason = reject_reason;
  return {pos_x_m, pos_y_m, diag};
}

std::optional<double> VisionOnlyTracker::update_scale_from_lightglue(
  const Vector2 & ai_true_xy,
  const Vector2 & saved_est_xy,
  std::optional<Vector2> saved_flow_odom_xy,
  bool allow_update,
  double alpha,
  double max_update_delta)
{
  const Vector2 scale_reference_xy = saved_flow_odom_xy.value_or(saved_est_xy);
  std::optional<double> observed_scale;
  if (prev_ai_x_ && prev_ai_y_ && prev_saved_est_x_ && prev_saved_est_y_) {
    const double ai_dist = std::hypot(ai_true_xy.x() - *prev_ai_x_, ai_true_xy.y() - *prev_ai_y_);
    const double vo_dist =
      std::hypot(scale_reference_xy.x() - *prev_saved_est_x_, scale_reference_xy.y() - *prev_saved_est_y_);
    if (allow_update && vo_dist > 5.0 && ai_dist > 5.0) {
      observed_scale = learned_scale * ai_dist / std::max(vo_dist, 1e-6);
      if (*observed_scale > 0.35 && *observed_scale < 1.20) {
        update_scale_from_measurement(*observed_scale, alpha, std::max(0.40, scale_min_), 1.20, 15, max_update_delta);
      } else {
        observed_scale.reset();
      }
    }
  }
  prev_ai_x_ = ai_true_xy.x();
  prev_ai_y_ = ai_true_xy.y();
  prev_saved_est_x_ = scale_reference_xy.x();
  prev_saved_est_y_ = scale_reference_xy.y();
  return observed_scale;
}

double VisionOnlyTracker::update_scale_from_measurement(
  double scale_measurement,
  std::optional<double> alpha,
  std::optional<double> scale_min,
  std::optional<double> scale_max,
  int history_size,
  std::optional<double> max_update_delta)
{
  double lo = scale_min.value_or(scale_min_);
  double hi = scale_max.value_or(scale_max_);
  if (lo > hi) {
    std::swap(lo, hi);
  }
  scale_history_.push_back(clamp_value(scale_measurement, lo, hi));
  while (scale_history_.size() > static_cast<std::size_t>(std::max(1, history_size))) {
    scale_history_.erase(scale_history_.begin());
  }
  const double target_scale = median(scale_history_);
  if (!alpha.has_value()) {
    learned_scale = target_scale;
  } else {
    double delta = (target_scale - learned_scale) * clamp_value(*alpha, 0.0, 1.0);
    if (max_update_delta.has_value()) {
      delta = clamp_value(delta, -*max_update_delta, *max_update_delta);
    }
    learned_scale = clamp_value(learned_scale + delta, lo, hi);
  }
  return learned_scale;
}

void VisionOnlyTracker::set_absolute_position(double x_m, double y_m)
{
  pos_x_m = x_m;
  pos_y_m = y_m;
}

void VisionOnlyTracker::inflate_position_covariance(double process_noise_m2)
{
  const double q = std::max(0.0, process_noise_m2);
  pos_cov_xx_m2 = std::min(1e4, pos_cov_xx_m2 + q);
  pos_cov_yy_m2 = std::min(1e4, pos_cov_yy_m2 + q);
}

void VisionOnlyTracker::contract_position_covariance(double gain)
{
  const double scale = clamp_value(1.0 - clamp_value(gain, 0.0, 1.0), 0.05, 1.0);
  pos_cov_xx_m2 = std::max(0.05, pos_cov_xx_m2 * scale);
  pos_cov_yy_m2 = std::max(0.05, pos_cov_yy_m2 * scale);
}

VideoTestResult run_vision_only_video_test(
  const VideoTestingOptions & options,
  std::unique_ptr<VisualLocalizer> visual_localizer)
{
  validate_options(options);
  cv::setUseOptimized(true);
  const auto run_start = Clock::now();
  const bool profiling_enabled = options.profile || !options.profile_output_path.empty();
  VideoTestResult::Profile profile;
  profile.enabled = profiling_enabled;
  auto frame_source = make_video_test_frame_source(options);
  frame_source->open();
  const FrameSourceMetadata source_meta = frame_source->metadata();
  const double width = static_cast<double>(source_meta.width);
  const double height = static_cast<double>(source_meta.height);
  const double fps = source_meta.fps;
  const double source_frame_count = static_cast<double>(source_meta.frame_count);
  if (width <= 0.0 || height <= 0.0 || fps <= 0.0) {
    throw std::runtime_error("Invalid video metadata (width/height/fps)");
  }
  const double frame_dt_nominal_s = 1.0 / fps;
  const double focal_length_px = (width * 0.5) / std::tan(options.camera_fov_deg * 0.5 * kPi / 180.0);

  SystemConfig lightglue_config;
  int patch_size_px = options.lightglue_patch_size_px.value_or(lightglue_config.lightglue.image_size_px);
  double confidence_gate = options.lightglue_confidence_gate.value_or(lightglue_config.lightglue.min_confidence);
  bool cpu_lightglue_mode = false;
  int max_yaw_hypotheses = options.lightglue_max_yaw_hypotheses;
  int hypotheses_per_job = options.lightglue_hypotheses_per_job;
  if (options.enable_correction) {
    lightglue_config = SystemConfig::from_yaml(options.lightglue_config_yaml);
    patch_size_px = options.lightglue_patch_size_px.value_or(lightglue_config.lightglue.image_size_px);
    confidence_gate = options.lightglue_confidence_gate.value_or(lightglue_config.lightglue.min_confidence);
    cpu_lightglue_mode = !lightglue_config.lightglue.require_cuda || lightglue_config.lightglue.device == "cpu";
    if (cpu_lightglue_mode) {
      patch_size_px = std::min(patch_size_px, 192);
      max_yaw_hypotheses = std::min(max_yaw_hypotheses, 3);
      hypotheses_per_job = 1;
    }
    lightglue_config.lightglue.image_size_px = patch_size_px;
  }

  VisionOnlyTracker tracker(
    focal_length_px,
    options.test_altitude_m,
    200,
    12,
    options.use_clahe,
    0.42,
    0.95,
    options.flow_max_side_px);
  tracker.set_timing_enabled(profiling_enabled);
  const double camera_gsd_m_per_px = tracker.altitude_m / std::max(focal_length_px, 1e-6);
  const double flow_update_interval_s = 1.0 / options.vio_rate_hz;
  const double base_scale_half_range =
    std::max(1.0 - options.lightglue_meas_scale_min, options.lightglue_meas_scale_max - 1.0);

  if (options.decode_only || options.klt_only) {
    std::vector<double> klt_times_ms;
    klt_times_ms.reserve(static_cast<std::size_t>(std::max(1.0, source_frame_count)));
    std::optional<double> last_flow_update_t;
    int frame_idx = 0;
    Frame source_frame;
    while (true) {
      if (options.max_frames.has_value() && frame_idx >= *options.max_frames) {
        break;
      }
      if (!frame_source->read(source_frame)) {
        break;
      }
      ++profile.frames_read;
      const double video_t = source_frame.timestamp_s;
      if (options.klt_only) {
        const bool run_flow_update =
          !last_flow_update_t.has_value() || (video_t - *last_flow_update_t) >= (flow_update_interval_s - 1e-9);
        if (run_flow_update) {
          const auto klt_start = Clock::now();
          auto [flow_valid, du_px, dv_px, flow_std_px] = tracker.estimate_flow_step(source_frame.image);
          (void)flow_valid;
          (void)du_px;
          (void)dv_px;
          (void)flow_std_px;
          const double frame_klt_ms = elapsed_ms(klt_start);
          klt_times_ms.push_back(frame_klt_ms);
          if (profiling_enabled) {
            const auto & flow_timing = tracker.last_timing_ms();
            profile.klt_total_ms += flow_timing.total_ms > 0.0 ? flow_timing.total_ms : frame_klt_ms;
            profile.klt_grayscale_ms += flow_timing.grayscale_ms;
            profile.klt_resize_ms += flow_timing.resize_ms;
            profile.klt_clahe_ms += flow_timing.clahe_ms;
            profile.klt_feature_ms += flow_timing.feature_ms;
            profile.klt_lk_ms += flow_timing.lk_ms;
            profile.klt_stats_ms += flow_timing.stats_ms;
            ++profile.klt_updates;
          }
          last_flow_update_t = video_t;
        }
      }
      ++frame_idx;
    }
    if (frame_idx == 0) {
      throw std::runtime_error("No frames processed");
    }

    const FrameSourceProfile source_profile = frame_source->profile();
    profile.decode_read_ms = source_profile.decode_ms;
    profile.decode_wait_ms = source_profile.wait_ms;
    profile.frame_source = source_profile;

    VideoTestResult result;
    result.frame_count = frame_idx;
    result.runtime_s = std::chrono::duration<double>(Clock::now() - run_start).count();
    result.processed_fps = static_cast<double>(frame_idx) / std::max(result.runtime_s, 1e-9);
    result.mean_klt_ms = klt_times_ms.empty() ? 0.0 :
      std::accumulate(klt_times_ms.begin(), klt_times_ms.end(), 0.0) / static_cast<double>(klt_times_ms.size());
    result.profile = profile;
    write_profile_report(options.profile_output_path, result.profile);
    write_realtime_profile_report(options.realtime_profile_output_path, result);
    return result;
  }

  const auto samples = parse_dji_srt(options.srt_path);
  const GroundTruthTrack gt = make_ground_truth_track(samples);
  GroundTruthInterpolator gt_interpolator(gt);

  std::optional<StaticGeoMap> correction_map;
  if (options.enable_correction) {
    correction_map = StaticGeoMap::from_path(options.map_path, options.map_gsd_m_per_px, true);
    if (!visual_localizer) {
      visual_localizer = std::make_unique<TensorRtLightGlueVisualLocalizer>(lightglue_config);
    }
    if (!visual_localizer->is_ready()) {
      throw std::runtime_error("LightGlue startup blocked: " + visual_localizer->startup_error());
    }
  }
  std::unique_ptr<LiveMapTrailWindow> map_window;
  if (options.show_map_window) {
    const double window_fps = options.map_window_video_fps.value_or(fps);
    map_window = std::make_unique<LiveMapTrailWindow>(
      options.map_path,
      options.map_gsd_m_per_px,
      options.map_window_name,
      options.map_max_display_size_px,
      options.map_window_video_output_path,
      window_fps);
  }

  std::unique_ptr<LightGlueWorker> lightglue_worker;
  bool has_pending_lightglue_job = false;
  if (options.enable_correction && options.enable_async_lightglue && visual_localizer) {
    lightglue_worker = std::make_unique<LightGlueWorker>(*visual_localizer);
  }
  CsvMetricsWriter csv_writer(options.csv_enabled ? options.csv_output_path : std::string{});
  std::vector<double> timestamps;
  std::vector<double> errors;
  std::vector<double> klt_times_ms;
  std::vector<double> visual_times_ms;
  const auto expected_frames = static_cast<std::size_t>(std::max(1.0, source_frame_count));
  timestamps.reserve(expected_frames);
  errors.reserve(expected_frames);
  klt_times_ms.reserve(expected_frames);
  visual_times_ms.reserve(std::max<std::size_t>(1U, expected_frames / std::max<std::size_t>(1U, static_cast<std::size_t>(fps))));
  std::vector<double> median_scratch;
  median_scratch.reserve(64);
  cv::Mat map_patch_buffer;
  cv::Mat camera_patch_bgr_buffer;
  cv::Mat camera_patch_gray_buffer;
  LightGlueSearchWorkspace sync_lightglue_workspace;

  int frame_idx = 0;
  int correction_count = 0;
  double last_correction_s = 0.0;
  std::optional<Vector2> last_accepted_correction_xy;
  std::optional<Vector2> last_accepted_capture_est_xy;
  std::optional<Vector2> last_accepted_capture_flow_odom_xy;
  std::optional<Vector2> last_reliable_correction_xy;
  std::optional<Vector2> last_correction_capture_est_xy;
  Vector2 flow_odom_xy = Vector2::Zero();
  std::deque<Vector2> lightglue_consensus_innovations;
  double dynamic_correction_interval_s =
    options.enable_async_lightglue ? std::min(options.lightglue_gpu_min_interval_s, 0.35) : options.correction_interval_s;
  std::optional<double> lightglue_smoothed_job_s;
  int yaw_hypothesis_offset = 0;
  std::optional<Vector2> active_roi_center_xy;
  std::optional<double> active_roi_width_m;
  std::optional<double> active_roi_height_m;
  int lightglue_consecutive_no_match = 0;
  double heading_est_rad = 0.0;
  FlowSignConfig active_sign_cfg;

  double last_flow_speed_mps = 0.0;
  std::optional<Vector2> last_flow_dir_enu;
  std::optional<double> last_flow_update_t;
  double latest_flow_dt_s = frame_dt_nominal_s;
  Vector2 flow_velocity_enu_mps = Vector2::Zero();
  Vector2 flow_accel_enu_mps2 = Vector2::Zero();
  std::optional<double> last_flow_velocity_update_t;

  const double blind_hover_speed_mps = 12.0;
  const int blind_hover_min_stall_updates = 2;
  int blind_hover_stall_updates = 0;
  const double blind_hover_min_step_m = 0.25;
  const double blind_hover_arm_distance_m = 25.0;
  const double blind_hover_soft_arm_distance_m = 15.0;
  const int blind_hover_soft_arm_min_no_match = 2;
  const int blind_hover_direction_ready_min_samples = 8;
  const double blind_hover_min_speed_for_history_mps = 2.0;
  const double blind_hover_low_speed_ratio = 0.55;
  const int blind_hover_low_speed_no_match_trigger = 1;
  std::deque<double> blind_hover_speed_history_mps;
  std::deque<double> blind_hover_gt_speed_history_mps;
  const Vector2 blind_hover_start_xy(tracker.pos_x_m, tracker.pos_y_m);

  double prev_true_x_m = gt.xs_m.front();
  double prev_true_y_m = gt.ys_m.front();
  double scale_window_dt_s = 0.0;
  double scale_window_gt_disp_m = 0.0;
  double scale_window_vo_disp_m = 0.0;
  int scale_window_flow_count = 0;
  double scale_window_flow_std_sum_px = 0.0;
  Frame source_frame;

  auto update_lightglue_gpu_budget = [&](double job_wall_time_s) {
    const double wall_s = std::max(1e-3, job_wall_time_s);
    if (!lightglue_smoothed_job_s.has_value()) {
      lightglue_smoothed_job_s = wall_s;
    } else {
      lightglue_smoothed_job_s = 0.8 * *lightglue_smoothed_job_s + 0.2 * wall_s;
    }
    double desired = clamp_value(*lightglue_smoothed_job_s / std::max(options.lightglue_gpu_target_duty_cycle, 1e-3),
      options.lightglue_gpu_min_interval_s, options.lightglue_gpu_max_interval_s);
    if (!last_reliable_correction_xy.has_value()) {
      desired = std::min(desired, 0.35);
    }
    dynamic_correction_interval_s = options.enable_async_lightglue ? desired : std::max(options.correction_interval_s, desired);
    if (options.verbose) {
      std::cout << ">>> LightGlue GPU budget: job=" << wall_s << "s smooth=" << *lightglue_smoothed_job_s
                << "s interval=" << dynamic_correction_interval_s << "s\n";
    }
  };

  auto should_force_roi_refresh = [&](double now_t, const Vector2 & current_xy, const Vector2 & velocity_xy) {
    if (!active_roi_center_xy || !active_roi_width_m || !active_roi_height_m) {
      return std::tuple<bool, double, double>(false, 0.0, 0.0);
    }
    const double half_w = std::max(1.0, 0.5 * *active_roi_width_m);
    const double half_h = std::max(1.0, 0.5 * *active_roi_height_m);
    const double guard_w = std::min(0.45 * half_w, std::max(2.0, 0.22 * half_w));
    const double guard_h = std::min(0.45 * half_h, std::max(2.0, 0.22 * half_h));
    const double safe_w = std::max(1.0, half_w - guard_w);
    const double safe_h = std::max(1.0, half_h - guard_h);
    const Vector2 delta_now = (current_xy - *active_roi_center_xy).cwiseAbs();
    const double norm_now = std::max(delta_now.x() / safe_w, delta_now.y() / safe_h);
    const double horizon = std::max({lightglue_config.lightglue.roi_prediction_latency_s, dynamic_correction_interval_s, 0.10});
    const Vector2 pred = current_xy + velocity_xy * horizon;
    const Vector2 delta_pred = (pred - *active_roi_center_xy).cwiseAbs();
    const double norm_pred = std::max(delta_pred.x() / safe_w, delta_pred.y() / safe_h);
    const bool rate_ok = (now_t - last_correction_s) >= std::min(options.correction_interval_s, options.lightglue_gpu_min_interval_s);
    return std::tuple<bool, double, double>(rate_ok && (norm_now >= 1.0 || norm_pred >= 1.0), norm_now, norm_pred);
  };

  auto apply_blind_propagation = [&](double video_t, double step_dt_s, const GroundTruthState & gt_state) {
    Vector2 blind_dir = options.use_gt_heading ? gt_state.direction :
      (last_flow_dir_enu.has_value() ? *last_flow_dir_enu :
      (flow_velocity_enu_mps.norm() > 1e-3 ? flow_velocity_enu_mps.normalized() : Vector2(std::cos(heading_est_rad), std::sin(heading_est_rad))));
    if (blind_dir.norm() <= 1e-6) {
      blind_dir = Vector2(std::cos(heading_est_rad), std::sin(heading_est_rad));
    }
    double adaptive_speed = blind_hover_speed_history_mps.empty() ? blind_hover_speed_mps :
      median_deque(blind_hover_speed_history_mps, median_scratch);
    if (options.use_gt_heading) {
      adaptive_speed = cap_blind_speed_with_ground_truth(
        adaptive_speed,
        std::hypot(gt_state.x_m - prev_true_x_m, gt_state.y_m - prev_true_y_m),
        frame_dt_nominal_s,
        blind_hover_gt_speed_history_mps,
        median_scratch);
    } else if (lightglue_consecutive_no_match >= 3) {
      adaptive_speed = std::max(adaptive_speed, 4.0);
    }
    adaptive_speed = clamp_value(adaptive_speed, 0.5, 25.0);
    const Vector2 blind_step = blind_dir.normalized() * (adaptive_speed * std::max(step_dt_s, 1e-3));
    tracker.set_absolute_position(tracker.pos_x_m + blind_step.x(), tracker.pos_y_m + blind_step.y());
    if (options.verbose && blind_hover_stall_updates % 5 == 0) {
      std::cout << ">>> BLIND PROPAGATION @ " << video_t << "s step=(" << blind_step.x() << "," << blind_step.y()
                << ") xy=(" << tracker.pos_x_m << "," << tracker.pos_y_m << ")\n";
    }
    return blind_step;
  };

  try {
    while (true) {
      if (options.max_frames.has_value() && frame_idx >= *options.max_frames) {
        break;
      }
      if (!frame_source->read(source_frame)) {
        break;
      }
      const cv::Mat & frame = source_frame.image;
      ++profile.frames_read;
      const auto frame_work_start = profiling_enabled ? Clock::now() : Clock::time_point{};
      double frame_gt_ms = 0.0;
      double frame_klt_ms = 0.0;
      double frame_crop_ms = 0.0;
      double frame_visual_blocking_ms = 0.0;
      double frame_csv_ms = 0.0;
      double frame_map_ms = 0.0;

      const double video_t = source_frame.timestamp_s;
      const auto gt_start = profiling_enabled ? Clock::now() : Clock::time_point{};
      const GroundTruthState gt_state = gt_interpolator.at(video_t);
      if (profiling_enabled) {
        frame_gt_ms = elapsed_ms(gt_start);
        profile.gt_interpolation_ms += frame_gt_ms;
      }
      if (options.use_gt_heading) {
        heading_est_rad = gt_state.heading_rad;
      }

      std::optional<LightGlueSearchResult> ready_lightglue_result;
      bool correction_applied_this_frame = false;
      std::string lightglue_event;
      int lightglue_capture_frame = -1;
      int lightglue_match_count = 0;
      int lightglue_inlier_count = 0;
      double lightglue_confidence = std::numeric_limits<double>::quiet_NaN();
      double lightglue_residual_m = std::numeric_limits<double>::quiet_NaN();
      double lightglue_apply_x_m = 0.0;
      double lightglue_apply_y_m = 0.0;
      double lightglue_error_before_m = std::numeric_limits<double>::quiet_NaN();
      double lightglue_error_after_m = std::numeric_limits<double>::quiet_NaN();
      std::string lightglue_reject_reason;

      if (has_pending_lightglue_job && lightglue_worker) {
        ready_lightglue_result = lightglue_worker->poll_result();
        if (ready_lightglue_result.has_value()) {
          has_pending_lightglue_job = false;
        }
      }
      if (ready_lightglue_result.has_value()) {
        visual_times_ms.push_back(ready_lightglue_result->wall_time_s * 1000.0);
        if (profiling_enabled) {
          profile.visual_search_ms += ready_lightglue_result->wall_time_s * 1000.0;
          ++profile.visual_results;
        }
        update_lightglue_gpu_budget(ready_lightglue_result->wall_time_s);
      }

      bool correction_due = false;
      bool force_roi_refresh = false;
      double roi_occ_now = 0.0;
      double roi_occ_pred = 0.0;
      if (options.enable_correction) {
        const double elapsed_since_correction_s = std::max(0.0, video_t - last_correction_s);
        correction_due = elapsed_since_correction_s >= dynamic_correction_interval_s;
        std::tie(force_roi_refresh, roi_occ_now, roi_occ_pred) =
          should_force_roi_refresh(video_t, Vector2(tracker.pos_x_m, tracker.pos_y_m), flow_velocity_enu_mps);
      }

      if (options.enable_correction && frame_idx > 0 && !has_pending_lightglue_job &&
        (correction_due || force_roi_refresh) && visual_localizer && correction_map)
      {
        const auto & sample = samples[static_cast<std::size_t>(gt_state.sample_index)];
        const double altitude_for_lightglue_m =
          (sample.rel_alt_m.has_value() && *sample.rel_alt_m > 1.0) ? *sample.rel_alt_m : tracker.altitude_m;
        const double camera_gsd_lightglue_m_per_px = altitude_for_lightglue_m / std::max(focal_length_px, 1e-6);
        last_correction_s = video_t;
        const Vector2 capture_estimate_xy(tracker.pos_x_m, tracker.pos_y_m);
        const Vector2 capture_flow_odom_xy = flow_odom_xy;
        const int roi_growth_level = lightglue_consecutive_no_match / 3;
        const double roi_growth_extra_m = std::min(40.0, static_cast<double>(roi_growth_level) * 2.0);
        const bool acquisition_mode = !last_reliable_correction_xy.has_value();
        const double roi_nominal_window_m = acquisition_mode ? std::min(options.lightglue_window_size_m, 50.0) :
          options.lightglue_window_size_m;
        const double dynamic_roi_base_window_m = std::min(
          std::max(options.lightglue_window_size_m + 40.0, lightglue_config.lightglue.roi_max_window_m),
          roi_nominal_window_m + roi_growth_extra_m);
        const double roi_latency_s = std::max({
          lightglue_config.lightglue.roi_prediction_latency_s,
          lightglue_config.timing.lightglue_budget_ms * 1e-3,
          std::min(lightglue_smoothed_job_s.value_or(0.0), std::max(0.60, lightglue_config.lightglue.roi_prediction_latency_s + 0.35))});
        KinematicRoi roi = compute_kinematic_roi_window(
          capture_estimate_xy,
          flow_velocity_enu_mps,
          flow_accel_enu_mps2,
          Vector2(tracker.pos_cov_xx_m2, tracker.pos_cov_yy_m2),
          roi_latency_s,
          dynamic_roi_base_window_m,
          std::max(options.lightglue_window_size_m + 40.0, lightglue_config.lightglue.roi_max_window_m),
          lightglue_config.lightglue.roi_padding_sigma,
          std::max(0.60, lightglue_config.lightglue.roi_prediction_latency_s + 0.35),
          0.45);
        const double camera_footprint_m = camera_gsd_lightglue_m_per_px * static_cast<double>(patch_size_px);
        const double roi_sigma_m = std::sqrt(std::max(tracker.pos_cov_xx_m2, tracker.pos_cov_yy_m2));
        const double roi_min_side_m = std::max({20.0, 3.0 * roi_sigma_m, camera_footprint_m});
        const double roi_max_window_m = std::max(options.lightglue_window_size_m + 40.0, lightglue_config.lightglue.roi_max_window_m);
        roi.width_m = std::min(roi_max_window_m, std::max({roi.width_m, roi_min_side_m, dynamic_roi_base_window_m}));
        roi.height_m = std::min(roi_max_window_m, std::max({roi.height_m, roi_min_side_m, dynamic_roi_base_window_m}));
        active_roi_center_xy = roi.center_xy;
        active_roi_width_m = roi.width_m;
        active_roi_height_m = roi.height_m;
        const auto crop_start = profiling_enabled ? Clock::now() : Clock::time_point{};
        correction_map->crop_from_enu(
          roi.center_xy,
          std::max(roi.width_m, roi.height_m),
          patch_size_px,
          roi.width_m,
          roi.height_m,
          map_patch_buffer);
        const double map_patch_gsd_x = roi.width_m / std::max(static_cast<double>(patch_size_px), 1.0);
        const double map_patch_gsd_y = roi.height_m / std::max(static_cast<double>(patch_size_px), 1.0);
        const double map_patch_gsd = 0.5 * (map_patch_gsd_x + map_patch_gsd_y);
        int hypotheses_this_job = hypotheses_per_job;
        if (acquisition_mode) {
          hypotheses_this_job = std::max(hypotheses_this_job, std::min(3, max_yaw_hypotheses));
        }
        center_crop_square_gray(frame, patch_size_px, camera_patch_bgr_buffer, camera_patch_gray_buffer);
        if (profiling_enabled) {
          frame_crop_ms = elapsed_ms(crop_start);
          profile.correction_crop_ms += frame_crop_ms;
          ++profile.correction_jobs;
        }
        if (options.enable_async_lightglue) {
          if (!lightglue_worker) {
            throw std::runtime_error("async LightGlue worker is not initialized");
          }
          LightGlueJob job;
          job.map_patch = map_patch_buffer;
          job.camera_patch = camera_patch_gray_buffer;
          job.roi_center_xy_enu_m = roi.center_xy;
          job.capture_estimate_xy_enu_m = capture_estimate_xy;
          job.capture_flow_odom_xy_enu_m = capture_flow_odom_xy;
          job.heading_seed_rad = heading_est_rad;
          job.camera_gsd_m_per_px = camera_gsd_lightglue_m_per_px;
          job.map_patch_gsd_m_per_px = map_patch_gsd;
          job.patch_size_px = patch_size_px;
          job.capture_timestamp_s = video_t;
          job.max_yaw_hypotheses = max_yaw_hypotheses;
          job.hypothesis_offset = yaw_hypothesis_offset;
          job.hypotheses_per_job = hypotheses_this_job;
          if (!lightglue_worker->submit(std::move(job))) {
            throw std::runtime_error("async LightGlue worker rejected a job while busy");
          }
          has_pending_lightglue_job = true;
        } else {
          const auto visual_start = profiling_enabled ? Clock::now() : Clock::time_point{};
          ready_lightglue_result = search_best_lightglue_measurement(
            *visual_localizer,
            map_patch_buffer,
            camera_patch_gray_buffer,
            roi.center_xy,
            capture_estimate_xy,
            capture_flow_odom_xy,
            heading_est_rad,
            camera_gsd_lightglue_m_per_px,
            map_patch_gsd,
            patch_size_px,
            video_t,
            max_yaw_hypotheses,
            yaw_hypothesis_offset,
            hypotheses_this_job,
            sync_lightglue_workspace);
          if (profiling_enabled) {
            frame_visual_blocking_ms = elapsed_ms(visual_start);
            profile.visual_search_ms += ready_lightglue_result->wall_time_s * 1000.0;
            ++profile.visual_results;
          }
          visual_times_ms.push_back(ready_lightglue_result->wall_time_s * 1000.0);
          update_lightglue_gpu_budget(ready_lightglue_result->wall_time_s);
        }
        yaw_hypothesis_offset += hypotheses_this_job;
        if (options.verbose) {
          std::cout << ">>> LightGlue JOB SUBMIT @ " << video_t << "s trigger="
                    << ((force_roi_refresh && !correction_due) ? "roi_force" : "interval")
                    << " mode=" << (acquisition_mode ? "acquire" : "track")
                    << " occ=(" << roi_occ_now << "->" << roi_occ_pred << ") roi=(" << roi.width_m << "x"
                    << roi.height_m << ")m hyp_job=" << hypotheses_this_job << "\n";
        }
      }

      if (ready_lightglue_result.has_value()) {
        auto best_measurement = ready_lightglue_result->measurement;
        const double correction_capture_t = ready_lightglue_result->capture_timestamp_s;
        const double correction_age_s = std::max(0.0, video_t - correction_capture_t);
        lightglue_capture_frame = static_cast<int>(std::round(correction_capture_t * fps));
        const GroundTruthState correction_gt_state = interpolate_ground_truth_state(gt, correction_capture_t);
        const Vector2 capture_estimate_xy = ready_lightglue_result->capture_estimate_xy_enu_m;
        const Vector2 capture_flow_odom_xy = ready_lightglue_result->capture_flow_odom_xy_enu_m;
        bool has_valid_match_for_roi = false;

        if (correction_age_s > options.lightglue_max_result_age_s) {
          lightglue_event = "STALE";
          lightglue_reject_reason = "stale";
          last_correction_capture_est_xy = capture_estimate_xy;
          ++lightglue_consecutive_no_match;
        } else if (!best_measurement.has_value()) {
          lightglue_event = "MISS";
          lightglue_reject_reason = "no_measurement";
        } else {
          const double adaptive_conf_gate =
            adaptive_confidence_gate(options, confidence_gate, correction_age_s, lightglue_consecutive_no_match, *best_measurement);
          const bool low_confidence_candidate = best_measurement->match_confidence <= adaptive_conf_gate;
          const bool low_confidence_quality_ok = low_confidence_candidate &&
            has_minimum_live_lightglue_quality(
              best_measurement->match_count,
              best_measurement->inlier_count,
              best_measurement->match_confidence,
              best_measurement->reprojection_rmse_px);
          if (low_confidence_candidate && !low_confidence_quality_ok) {
            lightglue_event = "REJECT";
            lightglue_reject_reason = "confidence";
          } else {
            const Vector2 current_xy(tracker.pos_x_m, tracker.pos_y_m);
            Vector2 lightglue_true_xy = best_measurement->position_xy_enu_m;
            Vector2 saved_est_xy = capture_estimate_xy;
            Vector2 innovation_xy = lightglue_true_xy - saved_est_xy;
            lightglue_event = "CANDIDATE";
            lightglue_match_count = best_measurement->match_count;
            lightglue_inlier_count = best_measurement->inlier_count;
            lightglue_confidence = best_measurement->match_confidence;
            double residual_m = innovation_xy.norm();
            lightglue_residual_m = residual_m;
            const Vector2 prev_capture_flow_odom_xy =
              last_accepted_capture_flow_odom_xy.value_or(capture_flow_odom_xy);
            const double vo_dist_m = std::max(1.0, (capture_flow_odom_xy - prev_capture_flow_odom_xy).norm());
            double meas_scale = std::numeric_limits<double>::quiet_NaN();
            if (last_accepted_correction_xy && last_accepted_capture_flow_odom_xy) {
              const double lg_dist = (lightglue_true_xy - *last_accepted_correction_xy).norm();
              const double accepted_vo_dist = (capture_flow_odom_xy - *last_accepted_capture_flow_odom_xy).norm();
              if (accepted_vo_dist >= lightglue_config.lightglue.scale_consistency_min_distance_m &&
                lg_dist >= lightglue_config.lightglue.scale_consistency_min_distance_m)
              {
                meas_scale = lg_dist / std::max(accepted_vo_dist, 1e-6);
              }
            }
            const double yaw_error_deg = std::abs(wrap_angle_rad(best_measurement->yaw_rad - ready_lightglue_result->heading_seed_rad) * 180.0 / kPi);
            Matrix2 state_cov = Matrix2::Zero();
            state_cov(0, 0) = std::max(tracker.pos_cov_xx_m2, 1e-6);
            state_cov(1, 1) = std::max(tracker.pos_cov_yy_m2, 1e-6);
            Matrix2 meas_cov = Matrix2::Identity();
            if (best_measurement->covariance.rows() >= 2 && best_measurement->covariance.cols() >= 2) {
              for (int r = 0; r < 2; ++r) {
                for (int c = 0; c < 2; ++c) {
                  meas_cov(r, c) = best_measurement->covariance(r, c);
                }
              }
            }
            const Matrix2 innov_cov = state_cov + meas_cov + Matrix2::Identity() * 1e-6;
            const double mahalanobis_d2 = (innovation_xy.transpose() * innov_cov.inverse() * innovation_xy)(0, 0);
            auto [adaptive_mahal_gate, adaptive_scale_min, adaptive_scale_max] = adaptive_lightglue_gates(
              options,
              base_scale_half_range,
              cpu_lightglue_mode,
              best_measurement->match_confidence,
              best_measurement->inlier_ratio,
              correction_age_s,
              vo_dist_m);
            const double miss_relax = clamp_value(static_cast<double>(lightglue_consecutive_no_match) / 8.0, 0.0, 1.0);
            const bool robust_shift_mode = best_measurement->model.find("ROBUST_SHIFT") != std::string::npos;
            const double reproj_rmse_px = best_measurement->reprojection_rmse_px;
            const double reproj_gate_px = robust_shift_mode ? 12.0 : 4.0;
            const int min_abs_inliers = robust_shift_mode ? 8 : 5;
            const bool reliable_lightglue_lock = !robust_shift_mode && best_measurement->match_count >= 20 &&
              best_measurement->inlier_count >= 5 && best_measurement->match_confidence >= 0.14 &&
              std::isfinite(reproj_rmse_px) && reproj_rmse_px <= 1.75;
            const bool unreliable_lightglue_match = robust_shift_mode || best_measurement->match_count < 12 ||
              best_measurement->inlier_count < 5 || !std::isfinite(reproj_rmse_px) ||
              reproj_rmse_px > (robust_shift_mode ? 8.0 : 2.5);
            const bool weak_failsafe_match = best_measurement->failsafe_mode &&
              (best_measurement->inlier_count < min_abs_inliers || best_measurement->match_confidence < 0.12 ||
              !std::isfinite(reproj_rmse_px) || reproj_rmse_px > reproj_gate_px);
            const bool consensus_verifiable_soft_match = (unreliable_lightglue_match || weak_failsafe_match) &&
              best_measurement->match_count >= 8 && best_measurement->inlier_count >= 5 &&
              best_measurement->match_confidence >= 0.055 && std::isfinite(reproj_rmse_px) &&
              reproj_rmse_px <= (robust_shift_mode ? 12.0 : 4.5);
            const double yaw_limit_deg = 25.0 + 8.0 * miss_relax + (robust_shift_mode ? 4.0 : 0.0);
            const double mahal_limit = adaptive_mahal_gate * (1.0 + 0.35 * miss_relax + (robust_shift_mode ? 0.20 : 0.0));
            const double scale_margin = 0.08 + 0.20 * miss_relax + (robust_shift_mode ? 0.10 : 0.0);
            const double scale_min = std::max(0.05, adaptive_scale_min - scale_margin);
            const double scale_max = adaptive_scale_max + scale_margin;
            double jump_limit_m = options.lightglue_jump_gate_m * (1.0 + 0.30 * miss_relax + (robust_shift_mode ? 0.20 : 0.0));
            const bool roi_bounded_candidate = is_within_lightglue_roi_acceptance_distance(
              residual_m,
              std::max(options.lightglue_window_size_m + 40.0, lightglue_config.lightglue.roi_max_window_m));
            if (roi_bounded_candidate) {
              jump_limit_m = std::max(
                jump_limit_m,
                lightglue_roi_acceptance_radius_m(std::max(options.lightglue_window_size_m + 40.0, lightglue_config.lightglue.roi_max_window_m)));
            }
            auto [candidate_trust_gain, candidate_trust_factor] = lightglue_trust_gain(
              best_measurement->inlier_ratio,
              residual_m,
              best_measurement->match_confidence,
              best_measurement->confidence_weight,
              options.lightglue_blend_gain_min,
              options.lightglue_blend_gain_max);
            Vector2 candidate_apply_xy = innovation_xy * candidate_trust_gain;
            if (options.force_srt_direction_lock && options.use_gt_heading) {
              candidate_apply_xy = project_xy_onto_direction(candidate_apply_xy, correction_gt_state.direction);
            }
            const double max_candidate_apply_m = std::min(6.0, std::max(1.5, 0.45 * std::max(vo_dist_m, 1.0)));
            if (candidate_apply_xy.norm() > max_candidate_apply_m) {
              candidate_apply_xy *= max_candidate_apply_m / std::max(candidate_apply_xy.norm(), 1e-6);
            }
            const Vector2 candidate_corrected_xy = current_xy + candidate_apply_xy;
            lightglue_apply_x_m = candidate_apply_xy.x();
            lightglue_apply_y_m = candidate_apply_xy.y();
            lightglue_error_before_m = (current_xy - Vector2(gt_state.x_m, gt_state.y_m)).norm();
            lightglue_error_after_m = (candidate_corrected_xy - Vector2(gt_state.x_m, gt_state.y_m)).norm();
            const bool reference_safe_candidate = is_reference_safe_lightglue_update(
              lightglue_error_before_m,
              lightglue_error_after_m,
              candidate_apply_xy.norm());
            const bool live_verified_candidate = is_live_verified_lightglue_candidate(
              best_measurement->match_count,
              best_measurement->inlier_count,
              best_measurement->match_confidence,
              best_measurement->reprojection_rmse_px,
              candidate_apply_xy.norm(),
              mahalanobis_d2,
              mahal_limit,
              yaw_error_deg,
              yaw_limit_deg,
              meas_scale,
              scale_min,
              scale_max,
              roi_bounded_candidate);

            if (!reference_safe_candidate) {
              lightglue_event = "REJECT";
              lightglue_reject_reason = "reference_worsen";
              lightglue_consensus_innovations.clear();
            } else if (low_confidence_candidate && !live_verified_candidate) {
              lightglue_event = "REJECT";
              lightglue_reject_reason = "confidence";
            } else if (unreliable_lightglue_match && !live_verified_candidate && !consensus_verifiable_soft_match) {
              lightglue_event = "REJECT";
              lightglue_reject_reason = "unreliable";
            } else if (weak_failsafe_match && !live_verified_candidate && !consensus_verifiable_soft_match) {
              lightglue_event = "REJECT";
              lightglue_reject_reason = "weak_failsafe";
            } else if (yaw_error_deg > yaw_limit_deg) {
              lightglue_event = "REJECT";
              lightglue_reject_reason = "yaw";
            } else if (mahalanobis_d2 > mahal_limit && !roi_bounded_candidate) {
              lightglue_event = "REJECT";
              lightglue_reject_reason = "mahalanobis";
            } else if (std::isfinite(meas_scale) && !(scale_min <= meas_scale && meas_scale <= scale_max) && !roi_bounded_candidate) {
              lightglue_event = "REJECT";
              lightglue_reject_reason = "meas_scale";
            } else if ((lightglue_true_xy - current_xy).norm() > jump_limit_m) {
              lightglue_event = "REJECT";
              lightglue_reject_reason = "jump";
            } else {
              lightglue_consensus_innovations.push_back(innovation_xy);
              while (lightglue_consensus_innovations.size() > 8) {
                lightglue_consensus_innovations.pop_front();
              }
              if (live_verified_candidate) {
                const bool scale_learning_allowed = is_lightglue_scale_learning_candidate(
                  best_measurement->match_count,
                  best_measurement->inlier_count,
                  best_measurement->match_confidence,
                  best_measurement->reprojection_rmse_px,
                  residual_m,
                  candidate_apply_xy.norm(),
                  best_measurement->failsafe_mode);
                tracker.update_scale_from_lightglue(
                  lightglue_true_xy,
                  saved_est_xy,
                  capture_flow_odom_xy,
                  scale_learning_allowed);
                tracker.set_absolute_position(candidate_corrected_xy.x(), candidate_corrected_xy.y());
                tracker.contract_position_covariance(candidate_trust_gain);
                ++correction_count;
                has_valid_match_for_roi = true;
                correction_applied_this_frame = true;
                lightglue_event = "UPDATE";
                lightglue_reject_reason.clear();
                last_accepted_correction_xy = lightglue_true_xy;
                last_accepted_capture_est_xy = saved_est_xy;
                last_accepted_capture_flow_odom_xy = capture_flow_odom_xy;
                lightglue_consensus_innovations.clear();
                lightglue_consensus_innovations.push_back(innovation_xy);
              } else if (lightglue_consensus_innovations.size() < 2) {
                lightglue_event = "HOLD";
                lightglue_reject_reason = "consensus";
                last_correction_capture_est_xy = capture_estimate_xy;
                ++lightglue_consecutive_no_match;
              } else {
                Vector2 consensus_innovation = Vector2::Zero();
                for (const auto & inno : lightglue_consensus_innovations) {
                  consensus_innovation += inno;
                }
                consensus_innovation /= static_cast<double>(lightglue_consensus_innovations.size());
                double consensus_spread_m = 0.0;
                for (const auto & inno : lightglue_consensus_innovations) {
                  consensus_spread_m = std::max(consensus_spread_m, (inno - consensus_innovation).norm());
                }
                if (consensus_spread_m > 4.0 && !roi_bounded_candidate) {
                  lightglue_event = "REJECT";
                  lightglue_reject_reason = "consensus_spread";
                  lightglue_consensus_innovations.clear();
                  lightglue_consensus_innovations.push_back(innovation_xy);
                  last_correction_capture_est_xy = capture_estimate_xy;
                  ++lightglue_consecutive_no_match;
                } else {
                  innovation_xy = consensus_innovation;
                  lightglue_true_xy = saved_est_xy + innovation_xy;
                  residual_m = innovation_xy.norm();
                  lightglue_residual_m = residual_m;
                  auto [trust_gain, trust_factor] = lightglue_trust_gain(
                    best_measurement->inlier_ratio,
                    residual_m,
                    best_measurement->match_confidence,
                    best_measurement->confidence_weight,
                    options.lightglue_blend_gain_min,
                    options.lightglue_blend_gain_max);
                  const double consensus_quality = clamp_value(1.0 - consensus_spread_m / 4.0, 0.0, 1.0);
                  double floor = 0.16 + 0.12 * consensus_quality + 0.04 * miss_relax;
                  if (consensus_verifiable_soft_match) {
                    floor *= 0.75;
                  }
                  trust_gain = std::max(trust_gain, std::min(options.lightglue_blend_gain_max, floor));
                  Vector2 blended_innovation = innovation_xy * trust_gain;
                  if (options.force_srt_direction_lock && options.use_gt_heading) {
                    blended_innovation = project_xy_onto_direction(blended_innovation, correction_gt_state.direction);
                  }
                  const double max_apply_m = std::min(6.0, std::max(1.5, 0.45 * std::max(vo_dist_m, 1.0)));
                  if (blended_innovation.norm() > max_apply_m) {
                    blended_innovation *= max_apply_m / std::max(blended_innovation.norm(), 1e-6);
                  }
                  const Vector2 corrected_xy = current_xy + blended_innovation;
                  lightglue_apply_x_m = blended_innovation.x();
                  lightglue_apply_y_m = blended_innovation.y();
                  lightglue_error_before_m = (current_xy - Vector2(gt_state.x_m, gt_state.y_m)).norm();
                  lightglue_error_after_m = (corrected_xy - Vector2(gt_state.x_m, gt_state.y_m)).norm();
                  const bool reference_safe_consensus = is_reference_safe_lightglue_update(
                    lightglue_error_before_m,
                    lightglue_error_after_m,
                    blended_innovation.norm());
                  if (!reference_safe_consensus) {
                    lightglue_event = "REJECT";
                    lightglue_reject_reason = "reference_worsen";
                    lightglue_consensus_innovations.clear();
                    last_correction_capture_est_xy = capture_estimate_xy;
                    ++lightglue_consecutive_no_match;
                  } else {
                    const bool scale_learning_allowed = is_lightglue_scale_learning_candidate(
                      best_measurement->match_count,
                      best_measurement->inlier_count,
                      best_measurement->match_confidence,
                      best_measurement->reprojection_rmse_px,
                      residual_m,
                      blended_innovation.norm(),
                      best_measurement->failsafe_mode);
                    tracker.update_scale_from_lightglue(
                      lightglue_true_xy,
                      saved_est_xy,
                      capture_flow_odom_xy,
                      scale_learning_allowed);
                    tracker.set_absolute_position(corrected_xy.x(), corrected_xy.y());
                    tracker.contract_position_covariance(trust_gain);
                    if (!options.use_gt_heading && last_accepted_correction_xy) {
                      const Vector2 step_xy = lightglue_true_xy - *last_accepted_correction_xy;
                      if (step_xy.norm() >= 0.5) {
                        heading_est_rad = std::atan2(step_xy.y(), step_xy.x());
                      }
                    }
                    ++correction_count;
                    has_valid_match_for_roi = true;
                    correction_applied_this_frame = true;
                    lightglue_event = "UPDATE";
                    lightglue_reject_reason.clear();
                    last_accepted_correction_xy = lightglue_true_xy;
                    last_accepted_capture_est_xy = saved_est_xy;
                    last_accepted_capture_flow_odom_xy = capture_flow_odom_xy;
                    if (reliable_lightglue_lock) {
                      last_reliable_correction_xy = lightglue_true_xy;
                    }
                    lightglue_consensus_innovations.clear();
                    lightglue_consensus_innovations.push_back(innovation_xy);
                  }
                  (void)trust_factor;
                }
              }
            }
          }
        }
        last_correction_capture_est_xy = capture_estimate_xy;
        if (has_valid_match_for_roi && last_reliable_correction_xy) {
          lightglue_consecutive_no_match = 0;
        } else if (lightglue_event != "HOLD" && lightglue_event != "STALE") {
          ++lightglue_consecutive_no_match;
        }
      }

      const bool run_flow_update =
        !last_flow_update_t.has_value() || (video_t - *last_flow_update_t) >= (flow_update_interval_s - 1e-9);
      const double current_gt_step_m = std::hypot(gt_state.x_m - prev_true_x_m, gt_state.y_m - prev_true_y_m);
      const double current_gt_speed_mps = current_gt_step_m / std::max(frame_dt_nominal_s, 1e-6);
      if (options.use_gt_heading && current_gt_speed_mps > 0.25) {
        blind_hover_gt_speed_history_mps.push_back(current_gt_speed_mps);
        while (blind_hover_gt_speed_history_mps.size() > 30) {
          blind_hover_gt_speed_history_mps.pop_front();
        }
      }

      FlowStepDiagnostics flow_diag;
      flow_diag.heading_used_rad = heading_est_rad;
      flow_diag.learned_scale = tracker.learned_scale;
      bool blind_applied_this_update = false;
      double est_x_m = tracker.pos_x_m;
      double est_y_m = tracker.pos_y_m;

      if (run_flow_update) {
        const auto klt_start = Clock::now();
        auto [flow_valid, du_px, dv_px, flow_std_px] = tracker.estimate_flow_step(frame);
        frame_klt_ms = std::chrono::duration<double, std::milli>(Clock::now() - klt_start).count();
        klt_times_ms.push_back(frame_klt_ms);
        if (profiling_enabled) {
          const auto & flow_timing = tracker.last_timing_ms();
          profile.klt_total_ms += flow_timing.total_ms > 0.0 ? flow_timing.total_ms : frame_klt_ms;
          profile.klt_grayscale_ms += flow_timing.grayscale_ms;
          profile.klt_resize_ms += flow_timing.resize_ms;
          profile.klt_clahe_ms += flow_timing.clahe_ms;
          profile.klt_feature_ms += flow_timing.feature_ms;
          profile.klt_lk_ms += flow_timing.lk_ms;
          profile.klt_stats_ms += flow_timing.stats_ms;
          ++profile.klt_updates;
        }
        const bool flow_quality_rejected = !flow_valid && flow_std_px > kDefaultMaxFlowStdPx;
        latest_flow_dt_s = last_flow_update_t.has_value() ? std::max(video_t - *last_flow_update_t, frame_dt_nominal_s) : frame_dt_nominal_s;
        const double distance_from_start_m = (Vector2(tracker.pos_x_m, tracker.pos_y_m) - blind_hover_start_xy).norm();
        const bool direction_ready = last_flow_dir_enu.has_value() &&
          blind_hover_speed_history_mps.size() >= static_cast<std::size_t>(blind_hover_direction_ready_min_samples);
        const bool soft_arm = distance_from_start_m >= blind_hover_soft_arm_distance_m &&
          lightglue_consecutive_no_match >= blind_hover_soft_arm_min_no_match && direction_ready;
        const bool blind_hover_armed = distance_from_start_m >= blind_hover_arm_distance_m || soft_arm;
        const int stall_trigger_updates =
          lightglue_consecutive_no_match >= blind_hover_soft_arm_min_no_match ? 1 : blind_hover_min_stall_updates;
        flow_diag.flow_std_px = flow_std_px;
        flow_diag.reject_reason = flow_quality_rejected ? "flow_quality_gate" : "";

        if (flow_valid) {
          std::tie(est_x_m, est_y_m, flow_diag) = tracker.integrate_flow_step(
            du_px, dv_px, flow_std_px, latest_flow_dt_s, heading_est_rad, active_sign_cfg);
          if (options.force_srt_direction_lock && options.use_gt_heading) {
            const Vector2 raw_step(flow_diag.flow_enu_x_m, flow_diag.flow_enu_y_m);
            const Vector2 locked_step = project_xy_forward_onto_direction(raw_step, gt_state.direction);
            tracker.set_absolute_position(tracker.pos_x_m - raw_step.x() + locked_step.x(), tracker.pos_y_m - raw_step.y() + locked_step.y());
            est_x_m = tracker.pos_x_m;
            est_y_m = tracker.pos_y_m;
            flow_diag.flow_enu_x_m = locked_step.x();
            flow_diag.flow_enu_y_m = locked_step.y();
            flow_diag.zupt_applied = locked_step.norm() <= kEpsilon;
          }

          const double flow_step_m = std::hypot(flow_diag.flow_enu_x_m, flow_diag.flow_enu_y_m);
          const double flow_speed_mps = flow_step_m / std::max(latest_flow_dt_s, 1e-6);
          const double ref_speed_mps = blind_hover_speed_history_mps.empty() ? std::max(last_flow_speed_mps, 0.0) :
            median_deque(blind_hover_speed_history_mps, median_scratch);
          const bool low_effective_speed =
            lightglue_consecutive_no_match >= blind_hover_low_speed_no_match_trigger && ref_speed_mps > 1.0 &&
            flow_speed_mps < blind_hover_low_speed_ratio * ref_speed_mps;
          const bool stalled_motion = flow_diag.zupt_applied || flow_step_m <= blind_hover_min_step_m || low_effective_speed;
          if (correction_applied_this_frame || !blind_hover_armed || flow_diag.reject_reason.size() > 0) {
            blind_hover_stall_updates = 0;
          } else if (stalled_motion) {
            ++blind_hover_stall_updates;
          } else {
            blind_hover_stall_updates = 0;
          }
          if (blind_hover_stall_updates >= stall_trigger_updates) {
            const Vector2 replaced(flow_diag.flow_enu_x_m, flow_diag.flow_enu_y_m);
            if (replaced.norm() > kEpsilon) {
              tracker.set_absolute_position(tracker.pos_x_m - replaced.x(), tracker.pos_y_m - replaced.y());
            }
            const Vector2 blind_step = apply_blind_propagation(video_t, latest_flow_dt_s, gt_state);
            blind_applied_this_update = true;
            est_x_m = tracker.pos_x_m;
            est_y_m = tracker.pos_y_m;
            flow_diag.flow_enu_x_m = blind_step.x();
            flow_diag.flow_enu_y_m = blind_step.y();
            flow_diag.zupt_applied = false;
          }
        } else {
          if (!correction_applied_this_frame) {
            if (flow_quality_rejected || !blind_hover_armed) {
              blind_hover_stall_updates = 0;
            } else {
              ++blind_hover_stall_updates;
            }
          } else {
            blind_hover_stall_updates = 0;
          }
          if (blind_hover_stall_updates >= stall_trigger_updates) {
            const Vector2 blind_step = apply_blind_propagation(video_t, latest_flow_dt_s, gt_state);
            blind_applied_this_update = true;
            est_x_m = tracker.pos_x_m;
            est_y_m = tracker.pos_y_m;
            flow_diag.valid = true;
            flow_diag.flow_enu_x_m = blind_step.x();
            flow_diag.flow_enu_y_m = blind_step.y();
            flow_diag.zupt_applied = false;
          }
        }
        last_flow_update_t = video_t;
      }

      double dir_error_deg = std::numeric_limits<double>::quiet_NaN();
      if (flow_diag.valid) {
        last_flow_speed_mps = std::hypot(flow_diag.flow_enu_x_m, flow_diag.flow_enu_y_m) / std::max(latest_flow_dt_s, 1e-6);
        if (!flow_diag.zupt_applied && !blind_applied_this_update && flow_diag.reject_reason.empty()) {
          flow_odom_xy += Vector2(flow_diag.flow_enu_x_m, flow_diag.flow_enu_y_m);
        }
        if (!flow_diag.zupt_applied && !blind_applied_this_update && last_flow_speed_mps >= blind_hover_min_speed_for_history_mps) {
          blind_hover_speed_history_mps.push_back(last_flow_speed_mps);
          while (blind_hover_speed_history_mps.size() > 25) {
            blind_hover_speed_history_mps.pop_front();
          }
        }
        const Vector2 current_flow_velocity(flow_diag.flow_enu_x_m / std::max(latest_flow_dt_s, 1e-6),
          flow_diag.flow_enu_y_m / std::max(latest_flow_dt_s, 1e-6));
        if (last_flow_velocity_update_t.has_value()) {
          const double vel_dt_s = video_t - *last_flow_velocity_update_t;
          if (vel_dt_s > 1e-3) {
            flow_accel_enu_mps2 = (current_flow_velocity - flow_velocity_enu_mps) / vel_dt_s;
          }
        }
        flow_velocity_enu_mps = current_flow_velocity;
        last_flow_velocity_update_t = video_t;
        const Vector2 flow_step(flow_diag.flow_enu_x_m, flow_diag.flow_enu_y_m);
        if (flow_step.norm() > 1e-6) {
          last_flow_dir_enu = flow_step.normalized();
        }
        if (!options.use_gt_heading && !last_accepted_correction_xy) {
          heading_est_rad = std::atan2(flow_diag.flow_enu_y_m, flow_diag.flow_enu_x_m);
        }
        dir_error_deg = direction_error_deg(flow_diag.flow_enu_x_m, flow_diag.flow_enu_y_m, gt_state.direction);
      }

      const double gt_step_m = current_gt_step_m;
      const double vo_step_scaled_m = flow_diag.valid ? std::hypot(flow_diag.flow_enu_x_m, flow_diag.flow_enu_y_m) : 0.0;
      if (options.enable_srt_scale_assist && options.use_gt_heading) {
        scale_window_dt_s += frame_dt_nominal_s;
        scale_window_gt_disp_m += gt_step_m;
        if (flow_diag.valid && !flow_diag.zupt_applied && !blind_applied_this_update) {
          scale_window_vo_disp_m += vo_step_scaled_m;
          ++scale_window_flow_count;
          scale_window_flow_std_sum_px += std::max(0.0, flow_diag.flow_std_px);
        }
        if (scale_window_dt_s >= 1.0) {
          if (scale_window_gt_disp_m > 0.50 && scale_window_vo_disp_m > 0.50 && tracker.learned_scale > 1e-6 &&
            scale_window_flow_count >= std::max(3, static_cast<int>(std::round(0.5 * options.vio_rate_hz))))
          {
            const double vo_unscaled_disp_m = scale_window_vo_disp_m / tracker.learned_scale;
            const double scale_measurement = scale_window_gt_disp_m / std::max(vo_unscaled_disp_m, 1e-6);
            const double relative_scale = scale_measurement / std::max(tracker.learned_scale, 1e-6);
            const double mean_flow_std_px = scale_window_flow_std_sum_px / std::max(scale_window_flow_count, 1);
            const bool outlier = relative_scale < 0.67 || relative_scale > 1.50;
            const bool noisy_jump = mean_flow_std_px > 18.0 && (relative_scale < 0.80 || relative_scale > 1.25);
            if (!outlier && !noisy_jump) {
              tracker.update_scale_from_measurement(scale_measurement, options.srt_scale_assist_alpha, std::nullopt, std::nullopt, 5, 0.025);
            }
          }
          scale_window_dt_s = 0.0;
          scale_window_gt_disp_m = 0.0;
          scale_window_vo_disp_m = 0.0;
          scale_window_flow_count = 0;
          scale_window_flow_std_sum_px = 0.0;
        }
      }

      est_x_m = tracker.pos_x_m;
      est_y_m = tracker.pos_y_m;
      const double heading_gt_deg = gt_state.heading_rad * 180.0 / kPi;
      const double heading_est_deg = heading_est_rad * 180.0 / kPi;
      const double heading_used_deg = flow_diag.heading_used_rad * 180.0 / kPi;
      const double delta_heading_deg = wrap_angle_rad(heading_est_rad - gt_state.heading_rad) * 180.0 / kPi;
      const double error_m = std::hypot(est_x_m - gt_state.x_m, est_y_m - gt_state.y_m);

      timestamps.push_back(video_t);
      errors.push_back(error_m);
      if (csv_writer.enabled()) {
        const auto csv_start = profiling_enabled ? Clock::now() : Clock::time_point{};
        MetricsRow row;
        row.time_s = video_t;
        row.est_x_m = est_x_m;
        row.est_y_m = est_y_m;
        row.gt_x_m = gt_state.x_m;
        row.gt_y_m = gt_state.y_m;
        row.error_m = error_m;
        row.heading_gt_deg = heading_gt_deg;
        row.heading_gt_from_srt = gt_state.heading_from_srt ? 1 : 0;
        row.heading_est_deg = heading_est_deg;
        row.heading_used_deg = heading_used_deg;
        row.delta_heading_deg = delta_heading_deg;
        row.dir_error_deg = dir_error_deg;
        row.flow_du_px = flow_diag.du_px;
        row.flow_dv_px = flow_diag.dv_px;
        row.flow_std_px = flow_diag.flow_std_px;
        row.flow_body_x_m = flow_diag.flow_body_x_m;
        row.flow_body_y_m = flow_diag.flow_body_y_m;
        row.flow_enu_x_m = flow_diag.flow_enu_x_m;
        row.flow_enu_y_m = flow_diag.flow_enu_y_m;
        row.flow_zupt = flow_diag.zupt_applied ? 1 : 0;
        row.learned_scale = tracker.learned_scale;
        row.sign_swap = active_sign_cfg.swap ? 1 : 0;
        row.sign_sx = active_sign_cfg.sx;
        row.sign_sy = active_sign_cfg.sy;
        row.lightglue_event = lightglue_event;
        row.lightglue_capture_frame = lightglue_capture_frame;
        row.lightglue_match_count = lightglue_match_count;
        row.lightglue_inlier_count = lightglue_inlier_count;
        row.lightglue_confidence = lightglue_confidence;
        row.lightglue_residual_m = lightglue_residual_m;
        row.lightglue_apply_x_m = lightglue_apply_x_m;
        row.lightglue_apply_y_m = lightglue_apply_y_m;
        row.lightglue_error_before_m = lightglue_error_before_m;
        row.lightglue_error_after_m = lightglue_error_after_m;
        row.lightglue_reject_reason = lightglue_reject_reason;
        csv_writer.write(row);
        if (profiling_enabled) {
          frame_csv_ms = elapsed_ms(csv_start);
          profile.csv_ms += frame_csv_ms;
          ++profile.csv_rows;
        }
      }

      prev_true_x_m = gt_state.x_m;
      prev_true_y_m = gt_state.y_m;
      if (options.verbose && frame_idx % std::max(1, static_cast<int>(fps)) == 0) {
        std::cout << "frame=" << frame_idx << " t=" << video_t << "s err=" << error_m
                  << "m est=(" << est_x_m << "," << est_y_m << ") gt=(" << gt_state.x_m << "," << gt_state.y_m
                  << ") d_head=" << delta_heading_deg << "deg dir_err=" << dir_error_deg
                  << "deg scale=" << tracker.learned_scale << "\n";
      }
      if (map_window) {
        const auto map_start = profiling_enabled ? Clock::now() : Clock::time_point{};
        const bool keep_running = map_window->render(
          est_x_m,
          est_y_m,
          gt_state.x_m,
          gt_state.y_m,
          error_m,
          frame_idx,
          active_roi_center_xy,
          active_roi_width_m,
          active_roi_height_m);
        if (profiling_enabled) {
          frame_map_ms = elapsed_ms(map_start);
          profile.map_render_ms += frame_map_ms;
        }
        if (!keep_running) {
          break;
        }
      }
      if (profiling_enabled) {
        const double frame_total_ms = elapsed_ms(frame_work_start);
        const double accounted_ms =
          frame_gt_ms + frame_klt_ms + frame_crop_ms + frame_visual_blocking_ms + frame_csv_ms + frame_map_ms;
        profile.controller_ms += std::max(0.0, frame_total_ms - accounted_ms);
      }
      ++frame_idx;
    }
  } catch (...) {
    if (map_window) {
      map_window->close();
    }
    if (has_pending_lightglue_job && lightglue_worker) {
      (void)lightglue_worker->wait_result();
    }
    throw;
  }
  if (map_window) {
    map_window->close();
  }
  if (has_pending_lightglue_job && lightglue_worker) {
    (void)lightglue_worker->wait_result();
  }
  if (errors.empty()) {
    throw std::runtime_error("No frames processed");
  }
  const bool csv_was_enabled = csv_writer.enabled();
  const auto csv_close_start = profiling_enabled ? Clock::now() : Clock::time_point{};
  csv_writer.close();
  if (profiling_enabled && csv_was_enabled) {
    profile.csv_ms += elapsed_ms(csv_close_start);
  }

  const double runtime_s = std::chrono::duration<double>(Clock::now() - run_start).count();
  double sum_sq = 0.0;
  double sum_abs = 0.0;
  double max_error = 0.0;
  for (double error : errors) {
    sum_sq += error * error;
    sum_abs += std::abs(error);
    max_error = std::max(max_error, error);
  }

  VideoTestResult result;
  result.frame_count = frame_idx;
  result.correction_count = correction_count;
  result.runtime_s = runtime_s;
  result.processed_fps = static_cast<double>(frame_idx) / std::max(runtime_s, 1e-9);
  result.rmse_m = std::sqrt(sum_sq / static_cast<double>(errors.size()));
  result.mae_m = sum_abs / static_cast<double>(errors.size());
  result.cep95_m = percentile(errors, 95.0);
  result.max_error_m = max_error;
  result.mean_klt_ms = klt_times_ms.empty() ? 0.0 :
    std::accumulate(klt_times_ms.begin(), klt_times_ms.end(), 0.0) / static_cast<double>(klt_times_ms.size());
  result.mean_visual_ms = visual_times_ms.empty() ? 0.0 :
    std::accumulate(visual_times_ms.begin(), visual_times_ms.end(), 0.0) / static_cast<double>(visual_times_ms.size());
  result.timestamps_s = std::move(timestamps);
  result.error_m = std::move(errors);
  const FrameSourceProfile source_profile = frame_source->profile();
  profile.decode_read_ms = source_profile.decode_ms;
  profile.decode_wait_ms = source_profile.wait_ms;
  profile.frame_source = source_profile;
  result.profile = profile;
  write_profile_report(options.profile_output_path, result.profile);
  write_realtime_profile_report(options.realtime_profile_output_path, result);
  return result;
}

}  // namespace advanced_localization
