#include "advanced_localization_cpp/video_testing.hpp"

#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <cstdlib>

namespace advanced_localization
{
namespace
{

[[noreturn]] void throw_usage(const std::string & message)
{
  throw std::invalid_argument(message + "\nRun video_testing_runner --help for usage.");
}

std::string require_value(int & index, int argc, char ** argv)
{
  if (index + 1 >= argc) {
    throw_usage(std::string("Missing value for ") + argv[index]);
  }
  ++index;
  return argv[index];
}

void print_help()
{
  std::cout
    << "C++ vision-only video test pipeline (KLT + optional LightGlue-compatible correction)\n\n"
    << "Required:\n"
    << "  --video PATH\n"
    << "  --srt PATH                         Required for full metric runs; optional for --decode-only/--klt-only\n\n"
    << "Core options:\n"
    << "  --csv-output PATH\n"
    << "  --max-frames N\n"
    << "  --benchmark-preset parity|fast-wsl|orin-file|flight\n"
    << "  --frame-source opencv|gst-nvdec|gst-nvdec-gray|image-sequence|raw-gray\n"
    << "  --gst-pipeline PIPELINE           Override the default Orin NVDEC GStreamer pipeline\n"
    << "  --image-sequence-fps FPS          FPS for --frame-source image-sequence\n"
    << "  --raw-width N                     Width for --frame-source raw-gray\n"
    << "  --raw-height N                    Height for --frame-source raw-gray\n"
    << "  --raw-fps FPS                     FPS for --frame-source raw-gray\n"
    << "  --decode-only                     Benchmark frame-source decode/read only; --srt is optional\n"
    << "  --klt-only                        Benchmark frame-source plus KLT only; --srt is optional\n"
    << "  --no-csv                          Disable per-frame CSV even if --csv-output is present\n"
    << "  --camera-fov-deg DEG\n"
    << "  --altitude-m M                    Kept for Python CLI parity; video tests use fixed 100m scale\n"
    << "  --vio-rate-hz HZ\n"
    << "  --flow-max-side-px N              Run KLT on resized grayscale, then rescale flow back (0 disables)\n"
    << "  --sync-decode                     Disable background video decode queue\n"
    << "  --async-decode-queue-size N       Background decode queue capacity (default 4)\n"
    << "  --disable-clahe\n"
    << "  --profile                         Print per-stage timing summary\n"
    << "  --profile-output PATH             Write per-stage timing CSV\n"
    << "  --realtime-profile-output PATH    Write source/backend real-time profile CSV\n"
    << "  --verbose\n\n"
    << "Correction options:\n"
    << "  --disable-correction\n"
    << "  --map-path PATH\n"
    << "  --map-gsd M_PER_PX\n"
    << "  --lightglue-config-yaml PATH\n"
    << "  --lightglue-window-size-m M\n"
    << "  --lightglue-map-roi-size-px N\n"
    << "  --lightglue-patch-size-px N\n"
    << "  --lightglue-confidence-gate X\n"
    << "  --lightglue-mahalanobis-gate X\n"
    << "  --lightglue-max-jump-m M\n"
    << "  --lightglue-residual-gate-m M\n"
    << "  --lightglue-meas-scale-min X\n"
    << "  --lightglue-meas-scale-max X\n"
    << "  --lightglue-blend-gain-min X\n"
    << "  --lightglue-blend-gain-max X\n"
    << "  --lightglue-max-yaw-hypotheses N\n"
    << "  --lightglue-hypotheses-per-job N\n"
    << "  --lightglue-gpu-target-duty-cycle X\n"
    << "  --lightglue-gpu-min-interval-s S\n"
    << "  --lightglue-gpu-max-interval-s S\n"
    << "  --lightglue-max-result-age-s S\n"
    << "  --sync-lightglue\n\n"
    << "Python parity toggles:\n"
    << "  --legacy-flow-yaw\n"
    << "  --disable-srt-direction-lock\n"
    << "  --disable-srt-scale-assist\n";
}

void apply_benchmark_preset(VideoTestingOptions & options, const std::string & preset)
{
  options.benchmark_preset = preset;
  if (preset == "parity") {
    options.frame_source = "opencv";
    options.flow_max_side_px = 0;
    options.csv_enabled = true;
  } else if (preset == "fast-wsl") {
    options.frame_source = "image-sequence";
    options.flow_max_side_px = 1280;
    options.csv_enabled = false;
    options.enable_correction = false;
  } else if (preset == "orin-file") {
    options.frame_source = "gst-nvdec-gray";
    options.flow_max_side_px = 1280;
    options.csv_enabled = false;
    options.enable_async_decode = true;
  } else if (preset == "flight") {
    options.frame_source = "image-sequence";
    options.flow_max_side_px = 1280;
    options.csv_enabled = false;
    options.enable_correction = false;
  } else {
    throw_usage("Unknown benchmark preset: " + preset);
  }
}

VideoTestingOptions parse_args(int argc, char ** argv)
{
  VideoTestingOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help" || key == "-h") {
      print_help();
      std::exit(0);
    } else if (key == "--video") {
      options.video_path = require_value(i, argc, argv);
    } else if (key == "--srt") {
      options.srt_path = require_value(i, argc, argv);
    } else if (key == "--benchmark-preset") {
      apply_benchmark_preset(options, require_value(i, argc, argv));
    } else if (key == "--frame-source") {
      options.frame_source = require_value(i, argc, argv);
    } else if (key == "--gst-pipeline") {
      options.gst_pipeline = require_value(i, argc, argv);
    } else if (key == "--image-sequence-fps") {
      options.image_sequence_fps = std::stod(require_value(i, argc, argv));
    } else if (key == "--raw-width") {
      options.raw_width_px = std::stoi(require_value(i, argc, argv));
    } else if (key == "--raw-height") {
      options.raw_height_px = std::stoi(require_value(i, argc, argv));
    } else if (key == "--raw-fps") {
      options.raw_fps = std::stod(require_value(i, argc, argv));
    } else if (key == "--decode-only") {
      options.decode_only = true;
      options.enable_correction = false;
      options.csv_enabled = false;
    } else if (key == "--klt-only") {
      options.klt_only = true;
      options.enable_correction = false;
      options.csv_enabled = false;
    } else if (key == "--no-csv") {
      options.csv_enabled = false;
    } else if (key == "--camera-fov-deg") {
      options.camera_fov_deg = std::stod(require_value(i, argc, argv));
    } else if (key == "--altitude-m") {
      options.test_altitude_m = std::stod(require_value(i, argc, argv));
    } else if (key == "--correction-interval-s") {
      options.correction_interval_s = std::stod(require_value(i, argc, argv));
    } else if (key == "--correction-noise-std-m") {
      options.correction_noise_std_m = std::stod(require_value(i, argc, argv));
    } else if (key == "--disable-correction") {
      options.enable_correction = false;
    } else if (key == "--lightglue-config-yaml" || key == "--config-yaml") {
      options.lightglue_config_yaml = require_value(i, argc, argv);
    } else if (key == "--lightglue-window-size-m") {
      options.lightglue_window_size_m = std::stod(require_value(i, argc, argv));
    } else if (key == "--lightglue-map-roi-size-px") {
      options.lightglue_map_roi_size_px = std::stoi(require_value(i, argc, argv));
    } else if (key == "--lightglue-patch-size-px") {
      options.lightglue_patch_size_px = std::stoi(require_value(i, argc, argv));
    } else if (key == "--lightglue-confidence-gate") {
      options.lightglue_confidence_gate = std::stod(require_value(i, argc, argv));
    } else if (key == "--lightglue-inlier-gate") {
      options.lightglue_inlier_gate = std::stod(require_value(i, argc, argv));
    } else if (key == "--lightglue-mahalanobis-gate") {
      options.lightglue_mahalanobis_gate = std::stod(require_value(i, argc, argv));
    } else if (key == "--lightglue-max-jump-m") {
      options.lightglue_jump_gate_m = std::stod(require_value(i, argc, argv));
    } else if (key == "--lightglue-residual-gate-m") {
      options.lightglue_residual_gate_m = std::stod(require_value(i, argc, argv));
    } else if (key == "--lightglue-meas-scale-min") {
      options.lightglue_meas_scale_min = std::stod(require_value(i, argc, argv));
    } else if (key == "--lightglue-meas-scale-max") {
      options.lightglue_meas_scale_max = std::stod(require_value(i, argc, argv));
    } else if (key == "--lightglue-blend-gain-min") {
      options.lightglue_blend_gain_min = std::stod(require_value(i, argc, argv));
    } else if (key == "--lightglue-blend-gain-max") {
      options.lightglue_blend_gain_max = std::stod(require_value(i, argc, argv));
    } else if (key == "--lightglue-max-yaw-hypotheses") {
      options.lightglue_max_yaw_hypotheses = std::stoi(require_value(i, argc, argv));
    } else if (key == "--lightglue-hypotheses-per-job") {
      options.lightglue_hypotheses_per_job = std::stoi(require_value(i, argc, argv));
    } else if (key == "--lightglue-gpu-target-duty-cycle") {
      options.lightglue_gpu_target_duty_cycle = std::stod(require_value(i, argc, argv));
    } else if (key == "--lightglue-gpu-min-interval-s") {
      options.lightglue_gpu_min_interval_s = std::stod(require_value(i, argc, argv));
    } else if (key == "--lightglue-gpu-max-interval-s") {
      options.lightglue_gpu_max_interval_s = std::stod(require_value(i, argc, argv));
    } else if (key == "--lightglue-max-result-age-s") {
      options.lightglue_max_result_age_s = std::stod(require_value(i, argc, argv));
    } else if (key == "--legacy-flow-yaw") {
      options.use_gt_heading = false;
    } else if (key == "--disable-srt-direction-lock") {
      options.force_srt_direction_lock = false;
    } else if (key == "--sync-lightglue") {
      options.enable_async_lightglue = false;
    } else if (key == "--disable-srt-scale-assist") {
      options.enable_srt_scale_assist = false;
    } else if (key == "--srt-scale-assist-alpha") {
      options.srt_scale_assist_alpha = std::stod(require_value(i, argc, argv));
    } else if (key == "--vio-rate-hz") {
      options.vio_rate_hz = std::stod(require_value(i, argc, argv));
    } else if (key == "--flow-max-side-px") {
      options.flow_max_side_px = std::stoi(require_value(i, argc, argv));
    } else if (key == "--sync-decode" || key == "--disable-async-decode") {
      options.enable_async_decode = false;
    } else if (key == "--async-decode-queue-size") {
      options.async_decode_queue_size = std::stoi(require_value(i, argc, argv));
    } else if (key == "--csv-output") {
      options.csv_output_path = require_value(i, argc, argv);
    } else if (key == "--max-frames") {
      const int max_frames = std::stoi(require_value(i, argc, argv));
      if (max_frames > 0) {
        options.max_frames = max_frames;
      }
    } else if (key == "--map-path") {
      options.map_path = require_value(i, argc, argv);
    } else if (key == "--map-gsd") {
      options.map_gsd_m_per_px = std::stod(require_value(i, argc, argv));
    } else if (key == "--show-map-window") {
      options.show_map_window = true;
    } else if (key == "--map-window-name") {
      options.map_window_name = require_value(i, argc, argv);
    } else if (key == "--map-max-size") {
      options.map_max_display_size_px = std::stoi(require_value(i, argc, argv));
    } else if (key == "--map-window-video-output") {
      options.map_window_video_output_path = require_value(i, argc, argv);
    } else if (key == "--map-window-video-fps") {
      options.map_window_video_fps = std::stod(require_value(i, argc, argv));
    } else if (key == "--disable-clahe") {
      options.use_clahe = false;
    } else if (key == "--profile") {
      options.profile = true;
    } else if (key == "--profile-output") {
      options.profile_output_path = require_value(i, argc, argv);
    } else if (key == "--realtime-profile-output") {
      options.realtime_profile_output_path = require_value(i, argc, argv);
    } else if (key == "--verbose") {
      options.verbose = true;
    } else if (key == "--seed") {
      options.random_seed = std::stoi(require_value(i, argc, argv));
    } else if (key == "--download-map" || key == "--map-center-lat" || key == "--map-center-lon" ||
      key == "--map-radius-m" || key == "--map-output" || key == "--log-output")
    {
      throw_usage("This C++ runner does not implement map download/log tee. Use the Python map pipeline to prepare --map-path.");
    } else {
      throw_usage("Unknown argument: " + key);
    }
  }
  return options;
}

}  // namespace
}  // namespace advanced_localization

namespace
{

void print_profile_summary(const advanced_localization::VideoTestResult::Profile & profile)
{
  if (!profile.enabled) {
    return;
  }
  const auto mean = [](double total_ms, int count) {
    return count > 0 ? total_ms / static_cast<double>(count) : 0.0;
  };
  const auto & source = profile.frame_source;
  std::cout << "=== C++ Video Test Profile ===\n";
  std::cout << "source_backend=" << source.backend
            << " codec=" << source.codec
            << " size=" << source.width << "x" << source.height
            << " fps=" << source.fps
            << " source_decode_fps=" << source.decode_fps
            << " queue_fill_mean=" << source.queue_fill_mean
            << " dropped_frames=" << source.dropped_frames << '\n';
  std::cout << "decode_read_ms_total=" << profile.decode_read_ms
            << " mean=" << mean(profile.decode_read_ms, profile.frames_read) << '\n';
  std::cout << "decode_wait_ms_total=" << profile.decode_wait_ms
            << " mean=" << mean(profile.decode_wait_ms, profile.frames_read) << '\n';
  std::cout << "gt_interpolation_ms_total=" << profile.gt_interpolation_ms
            << " mean=" << mean(profile.gt_interpolation_ms, profile.frames_read) << '\n';
  std::cout << "klt_ms_total=" << profile.klt_total_ms
            << " mean=" << mean(profile.klt_total_ms, profile.klt_updates) << '\n';
  std::cout << "  klt_grayscale_mean_ms=" << mean(profile.klt_grayscale_ms, profile.klt_updates) << '\n';
  std::cout << "  klt_resize_mean_ms=" << mean(profile.klt_resize_ms, profile.klt_updates) << '\n';
  std::cout << "  klt_clahe_mean_ms=" << mean(profile.klt_clahe_ms, profile.klt_updates) << '\n';
  std::cout << "  klt_feature_mean_ms=" << mean(profile.klt_feature_ms, profile.klt_updates) << '\n';
  std::cout << "  klt_lk_mean_ms=" << mean(profile.klt_lk_ms, profile.klt_updates) << '\n';
  std::cout << "  klt_stats_mean_ms=" << mean(profile.klt_stats_ms, profile.klt_updates) << '\n';
  std::cout << "correction_crop_ms_total=" << profile.correction_crop_ms
            << " mean=" << mean(profile.correction_crop_ms, profile.correction_jobs) << '\n';
  std::cout << "visual_search_ms_total=" << profile.visual_search_ms
            << " mean=" << mean(profile.visual_search_ms, profile.visual_results) << '\n';
  std::cout << "controller_ms_total=" << profile.controller_ms
            << " mean=" << mean(profile.controller_ms, profile.frames_read) << '\n';
  std::cout << "csv_ms_total=" << profile.csv_ms
            << " mean=" << mean(profile.csv_ms, profile.csv_rows) << '\n';
  std::cout << "map_render_ms_total=" << profile.map_render_ms
            << " mean=" << mean(profile.map_render_ms, profile.frames_read) << '\n';
}

}  // namespace

int main(int argc, char ** argv)
{
  try {
    auto options = advanced_localization::parse_args(argc, argv);
    const auto result = advanced_localization::run_vision_only_video_test(options);
    std::cout << "=== Vision-Only C++ Test Summary ===\n";
    std::cout << "frames=" << result.frame_count << '\n';
    std::cout << "corrections=" << result.correction_count << '\n';
    std::cout << "runtime_s=" << result.runtime_s << '\n';
    std::cout << "processed_fps=" << result.processed_fps << '\n';
    std::cout << "mean_klt_ms=" << result.mean_klt_ms << '\n';
    std::cout << "mean_visual_ms=" << result.mean_visual_ms << '\n';
    std::cout << "rmse_m=" << result.rmse_m << '\n';
    std::cout << "mae_m=" << result.mae_m << '\n';
    std::cout << "cep95_m=" << result.cep95_m << '\n';
    std::cout << "max_error_m=" << result.max_error_m << '\n';
    if (options.profile) {
      print_profile_summary(result.profile);
    }
    return 0;
  } catch (const std::exception & exc) {
    std::cerr << "video_testing_runner: " << exc.what() << '\n';
    return 2;
  }
}
