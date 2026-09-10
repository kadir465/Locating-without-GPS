from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from advanced_localization.video_testing import parse_dji_srt, run_vision_only_video_test


def _resolve_existing_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.exists():
        return candidate

    root_candidate = ROOT / path_str
    if root_candidate.exists():
        return root_candidate

    raise FileNotFoundError(
        f"Path not found: '{path_str}'. Tried '{candidate.resolve()}' and '{root_candidate.resolve()}'."
    )


def _resolve_output_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _default_map_output_for_video(video_path: Path) -> Path:
    stem = video_path.stem
    if not stem:
        raise ValueError(f"Unable to derive map output name from video path: {video_path}")
    return ROOT / "outputs" / f"{stem}.tif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision-only video test pipeline (KLT + periodic LightGlue correction)")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--srt", required=True, help="Path to synchronized DJI SRT file")
    parser.add_argument("--camera-fov-deg", type=float, default=84.0, help="Horizontal camera field of view in degrees")
    parser.add_argument(
        "--altitude-m",
        type=float,
        default=100.0,
        help="Deprecated in video tests; altitude is fixed to 100m",
    )
    parser.add_argument("--correction-interval-s", type=float, default=1.0, help="LightGlue absolute correction period in seconds")
    parser.add_argument(
        "--correction-noise-std-m",
        type=float,
        default=0.2,
        help="Deprecated legacy argument kept for compatibility; ignored by real LightGlue correction",
    )
    parser.add_argument("--disable-correction", action="store_true", help="Disable periodic absolute correction")
    parser.add_argument(
        "--lightglue-config-yaml",
        default="config/system_config_video_lightglue.yaml",
        help="Path to LightGlue/system config YAML (contains device, checkpoint_path, thresholds)",
    )
    parser.add_argument(
        "--lightglue-window-size-m",
        type=float,
        default=60.0,
        help="ENU map crop window size in meters used during LightGlue correction (default 70m)",
    )
    parser.add_argument(
        "--lightglue-map-roi-size-px",
        type=int,
        default=500,
        help="Raw map ROI side length in pixels before preprocessing (default 500)",
    )
    parser.add_argument(
        "--lightglue-patch-size-px",
        type=int,
        default=None,
        help="Square patch size in pixels used after pre-processing (default: config YAML image_size_px)",
    )
    parser.add_argument(
        "--lightglue-confidence-gate",
        type=float,
        default=None,
        help="Reject LightGlue corrections at or below this confidence threshold (default: config YAML min_confidence)",
    )
    parser.add_argument(
        "--lightglue-inlier-gate",
        type=float,
        default=0.30,
        help="Legacy option (ignored): LightGlue acceptance now uses confidence and jump gates only",
    )
    parser.add_argument(
        "--lightglue-mahalanobis-gate",
        type=float,
        default=9.21,
        help="Reject LightGlue corrections when Mahalanobis distance exceeds this threshold",
    )
    parser.add_argument(
        "--lightglue-max-jump-m",
        type=float,
        default=50.0,
        help="Reject LightGlue corrections with ENU jump larger than this value (meters)",
    )
    parser.add_argument(
        "--lightglue-residual-gate-m",
        type=float,
        default=25.0,
        help="Legacy residual gate parameter (Mahalanobis gate is primary in current flow)",
    )
    parser.add_argument(
        "--lightglue-meas-scale-min",
        type=float,
        default=0.5,
        help="Reject LightGlue corrections when measured residual scale is below this bound",
    )
    parser.add_argument(
        "--lightglue-meas-scale-max",
        type=float,
        default=1.5,
        help="Reject LightGlue corrections when measured residual scale is above this bound",
    )
    parser.add_argument(
        "--lightglue-blend-gain-min",
        type=float,
        default=0.05,
        help="Minimum trust gain applied to LightGlue innovation",
    )
    parser.add_argument(
        "--lightglue-blend-gain-max",
        type=float,
        default=0.70,
        help="Maximum trust gain applied to LightGlue innovation",
    )
    parser.add_argument(
        "--lightglue-max-yaw-hypotheses",
        type=int,
        default=5,
        help="Maximum yaw hypotheses evaluated per LightGlue correction (1-8)",
    )
    parser.add_argument(
        "--lightglue-hypotheses-per-job",
        type=int,
        default=1,
        help="Yaw hypotheses evaluated per correction job (lower values reduce GPU spikes)",
    )
    parser.add_argument(
        "--lightglue-gpu-target-duty-cycle",
        type=float,
        default=0.35,
        help="Target GPU duty-cycle share for LightGlue jobs (0.05-0.95)",
    )
    parser.add_argument(
        "--lightglue-gpu-min-interval-s",
        type=float,
        default=0.35,
        help="Lower bound for adaptive LightGlue correction interval (seconds)",
    )
    parser.add_argument(
        "--lightglue-gpu-max-interval-s",
        type=float,
        default=5.0,
        help="Upper bound for adaptive LightGlue correction interval (seconds)",
    )
    parser.add_argument(
        "--lightglue-max-result-age-s",
        type=float,
        default=3.5,
        help="Legacy option (ignored): result-age gate is disabled in current correction flow",
    )
    parser.add_argument(
        "--legacy-flow-yaw",
        action="store_true",
        help="Use flow/correction displacement yaw instead of SRT heading. Default uses SRT heading as compass proxy.",
    )
    parser.add_argument(
        "--use-srt-heading",
        action="store_true",
        help="Compatibility flag; SRT heading is already enabled by default as the video-test compass proxy.",
    )
    parser.add_argument(
        "--enable-srt-direction-lock",
        action="store_true",
        help="Compatibility flag; SRT direction lock is already enabled by default with SRT heading.",
    )
    parser.add_argument(
        "--disable-srt-direction-lock",
        action="store_true",
        help="Disable projecting flow/correction updates onto SRT heading direction.",
    )
    parser.add_argument(
        "--sync-lightglue",
        action="store_true",
        help="Run LightGlue correction synchronously (can cause frame-loop pauses)",
    )
    parser.add_argument(
        "--disable-srt-scale-assist",
        action="store_true",
        help="Deprecated compatibility flag; SRT scale assist is disabled by default.",
    )
    parser.add_argument(
        "--enable-srt-scale-assist",
        action="store_true",
        help="Use SRT speed-based online scale assist for ablation only.",
    )
    parser.add_argument(
        "--vio-rate-hz",
        type=float,
        default=10.0,
        help="VIO flow update rate in Hz (default 10.0). Lower values reduce processing load.",
    )
    parser.add_argument("--csv-output", default="", help="Optional CSV output path for per-frame metrics")
    parser.add_argument("--log-output", default="", help="Optional text file path for saving console logs")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap for processed frames (0 means full video)")

    parser.add_argument("--download-map", action="store_true", help="Download map from Google Maps XYZ before running video test")
    parser.add_argument("--map-center-lat", type=float, default=None, help="Map center latitude (default: first SRT sample)")
    parser.add_argument("--map-center-lon", type=float, default=None, help="Map center longitude (default: first SRT sample)")
    parser.add_argument("--map-radius-m", type=float, default=2000.0, help="Map download radius in meters")
    parser.add_argument(
        "--map-output",
        default=None,
        help="Output path for downloaded map (default: outputs/<video-name>.tif)",
    )

    parser.add_argument("--map-path", default="", help="Optional offline map path (GeoTIFF/JPG/PNG) for live point&trail view")
    parser.add_argument("--map-gsd", type=float, default=0.2, help="Map resolution in meters per pixel")
    parser.add_argument("--show-map-window", action="store_true", help="Open live map window with estimated point and trail")
    parser.add_argument("--map-window-name", default="Map Point&Trail", help="Title of the live map visualization window")
    parser.add_argument("--map-max-size", type=int, default=1000, help="Maximum displayed map side in pixels")
    parser.add_argument(
        "--map-window-video-output",
        default="outputs/video_test_window.mp4",
        help="Output path for recording the live map window video (empty string disables recording)",
    )
    parser.add_argument(
        "--map-window-video-fps",
        type=float,
        default=0.0,
        help="FPS for the saved map-window video (0 uses source video FPS)",
    )
    parser.add_argument("--disable-clahe", action="store_true", help="Disable CLAHE contrast enhancement to improve speed")
    parser.add_argument("--verbose", action="store_true", help="Print per-second runtime logs")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for correction noise")
    return parser.parse_args()


class TeeLogger:
    def __init__(self, filename: Path):
        self.terminal = sys.stdout
        filename.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

def main() -> None:
    run_start_s = perf_counter()
    args = parse_args()

    if args.log_output:
        log_path = _resolve_output_path(args.log_output)
        sys.stdout = TeeLogger(log_path)
        sys.stderr = sys.stdout

    video_path = _resolve_existing_path(args.video)
    srt_path = _resolve_existing_path(args.srt)
    map_path = _resolve_existing_path(args.map_path) if args.map_path else None
    csv_path = _resolve_output_path(args.csv_output) if args.csv_output else None
    map_window_video_path = (
        _resolve_output_path(args.map_window_video_output)
        if args.show_map_window and args.map_window_video_output
        else None
    )
    map_window_video_fps = float(args.map_window_video_fps) if args.map_window_video_fps > 0.0 else None
    lightglue_config_yaml = _resolve_output_path(args.lightglue_config_yaml) if args.lightglue_config_yaml else None

    if args.download_map:
        if args.map_center_lat is not None and args.map_center_lon is not None:
            center_lat = float(args.map_center_lat)
            center_lon = float(args.map_center_lon)
        else:
            gt_samples = parse_dji_srt(srt_path)
            center_lat = float(gt_samples[0].latitude_deg)
            center_lon = float(gt_samples[0].longitude_deg)

        map_output_path = (
            _resolve_output_path(args.map_output)
            if args.map_output
            else _default_map_output_for_video(video_path)
        )

        from preflight_map_pipeline import run_pipeline as run_map_download_pipeline

        print("Downloading map from Google Maps XYZ...")
        map_jpg_path, actual_mpp = run_map_download_pipeline(
            center_lat_deg=center_lat,
            center_lon_deg=center_lon,
            radius_m=float(args.map_radius_m),
            gsd_m_per_px=float(args.map_gsd),
            output_path=map_output_path,
        )
        map_path = map_jpg_path
        args.map_gsd = actual_mpp

    if args.show_map_window and map_path is None:
        raise ValueError("--show-map-window requires either --map-path or --download-map")
    if not args.disable_correction and map_path is None:
        raise ValueError("LightGlue correction requires --map-path or --download-map")

    result = run_vision_only_video_test(
        video_path=video_path,
        srt_path=srt_path,
        camera_fov_deg=args.camera_fov_deg,
        test_altitude_m=args.altitude_m,
        correction_interval_s=args.correction_interval_s,
        correction_noise_std_m=args.correction_noise_std_m,
        enable_correction=not args.disable_correction,
        verbose=args.verbose,
        random_seed=args.seed,
        lightglue_config_yaml=lightglue_config_yaml,
        lightglue_window_size_m=args.lightglue_window_size_m,
        lightglue_map_roi_size_px=args.lightglue_map_roi_size_px,
        lightglue_patch_size_px=args.lightglue_patch_size_px,
        lightglue_confidence_gate=args.lightglue_confidence_gate,
        lightglue_inlier_gate=args.lightglue_inlier_gate,
        lightglue_jump_gate_m=args.lightglue_max_jump_m,
        lightglue_residual_gate_m=args.lightglue_residual_gate_m,
        lightglue_meas_scale_min=args.lightglue_meas_scale_min,
        lightglue_meas_scale_max=args.lightglue_meas_scale_max,
        lightglue_blend_gain_min=args.lightglue_blend_gain_min,
        lightglue_blend_gain_max=args.lightglue_blend_gain_max,
        lightglue_mahalanobis_gate=args.lightglue_mahalanobis_gate,
        lightglue_max_yaw_hypotheses=args.lightglue_max_yaw_hypotheses,
        lightglue_hypotheses_per_job=args.lightglue_hypotheses_per_job,
        lightglue_gpu_target_duty_cycle=args.lightglue_gpu_target_duty_cycle,
        lightglue_gpu_min_interval_s=args.lightglue_gpu_min_interval_s,
        lightglue_gpu_max_interval_s=args.lightglue_gpu_max_interval_s,
        lightglue_max_result_age_s=args.lightglue_max_result_age_s,
        enable_srt_scale_assist=args.enable_srt_scale_assist and not args.disable_srt_scale_assist,
        use_gt_heading=not args.legacy_flow_yaw,
        force_srt_direction_lock=not args.disable_srt_direction_lock,
        vio_rate_hz=args.vio_rate_hz,
        csv_output_path=csv_path,
        max_frames=args.max_frames if args.max_frames > 0 else None,
        map_path=map_path,
        map_gsd_m_per_px=args.map_gsd,
        show_map_window=args.show_map_window,
        map_window_name=args.map_window_name,
        map_max_display_size_px=args.map_max_size,
        map_window_video_output_path=map_window_video_path,
        map_window_video_fps=map_window_video_fps,
        use_clahe=not args.disable_clahe,
        enable_async_LightGlue=not args.sync_lightglue,
    )

    print("=== Vision-Only Test Summary ===")
    print(f"frames={result.frame_count}")
    print(f"corrections={result.correction_count}")
    print(f"runtime_s={perf_counter() - run_start_s:.3f}")
    print(f"rmse_m={result.rmse_m:.3f}")
    print(f"mae_m={result.mae_m:.3f}")
    print(f"cep95_m={result.cep95_m:.3f}")
    print(f"max_error_m={result.max_error_m:.3f}")


if __name__ == "__main__":
    main()


