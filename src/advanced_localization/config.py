from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
from numpy.typing import NDArray


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass
class NoiseConfig:
    accel_noise_std_mps2: float = 0.35
    gyro_noise_std_rps: float = 0.03
    accel_bias_rw_std_mps2: float = 0.01
    gyro_bias_rw_std_rps: float = 0.0015


@dataclass
class GateConfig:
    chi2_threshold_3dof: float = 9.0
    chi2_threshold_2dof: float = 5.99
    max_position_correction_m: float = 3.0
    max_yaw_correction_deg: float = 5.0
    vmax_mps: float = 15.0
    kinematic_tolerance_m: float = 5.0
    max_consecutive_rejections: int = 5
    max_consecutive_vision_loss: int = 8
    max_position_trace_m2: float = 100.0
    min_measurement_variance_m2: float = 0.05
    inconsistency_threshold_m: float = 25.0


@dataclass
class TimingConfig:
    lightglue_budget_ms: float = 80.0
    total_vision_budget_ms: float = 500.0
    history_buffer_seconds: float = 0.5
    initial_time_delay_s: float = 0.0
    max_time_delay_s: float = 0.5


@dataclass
class FlowConfig:
    rate_hz: float = 10.0
    chi2_threshold_2dof: float = 5.99
    pzz_reject_threshold_m2: float = 16.0
    noise_std_mps: float = 0.8
    min_features: int = 40
    max_bidirectional_error_px: float = 1.5
    reference_altitude_m: float = 30.0


@dataclass
class LightGlueConfig:
    enabled: bool = True
    require_cuda: bool = True
    device: str = "auto"
    image_size_px: int = 256
    use_amp: bool = True
    min_matches: int = 10
    min_inliers: int = 6
    min_confidence: float = 0.35
    min_inlier_ratio: float = 0.30
    ransac_reproj_threshold_px: float = 3.0
    filter_threshold: float = 0.10
    depth_confidence: float = 0.95
    width_confidence: float = 0.99
    gpu_target_duty_cycle: float = 0.60
    gpu_min_interval_s: float = 0.08
    gpu_max_interval_s: float = 5.00
    yaw_hypotheses_per_job: int = 1
    warmup_runs: int = 1
    use_clahe: bool = True
    roi_base_window_m: float = 20.0
    roi_prediction_latency_s: float = 0.20
    roi_max_window_m: float = 50.0
    roi_padding_sigma: float = 3.0
    residual_gate_m: float = 25.0
    scale_consistency_min: float = 0.50
    scale_consistency_max: float = 1.50
    scale_consistency_min_distance_m: float = 1.0
    trust_gain_min: float = 0.05
    trust_gain_max: float = 0.70


@dataclass
class SuperPointConfig:
    max_keypoints: int = 2048
    keypoint_threshold: float = 0.005
    nms_radius: int = 4


@dataclass
class ObservabilityConfig:
    velocity_trace_limit_m2ps2: float = 25.0
    trigger_once: bool = True
    command_id: int = 31000


@dataclass
class CameraCalibrationConfig:
    info_yaml_path: str = "config/camera_info_dummy.yaml"
    image_width: int = 640
    image_height: int = 480
    camera_matrix: NDArray[np.float64] = field(
        default_factory=lambda: np.array(
            [
                [500.0, 0.0, 320.0],
                [0.0, 500.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )
    distortion_coeffs: NDArray[np.float64] = field(default_factory=lambda: np.zeros(5, dtype=np.float64))


@dataclass
class SystemConfig:
    imu_rate_hz: float = 100.0
    gravity_enu_mps2: NDArray[np.float64] = field(default_factory=lambda: np.array([0.0, 0.0, -9.80665], dtype=np.float64))
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    lightglue: LightGlueConfig = field(default_factory=LightGlueConfig)
    superpoint: SuperPointConfig = field(default_factory=SuperPointConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    camera: CameraCalibrationConfig = field(default_factory=CameraCalibrationConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "SystemConfig":
        path = Path(yaml_path)
        if not path.exists():
            return cls()

        with path.open("r", encoding="utf-8") as file:
            raw: Dict[str, Any] = yaml.safe_load(file) or {}

        config = cls()
        if "imu_rate_hz" in raw:
            config.imu_rate_hz = float(raw["imu_rate_hz"])
        if "gravity_enu_mps2" in raw:
            config.gravity_enu_mps2 = np.asarray(raw["gravity_enu_mps2"], dtype=np.float64)

        noise = raw.get("noise", {})
        for key in vars(config.noise).keys():
            if key in noise:
                setattr(config.noise, key, float(noise[key]))

        gate = raw.get("gate", {})
        for key in vars(config.gate).keys():
            if key in gate:
                value = gate[key]
                cast = int if key in {"max_consecutive_rejections", "max_consecutive_vision_loss"} else float
                setattr(config.gate, key, cast(value))

        timing = raw.get("timing", {})
        for key in vars(config.timing).keys():
            if key in timing:
                setattr(config.timing, key, float(timing[key]))

        flow = raw.get("flow", {})
        for key in vars(config.flow).keys():
            if key in flow:
                value = flow[key]
                cast = int if key == "min_features" else float
                setattr(config.flow, key, cast(value))

        lightglue = raw.get("lightglue", {})
        for key in vars(config.lightglue).keys():
            if key not in lightglue:
                continue
            value = lightglue[key]
            if key in {"enabled", "require_cuda", "use_clahe", "use_amp"}:
                setattr(config.lightglue, key, _to_bool(value))
            elif key == "device":
                setattr(config.lightglue, key, str(value))
            elif key in {"image_size_px", "min_matches", "min_inliers", "yaw_hypotheses_per_job", "warmup_runs"}:
                setattr(config.lightglue, key, int(value))
            else:
                setattr(config.lightglue, key, float(value))

        superpoint = raw.get("superpoint", {})
        for key in vars(config.superpoint).keys():
            if key not in superpoint:
                continue
            value = superpoint[key]
            if key in {"max_keypoints", "nms_radius"}:
                setattr(config.superpoint, key, int(value))
            else:
                setattr(config.superpoint, key, float(value))

        observability = raw.get("observability", {})
        for key in vars(config.observability).keys():
            if key in observability:
                value = observability[key]
                if key == "trigger_once":
                    setattr(config.observability, key, _to_bool(value))
                elif key == "command_id":
                    setattr(config.observability, key, int(value))
                else:
                    setattr(config.observability, key, float(value))

        camera = raw.get("camera", {})
        if "info_yaml_path" in camera:
            config.camera.info_yaml_path = str(camera["info_yaml_path"])
        if "image_width" in camera:
            config.camera.image_width = int(camera["image_width"])
        if "image_height" in camera:
            config.camera.image_height = int(camera["image_height"])
        if "camera_matrix" in camera:
            config.camera.camera_matrix = np.asarray(camera["camera_matrix"], dtype=np.float64).reshape(3, 3)
        if "distortion_coeffs" in camera:
            config.camera.distortion_coeffs = np.asarray(camera["distortion_coeffs"], dtype=np.float64)

        return config
