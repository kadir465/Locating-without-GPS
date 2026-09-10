from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray

Vector2 = NDArray[np.float64]
Vector3 = NDArray[np.float64]
Vector4 = NDArray[np.float64]
Matrix3 = NDArray[np.float64]
Matrix16 = NDArray[np.float64]


def _to_array(value: NDArray[np.float64] | list[float], shape: tuple[int, ...], name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


@dataclass(frozen=True)
class ImuSample:
    timestamp_s: float
    accel_mps2: Vector3
    gyro_rps: Vector3

    def __post_init__(self) -> None:
        object.__setattr__(self, "accel_mps2", _to_array(self.accel_mps2, (3,), "accel_mps2"))
        object.__setattr__(self, "gyro_rps", _to_array(self.gyro_rps, (3,), "gyro_rps"))


@dataclass(frozen=True)
class VisionMeasurement:
    timestamp_s: float
    position_xy_enu_m: Vector2
    yaw_rad: float
    covariance: NDArray[np.float64]
    source: str
    processing_latency_s: float = 0.0
    scale: float = 1.0
    inlier_ratio: float = 1.0
    vision_dof: int = 3
    yaw_valid: bool = True
    model: str = "SE2_PLANAR"
    match_confidence: float = 1.0
    match_count: int = 0
    inlier_count: int = 0
    reprojection_rmse_px: float = float("nan")
    failsafe_mode: bool = False
    confidence_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.vision_dof not in (2, 3):
            raise ValueError(f"vision_dof must be 2 or 3, got {self.vision_dof}")
        object.__setattr__(self, "position_xy_enu_m", _to_array(self.position_xy_enu_m, (2,), "position_xy_enu_m"))
        expected_shape = (self.vision_dof, self.vision_dof)
        object.__setattr__(self, "covariance", _to_array(self.covariance, expected_shape, "covariance"))
        if self.vision_dof == 2 and self.yaw_valid:
            object.__setattr__(self, "yaw_valid", False)


@dataclass(frozen=True)
class FilterSnapshot:
    timestamp_s: float
    position_enu_m: Vector3
    velocity_enu_mps: Vector3
    quaternion_wxyz: Vector4
    accel_bias_mps2: Vector3
    gyro_bias_rps: Vector3
    time_delay_s: float
    covariance_16x16: Matrix16

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_enu_m", _to_array(self.position_enu_m, (3,), "position_enu_m"))
        object.__setattr__(self, "velocity_enu_mps", _to_array(self.velocity_enu_mps, (3,), "velocity_enu_mps"))
        object.__setattr__(self, "quaternion_wxyz", _to_array(self.quaternion_wxyz, (4,), "quaternion_wxyz"))
        object.__setattr__(self, "accel_bias_mps2", _to_array(self.accel_bias_mps2, (3,), "accel_bias_mps2"))
        object.__setattr__(self, "gyro_bias_rps", _to_array(self.gyro_bias_rps, (3,), "gyro_bias_rps"))
        object.__setattr__(self, "covariance_16x16", _to_array(self.covariance_16x16, (16, 16), "covariance_16x16"))


@dataclass(frozen=True)
class FlowMeasurement:
    timestamp_s: float
    velocity_xy_mps: Vector2
    covariance_2x2: NDArray[np.float64]
    source: str = "optical_flow"
    feature_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "velocity_xy_mps", _to_array(self.velocity_xy_mps, (2,), "velocity_xy_mps"))
        object.__setattr__(self, "covariance_2x2", _to_array(self.covariance_2x2, (2, 2), "covariance_2x2"))


class FailsafeCode(str, Enum):
    NONE = "NONE"
    VISION_LOSS = "VISION_LOSS"
    INCONSISTENCY = "INCONSISTENCY"
    HIGH_COVARIANCE = "HIGH_COVARIANCE"


@dataclass(frozen=True)
class UpdateResult:
    accepted: bool
    reason: str
    source: str
    mahalanobis_distance: Optional[float] = None
    innovation: Optional[NDArray[np.float64]] = None
