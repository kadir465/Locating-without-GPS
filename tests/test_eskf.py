from __future__ import annotations

import numpy as np

from advanced_localization.config import SystemConfig
from advanced_localization.eskf import CascadedESKF
from advanced_localization.types import FlowMeasurement, ImuSample, VisionMeasurement


def _stationary_imu(timestamp_s: float) -> ImuSample:
    return ImuSample(
        timestamp_s=timestamp_s,
        accel_mps2=np.array([0.0, 0.0, 9.80665], dtype=np.float64),
        gyro_rps=np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )


def test_mahalanobis_gate_rejects_large_outlier() -> None:
    config = SystemConfig()
    config.gate.vmax_mps = 1e6
    config.gate.kinematic_tolerance_m = 1e6

    filt = CascadedESKF(config)
    filt.initialize(position_enu_m=np.zeros(3, dtype=np.float64), yaw_rad=0.0, timestamp_s=0.0)

    measurement = VisionMeasurement(
        timestamp_s=0.0,
        position_xy_enu_m=np.array([100.0, 100.0], dtype=np.float64),
        yaw_rad=0.0,
        covariance=np.diag(np.array([0.25, 0.25, np.deg2rad(1.0) ** 2], dtype=np.float64)),
        source="LightGlue",
        vision_dof=3,
        model="SE2_PLANAR",
    )

    result = filt.update_from_vision(measurement)
    assert not result.accepted
    assert result.reason == "mahalanobis_gate_rejected"


def test_position_and_yaw_correction_are_clipped() -> None:
    config = SystemConfig()
    filt = CascadedESKF(config)
    filt.initialize(position_enu_m=np.zeros(3, dtype=np.float64), yaw_rad=0.0, timestamp_s=0.0)

    measurement = VisionMeasurement(
        timestamp_s=0.0,
        position_xy_enu_m=np.array([10.0, 0.0], dtype=np.float64),
        yaw_rad=np.deg2rad(45.0),
        covariance=np.diag(np.array([100.0, 100.0, np.deg2rad(30.0) ** 2], dtype=np.float64)),
        source="lightglue",
        vision_dof=3,
        model="SE2_PLANAR",
    )

    result = filt.update_from_vision(measurement)
    assert result.accepted

    pos_norm = np.linalg.norm(filt.position_enu_m[0:2])
    assert pos_norm <= config.gate.max_position_correction_m + 1e-6
    assert abs(filt.yaw_rad) <= np.deg2rad(config.gate.max_yaw_correction_deg) + 1e-6


def test_historical_update_replays_imu_to_current_time() -> None:
    config = SystemConfig()
    filt = CascadedESKF(config)
    filt.initialize(position_enu_m=np.zeros(3, dtype=np.float64), yaw_rad=0.0, timestamp_s=0.0)

    for index in range(1, 51):
        filt.predict_from_imu(_stationary_imu(index * 0.01))

    measurement = VisionMeasurement(
        timestamp_s=0.25,
        position_xy_enu_m=np.array([0.2, -0.1], dtype=np.float64),
        yaw_rad=np.deg2rad(2.0),
        covariance=np.diag(np.array([1.0, 1.0, np.deg2rad(5.0) ** 2], dtype=np.float64)),
        source="LightGlue",
        vision_dof=3,
        model="SE2_PLANAR",
    )

    result = filt.apply_historical_vision_update(measurement)
    assert result.accepted
    assert filt.timestamp_s is not None
    assert abs(filt.timestamp_s - 0.5) < 1e-6


def test_historical_update_replays_flow_history_to_present() -> None:
    config = SystemConfig()
    config.gate.vmax_mps = 1e6
    config.gate.kinematic_tolerance_m = 1e6

    filt = CascadedESKF(config)
    filt.initialize(position_enu_m=np.zeros(3, dtype=np.float64), yaw_rad=0.0, timestamp_s=0.0)

    for index in range(1, 21):
        filt.predict_from_imu(_stationary_imu(index * 0.01))

    flow = FlowMeasurement(
        timestamp_s=0.2,
        velocity_xy_mps=np.array([1.5, 0.0], dtype=np.float64),
        covariance_2x2=np.diag(np.array([0.01, 0.01], dtype=np.float64)),
    )
    flow_result = filt.update_from_flow(flow)
    assert flow_result.accepted

    for index in range(21, 51):
        filt.predict_from_imu(_stationary_imu(index * 0.01))

    measurement = VisionMeasurement(
        timestamp_s=0.15,
        position_xy_enu_m=np.array([0.1, -0.05], dtype=np.float64),
        yaw_rad=0.0,
        covariance=np.diag(np.array([2.0, 2.0], dtype=np.float64)),
        source="LightGlue",
        vision_dof=2,
        yaw_valid=False,
        model="LIGHTGLUE_SUPERPOINT_XY_TRANSLATION",
    )

    result = filt.apply_historical_vision_update(measurement)
    assert result.accepted
    assert filt.timestamp_s is not None
    assert abs(filt.timestamp_s - 0.5) < 1e-6
    assert float(filt.velocity_enu_mps[0]) > 0.2


def test_historical_update_preserves_lightglue_quality_fields() -> None:
    config = SystemConfig()
    config.gate.vmax_mps = 1e6
    config.gate.kinematic_tolerance_m = 1e6

    filt = CascadedESKF(config)
    filt.initialize(position_enu_m=np.zeros(3, dtype=np.float64), yaw_rad=0.0, timestamp_s=0.0)

    for index in range(1, 11):
        filt.predict_from_imu(_stationary_imu(index * 0.01))

    measurement = VisionMeasurement(
        timestamp_s=0.05,
        position_xy_enu_m=np.array([0.1, -0.05], dtype=np.float64),
        yaw_rad=0.0,
        covariance=np.diag(np.array([2.0, 2.0], dtype=np.float64)),
        source="LightGlue",
        vision_dof=2,
        yaw_valid=False,
        model="LIGHTGLUE_SUPERPOINT_ROBUST_SHIFT",
        match_confidence=0.42,
        match_count=31,
        inlier_count=12,
        reprojection_rmse_px=1.7,
        failsafe_mode=True,
        confidence_weight=0.33,
    )
    captured_measurements: list[VisionMeasurement] = []
    original_update_from_vision = filt.update_from_vision

    def _capture_update(replayed_measurement: VisionMeasurement):
        captured_measurements.append(replayed_measurement)
        return original_update_from_vision(replayed_measurement)

    filt.update_from_vision = _capture_update  # type: ignore[method-assign]

    result = filt.apply_historical_vision_update(measurement)

    assert result.accepted
    assert len(captured_measurements) == 1
    assert captured_measurements[0].match_confidence == measurement.match_confidence
    assert captured_measurements[0].match_count == measurement.match_count
    assert captured_measurements[0].inlier_count == measurement.inlier_count
    assert captured_measurements[0].reprojection_rmse_px == measurement.reprojection_rmse_px
    assert captured_measurements[0].failsafe_mode == measurement.failsafe_mode
    assert captured_measurements[0].confidence_weight == measurement.confidence_weight
    assert filt.health.consecutive_lightglue_rejections == 0


def test_flow_update_rejected_when_pzz_is_high() -> None:
    config = SystemConfig()
    filt = CascadedESKF(config)
    filt.initialize(position_enu_m=np.zeros(3, dtype=np.float64), yaw_rad=0.0, timestamp_s=0.0)
    filt.inflate_altitude_uncertainty(config.flow.pzz_reject_threshold_m2 + 50.0)

    flow = FlowMeasurement(
        timestamp_s=0.1,
        velocity_xy_mps=np.array([0.5, -0.2], dtype=np.float64),
        covariance_2x2=np.diag(np.array([0.4, 0.4], dtype=np.float64)),
    )

    result = filt.update_from_flow(flow)
    assert not result.accepted
    assert result.reason == "flow_rejected_pzz_high"


def test_lightglue_rejection_counter_increases_for_lightglue_sources() -> None:
    config = SystemConfig()
    config.gate.vmax_mps = 1e6
    config.gate.kinematic_tolerance_m = 1e6

    filt = CascadedESKF(config)
    filt.initialize(position_enu_m=np.zeros(3, dtype=np.float64), yaw_rad=0.0, timestamp_s=0.0)

    lightglue_bad = VisionMeasurement(
        timestamp_s=0.1,
        position_xy_enu_m=np.array([200.0, -120.0], dtype=np.float64),
        yaw_rad=0.0,
        covariance=np.diag(np.array([0.2, 0.2, np.deg2rad(1.0) ** 2], dtype=np.float64)),
        source="lightglue",
        vision_dof=3,
        model="SE2_PLANAR",
    )
    filt.update_from_vision(lightglue_bad)
    assert filt.health.consecutive_lightglue_rejections == 1

    lightglue_bad_upper = VisionMeasurement(
        timestamp_s=0.2,
        position_xy_enu_m=np.array([250.0, -200.0], dtype=np.float64),
        yaw_rad=0.0,
        covariance=np.diag(np.array([0.2, 0.2, np.deg2rad(1.0) ** 2], dtype=np.float64)),
        source="LightGlue",
        vision_dof=3,
        model="SE2_PLANAR",
    )
    filt.update_from_vision(lightglue_bad_upper)
    assert filt.health.consecutive_lightglue_rejections == 2

