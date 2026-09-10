from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .config import SystemConfig
from .math_utils import (
    clip_vector_norm,
    force_symmetric,
    normalize_angle,
    quaternion_from_small_angle,
    quaternion_from_yaw,
    quaternion_multiply,
    quaternion_normalize,
    rotation_matrix_from_quaternion,
    skew_symmetric,
    yaw_from_quaternion,
)
from .ring_buffer import TimeIndexedRingBuffer
from .types import FilterSnapshot, FlowMeasurement, ImuSample, UpdateResult, VisionMeasurement


@dataclass(frozen=True)
class FilterHealth:
    consecutive_rejections: int
    consecutive_lightglue_rejections: int
    consecutive_vision_loss: int
    position_trace_xy_m2: float
    velocity_trace_m2ps2: float
    pzz_m2: float
    last_update_source: str


class CascadedESKF:
    """16-state delayed ESKF: [p, v, q, b_a, b_g, t_d]."""

    def __init__(self, config: SystemConfig) -> None:
        self._config = config

        self._position_enu_m = np.zeros(3, dtype=np.float64)
        self._velocity_enu_mps = np.zeros(3, dtype=np.float64)
        self._quaternion_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._accel_bias_mps2 = np.zeros(3, dtype=np.float64)
        self._gyro_bias_rps = np.zeros(3, dtype=np.float64)
        self._time_delay_s = float(self._config.timing.initial_time_delay_s)

        self._P = np.eye(16, dtype=np.float64) * 1e-3
        self._P[0:3, 0:3] = np.eye(3, dtype=np.float64) * 2.0
        self._P[3:6, 3:6] = np.eye(3, dtype=np.float64) * 1.0
        self._P[6:9, 6:9] = np.eye(3, dtype=np.float64) * np.deg2rad(10.0)
        self._P[9:12, 9:12] = np.eye(3, dtype=np.float64) * 0.2
        self._P[12:15, 12:15] = np.eye(3, dtype=np.float64) * 0.05
        self._P[15, 15] = 0.02

        self._is_initialized = False
        self._timestamp_s: Optional[float] = None

        self._last_accepted_measurement_time_s: Optional[float] = None
        self._last_accepted_position_xy_m: Optional[NDArray[np.float64]] = None

        self._consecutive_rejections = 0
        self._consecutive_lightglue_rejections = 0
        self._consecutive_vision_loss = 0
        self._last_update_source = "none"

        history_capacity = max(int(self._config.timing.history_buffer_seconds * self._config.imu_rate_hz * 4), 128)
        self._state_history = TimeIndexedRingBuffer[FilterSnapshot](capacity=history_capacity)
        self._imu_history = TimeIndexedRingBuffer[ImuSample](capacity=history_capacity)
        self._flow_history = TimeIndexedRingBuffer[FlowMeasurement](capacity=history_capacity)

        self._F = np.zeros((16, 16), dtype=np.float64)
        self._Phi = np.eye(16, dtype=np.float64)
        self._G = np.zeros((16, 12), dtype=np.float64)
        self._Qd = np.zeros((12, 12), dtype=np.float64)
        self._H3 = np.zeros((3, 16), dtype=np.float64)
        self._H2 = np.zeros((2, 16), dtype=np.float64)
        self._H_flow = np.zeros((2, 16), dtype=np.float64)
        self._I3 = np.eye(3, dtype=np.float64)
        self._I16 = np.eye(16, dtype=np.float64)

        self._H3[0, 0] = 1.0
        self._H3[1, 1] = 1.0
        self._H3[2, 8] = 1.0

        self._H2[0, 0] = 1.0
        self._H2[1, 1] = 1.0

        self._H_flow[0, 3] = 1.0
        self._H_flow[1, 4] = 1.0

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def timestamp_s(self) -> Optional[float]:
        return self._timestamp_s

    @property
    def position_enu_m(self) -> NDArray[np.float64]:
        return self._position_enu_m.copy()

    @property
    def velocity_enu_mps(self) -> NDArray[np.float64]:
        return self._velocity_enu_mps.copy()

    @property
    def quaternion_wxyz(self) -> NDArray[np.float64]:
        return self._quaternion_wxyz.copy()

    @property
    def yaw_rad(self) -> float:
        return yaw_from_quaternion(self._quaternion_wxyz)

    @property
    def covariance_16x16(self) -> NDArray[np.float64]:
        return self._P.copy()

    @property
    def position_covariance_xy_m2(self) -> NDArray[np.float64]:
        return np.array([float(self._P[0, 0]), float(self._P[1, 1])], dtype=np.float64)

    @property
    def pzz_m2(self) -> float:
        return float(self._P[2, 2])

    @property
    def velocity_trace_m2ps2(self) -> float:
        return float(self._P[3, 3] + self._P[4, 4] + self._P[5, 5])

    @property
    def time_delay_s(self) -> float:
        return float(self._time_delay_s)

    @property
    def health(self) -> FilterHealth:
        return FilterHealth(
            consecutive_rejections=self._consecutive_rejections,
            consecutive_lightglue_rejections=self._consecutive_lightglue_rejections,
            consecutive_vision_loss=self._consecutive_vision_loss,
            position_trace_xy_m2=float(self._P[0, 0] + self._P[1, 1]),
            velocity_trace_m2ps2=self.velocity_trace_m2ps2,
            pzz_m2=self.pzz_m2,
            last_update_source=self._last_update_source,
        )

    def initialize(self, position_enu_m: NDArray[np.float64], yaw_rad: float, timestamp_s: float) -> None:
        self._position_enu_m[:] = np.asarray(position_enu_m, dtype=np.float64)
        self._velocity_enu_mps.fill(0.0)
        self._quaternion_wxyz[:] = quaternion_from_yaw(yaw_rad)
        self._accel_bias_mps2.fill(0.0)
        self._gyro_bias_rps.fill(0.0)
        self._time_delay_s = float(np.clip(self._config.timing.initial_time_delay_s, 0.0, self._config.timing.max_time_delay_s))
        self._timestamp_s = timestamp_s
        self._is_initialized = True

        self._consecutive_rejections = 0
        self._consecutive_lightglue_rejections = 0
        self._consecutive_vision_loss = 0
        self._last_update_source = "init"
        self._last_accepted_measurement_time_s = timestamp_s
        self._last_accepted_position_xy_m = self._position_enu_m[0:2].copy()

        self._state_history.clear()
        self._imu_history.clear()
        self._flow_history.clear()
        self._state_history.append(timestamp_s, self._snapshot())

    def predict_from_imu(self, imu_sample: ImuSample) -> None:
        self._predict_internal(imu_sample, record_imu=True, record_snapshot=True)

    def _predict_internal(self, imu_sample: ImuSample, record_imu: bool, record_snapshot: bool) -> None:
        if not self._is_initialized:
            return

        if record_imu:
            self._imu_history.append(imu_sample.timestamp_s, imu_sample)

        if self._timestamp_s is None:
            self._timestamp_s = imu_sample.timestamp_s
            if record_snapshot:
                self._state_history.append(self._timestamp_s, self._snapshot())
            return

        dt = float(imu_sample.timestamp_s - self._timestamp_s)
        if dt <= 0.0:
            return

        accel_unbiased = imu_sample.accel_mps2 - self._accel_bias_mps2
        gyro_unbiased = imu_sample.gyro_rps - self._gyro_bias_rps

        rotation_bw = rotation_matrix_from_quaternion(self._quaternion_wxyz)
        gravity = self._config.gravity_enu_mps2
        accel_world = rotation_bw @ accel_unbiased + gravity

        dt2 = dt * dt
        self._position_enu_m += self._velocity_enu_mps * dt + 0.5 * accel_world * dt2
        self._velocity_enu_mps += accel_world * dt

        delta_q = quaternion_from_small_angle(gyro_unbiased * dt)
        self._quaternion_wxyz = quaternion_normalize(quaternion_multiply(self._quaternion_wxyz, delta_q))

        self._build_continuous_jacobians(rotation_bw, accel_unbiased, gyro_unbiased)

        self._Phi[:, :] = self._I16 + self._F * dt
        self._Qd.fill(0.0)
        self._Qd[0:3, 0:3] = self._I3 * (self._config.noise.accel_noise_std_mps2**2) * dt
        self._Qd[3:6, 3:6] = self._I3 * (self._config.noise.gyro_noise_std_rps**2) * dt
        self._Qd[6:9, 6:9] = self._I3 * (self._config.noise.accel_bias_rw_std_mps2**2) * dt
        self._Qd[9:12, 9:12] = self._I3 * (self._config.noise.gyro_bias_rw_std_rps**2) * dt

        self._P[:, :] = self._Phi @ self._P @ self._Phi.T + self._G @ self._Qd @ self._G.T
        self._P[:, :] = force_symmetric(self._P)

        self._timestamp_s = imu_sample.timestamp_s
        if record_snapshot:
            self._state_history.append(self._timestamp_s, self._snapshot())

    def _build_continuous_jacobians(
        self,
        rotation_bw: NDArray[np.float64],
        accel_unbiased: NDArray[np.float64],
        gyro_unbiased: NDArray[np.float64],
    ) -> None:
        self._F.fill(0.0)
        self._F[0:3, 3:6] = self._I3
        self._F[3:6, 6:9] = -rotation_bw @ skew_symmetric(accel_unbiased)
        self._F[3:6, 9:12] = -rotation_bw
        self._F[6:9, 6:9] = -skew_symmetric(gyro_unbiased)
        self._F[6:9, 12:15] = -self._I3

        self._G.fill(0.0)
        self._G[3:6, 0:3] = rotation_bw
        self._G[6:9, 3:6] = self._I3
        self._G[9:12, 6:9] = self._I3
        self._G[12:15, 9:12] = self._I3

    def update_from_vision(self, measurement: VisionMeasurement) -> UpdateResult:
        if not self._is_initialized:
            return self._reject(measurement.source, "filter_not_initialized")

        if not self._passes_kinematic_gate(measurement):
            return self._reject(measurement.source, "kinematic_limit_exceeded")

        if measurement.vision_dof == 3:
            H = self._H3
            predicted = np.array([self._position_enu_m[0], self._position_enu_m[1], self.yaw_rad], dtype=np.float64)
            innovation_raw = np.array(
                [
                    measurement.position_xy_enu_m[0] - predicted[0],
                    measurement.position_xy_enu_m[1] - predicted[1],
                    normalize_angle(measurement.yaw_rad - predicted[2]),
                ],
                dtype=np.float64,
            )
            chi2_threshold = self._config.gate.chi2_threshold_3dof
        else:
            H = self._H2
            predicted = np.array([self._position_enu_m[0], self._position_enu_m[1]], dtype=np.float64)
            innovation_raw = np.array(
                [
                    measurement.position_xy_enu_m[0] - predicted[0],
                    measurement.position_xy_enu_m[1] - predicted[1],
                ],
                dtype=np.float64,
            )
            chi2_threshold = self._config.gate.chi2_threshold_2dof

        covariance = self._apply_measurement_covariance_floor(measurement.covariance)
        S = H @ self._P @ H.T + covariance
        S = force_symmetric(S)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        mahalanobis_distance = float(innovation_raw.T @ S_inv @ innovation_raw)
        if mahalanobis_distance > chi2_threshold:
            return self._reject(
                measurement.source,
                "mahalanobis_gate_rejected",
                mahalanobis_distance=mahalanobis_distance,
                innovation=innovation_raw,
            )

        innovation = innovation_raw.copy()
        innovation[0:2] = clip_vector_norm(innovation[0:2], self._config.gate.max_position_correction_m)
        if measurement.vision_dof == 3:
            innovation[2] = float(
                np.clip(
                    innovation[2],
                    -np.deg2rad(self._config.gate.max_yaw_correction_deg),
                    np.deg2rad(self._config.gate.max_yaw_correction_deg),
                )
            )

        K = self._P @ H.T @ S_inv
        delta_x = K @ innovation

        delta_x[0:2] = clip_vector_norm(delta_x[0:2], self._config.gate.max_position_correction_m)
        if measurement.vision_dof == 3:
            delta_x[8] = float(
                np.clip(
                    delta_x[8],
                    -np.deg2rad(self._config.gate.max_yaw_correction_deg),
                    np.deg2rad(self._config.gate.max_yaw_correction_deg),
                )
            )

        self._inject_error_state(delta_x)

        I_KH = self._I16 - K @ H
        self._P[:, :] = I_KH @ self._P @ I_KH.T + K @ covariance @ K.T
        self._P[:, :] = force_symmetric(self._P)

        self._last_accepted_measurement_time_s = measurement.timestamp_s
        self._last_accepted_position_xy_m = measurement.position_xy_enu_m.copy()
        self._consecutive_rejections = 0
        self._consecutive_vision_loss = 0
        if measurement.source.lower() == "lightglue":
            self._consecutive_lightglue_rejections = 0
        self._last_update_source = measurement.source

        if self._timestamp_s is not None:
            self._state_history.append(self._timestamp_s, self._snapshot())

        return UpdateResult(
            accepted=True,
            reason="accepted",
            source=measurement.source,
            mahalanobis_distance=mahalanobis_distance,
            innovation=innovation,
        )

    def _apply_measurement_covariance_floor(self, covariance: NDArray[np.float64]) -> NDArray[np.float64]:
        covariance = np.asarray(covariance, dtype=np.float64).copy()
        min_var = float(self._config.gate.min_measurement_variance_m2)
        diag = np.diag(covariance)
        diag = np.maximum(diag, min_var)
        covariance[np.diag_indices_from(covariance)] = diag
        return covariance

    def update_from_flow(self, measurement: FlowMeasurement) -> UpdateResult:
        return self._update_from_flow_internal(measurement, record_history=True)

    def _update_from_flow_internal(self, measurement: FlowMeasurement, record_history: bool) -> UpdateResult:
        if not self._is_initialized:
            return self._reject(measurement.source, "filter_not_initialized")

        if self.pzz_m2 > self._config.flow.pzz_reject_threshold_m2:
            return self._reject(measurement.source, "flow_rejected_pzz_high")

        predicted = self._velocity_enu_mps[0:2]
        innovation_xy = measurement.velocity_xy_mps - predicted

        S = self._H_flow @ self._P @ self._H_flow.T + measurement.covariance_2x2
        S = force_symmetric(S)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        mahalanobis_distance = float(innovation_xy.T @ S_inv @ innovation_xy)
        if mahalanobis_distance > self._config.flow.chi2_threshold_2dof:
            return self._reject(
                measurement.source,
                "flow_mahalanobis_rejected",
                mahalanobis_distance=mahalanobis_distance,
                innovation=innovation_xy,
            )

        K = self._P @ self._H_flow.T @ S_inv
        delta_x = K @ innovation_xy
        self._inject_error_state(delta_x)

        I_KH = self._I16 - K @ self._H_flow
        self._P[:, :] = I_KH @ self._P @ I_KH.T + K @ measurement.covariance_2x2 @ K.T
        self._P[:, :] = force_symmetric(self._P)
        self._last_update_source = measurement.source

        if record_history:
            flow_ts = float(measurement.timestamp_s)
            latest_flow = self._flow_history.latest()
            if latest_flow is not None:
                flow_ts = max(flow_ts, float(latest_flow[0]))
            self._flow_history.append(flow_ts, measurement)

        if self._timestamp_s is not None:
            self._state_history.append(self._timestamp_s, self._snapshot())

        return UpdateResult(
            accepted=True,
            reason="accepted",
            source=measurement.source,
            mahalanobis_distance=mahalanobis_distance,
            innovation=innovation_xy,
        )

    def apply_historical_vision_update(self, measurement: VisionMeasurement) -> UpdateResult:
        if not self._is_initialized or self._timestamp_s is None:
            return self._reject(measurement.source, "filter_not_initialized")

        corrected_timestamp_s = float(measurement.timestamp_s)
        if corrected_timestamp_s > self._timestamp_s + 1e-3:
            return self._reject(measurement.source, "measurement_from_future")

        state_at_time = self._state_history.at_nearest(corrected_timestamp_s)
        if state_at_time is not None and state_at_time[0] > corrected_timestamp_s:
            state_before = self._state_history.at_or_before(corrected_timestamp_s)
            if state_before is not None:
                state_at_time = state_before
        if state_at_time is None:
            return self._reject(measurement.source, "state_history_too_short")

        snapshot_time_s, snapshot = state_at_time
        imu_future = self._imu_history.after(snapshot_time_s)
        flow_future = self._flow_history.after(snapshot_time_s)

        self._restore_snapshot(snapshot)
        self._state_history.clear_after(snapshot_time_s)
        self._imu_history.clear_after(snapshot_time_s)
        self._flow_history.clear_after(snapshot_time_s)

        replay_events: list[tuple[float, int, ImuSample | FlowMeasurement]] = []
        replay_events.extend((float(ts), 0, sample) for ts, sample in imu_future)
        replay_events.extend((float(ts), 1, sample) for ts, sample in flow_future)
        replay_events.sort(key=lambda event: (event[0], event[1]))

        pre_update_events = [event for event in replay_events if event[0] <= corrected_timestamp_s]
        post_update_events = [event for event in replay_events if event[0] > corrected_timestamp_s]

        for _, event_kind, event_payload in pre_update_events:
            if event_kind == 0 and isinstance(event_payload, ImuSample):
                self._predict_internal(event_payload, record_imu=True, record_snapshot=True)
            elif event_kind == 1 and isinstance(event_payload, FlowMeasurement):
                self._update_from_flow_internal(event_payload, record_history=True)

        corrected_measurement = VisionMeasurement(
            timestamp_s=corrected_timestamp_s,
            position_xy_enu_m=measurement.position_xy_enu_m,
            yaw_rad=measurement.yaw_rad,
            covariance=measurement.covariance,
            source=measurement.source,
            processing_latency_s=measurement.processing_latency_s,
            scale=measurement.scale,
            inlier_ratio=measurement.inlier_ratio,
            vision_dof=measurement.vision_dof,
            yaw_valid=measurement.yaw_valid,
            model=measurement.model,
            match_confidence=measurement.match_confidence,
            match_count=measurement.match_count,
            inlier_count=measurement.inlier_count,
            reprojection_rmse_px=measurement.reprojection_rmse_px,
            failsafe_mode=measurement.failsafe_mode,
            confidence_weight=measurement.confidence_weight,
        )

        update_result = self.update_from_vision(corrected_measurement)
        if update_result.accepted:
            self._time_delay_s = float(np.clip(measurement.processing_latency_s, 0.0, self._config.timing.max_time_delay_s))

        for _, event_kind, event_payload in post_update_events:
            if event_kind == 0 and isinstance(event_payload, ImuSample):
                self._predict_internal(event_payload, record_imu=True, record_snapshot=True)
            elif event_kind == 1 and isinstance(event_payload, FlowMeasurement):
                self._update_from_flow_internal(event_payload, record_history=True)

        return update_result

    def _passes_kinematic_gate(self, measurement: VisionMeasurement) -> bool:
        if self._last_accepted_measurement_time_s is None or self._last_accepted_position_xy_m is None:
            return True

        if measurement.timestamp_s <= self._last_accepted_measurement_time_s:
            return True

        dt = measurement.timestamp_s - self._last_accepted_measurement_time_s
        max_distance = self._config.gate.vmax_mps * dt + self._config.gate.kinematic_tolerance_m
        distance = float(np.linalg.norm(measurement.position_xy_enu_m - self._last_accepted_position_xy_m))
        if distance > max_distance:
            return False

        if distance > self._config.gate.inconsistency_threshold_m:
            return False

        return True

    def _inject_error_state(self, delta_x: NDArray[np.float64]) -> None:
        self._position_enu_m += delta_x[0:3]
        self._velocity_enu_mps += delta_x[3:6]

        delta_q = quaternion_from_small_angle(delta_x[6:9])
        self._quaternion_wxyz = quaternion_normalize(quaternion_multiply(self._quaternion_wxyz, delta_q))

        self._accel_bias_mps2 += delta_x[9:12]
        self._gyro_bias_rps += delta_x[12:15]
        if delta_x.shape[0] > 15:
            self._time_delay_s = float(
                np.clip(
                    self._time_delay_s + delta_x[15],
                    0.0,
                    self._config.timing.max_time_delay_s,
                )
            )

    def inflate_altitude_uncertainty(self, target_variance_m2: float) -> None:
        self._P[2, 2] = max(float(target_variance_m2), self._P[2, 2])

    def set_altitude_estimate(self, z_enu_m: float) -> None:
        self._position_enu_m[2] = float(z_enu_m)

    def register_vision_loss(self) -> None:
        self._consecutive_vision_loss += 1

    def reject_vision_measurement(self, source: str, reason: str) -> UpdateResult:
        return self._reject(source, reason)

    def _snapshot(self) -> FilterSnapshot:
        if self._timestamp_s is None:
            raise RuntimeError("snapshot requested before timestamp initialization")
        return FilterSnapshot(
            timestamp_s=self._timestamp_s,
            position_enu_m=self._position_enu_m.copy(),
            velocity_enu_mps=self._velocity_enu_mps.copy(),
            quaternion_wxyz=self._quaternion_wxyz.copy(),
            accel_bias_mps2=self._accel_bias_mps2.copy(),
            gyro_bias_rps=self._gyro_bias_rps.copy(),
            time_delay_s=self._time_delay_s,
            covariance_16x16=self._P.copy(),
        )

    def _restore_snapshot(self, snapshot: FilterSnapshot) -> None:
        self._timestamp_s = snapshot.timestamp_s
        self._position_enu_m[:] = snapshot.position_enu_m
        self._velocity_enu_mps[:] = snapshot.velocity_enu_mps
        self._quaternion_wxyz[:] = snapshot.quaternion_wxyz
        self._accel_bias_mps2[:] = snapshot.accel_bias_mps2
        self._gyro_bias_rps[:] = snapshot.gyro_bias_rps
        self._time_delay_s = float(snapshot.time_delay_s)
        self._P[:, :] = snapshot.covariance_16x16

    def _reject(
        self,
        source: str,
        reason: str,
        mahalanobis_distance: Optional[float] = None,
        innovation: Optional[NDArray[np.float64]] = None,
    ) -> UpdateResult:
        self._consecutive_rejections += 1
        if source.lower() == "lightglue":
            self._consecutive_lightglue_rejections += 1
        self._last_update_source = f"rejected:{source}"
        innovation_vec = innovation.copy() if innovation is not None else None
        return UpdateResult(
            accepted=False,
            reason=reason,
            source=source,
            mahalanobis_distance=mahalanobis_distance,
            innovation=innovation_vec,
        )


ErrorStateKalmanFilter = CascadedESKF
