from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from advanced_localization.config import SystemConfig
from advanced_localization.eskf import CascadedESKF
from advanced_localization.types import ImuSample, VisionMeasurement


@dataclass(frozen=True)
class SimulationResult:
    time_s: np.ndarray
    trace_xy: np.ndarray
    accepted_updates: int
    rejected_updates: int


def run_synthetic_flight(duration_s: float = 60.0, imu_hz: float = 100.0, vision_hz: float = 1.0) -> SimulationResult:
    config = SystemConfig()
    filt = CascadedESKF(config)
    filt.initialize(position_enu_m=np.zeros(3, dtype=np.float64), yaw_rad=0.0, timestamp_s=0.0)

    dt = 1.0 / imu_hz
    t = 0.0

    next_vision_t = 1.0 / vision_hz
    accepted = 0
    rejected = 0

    times = []
    trace_xy = []

    while t <= duration_s:
        yaw_rate = np.deg2rad(8.0) * np.sin(0.03 * t)
        imu = ImuSample(
            timestamp_s=t,
            accel_mps2=np.array([0.0, 0.0, 9.80665], dtype=np.float64),
            gyro_rps=np.array([0.0, 0.0, yaw_rate], dtype=np.float64),
        )
        filt.predict_from_imu(imu)

        if t >= next_vision_t - 1e-9:
            est_pos = filt.position_enu_m[0:2]
            noise = np.random.default_rng(42 + int(t * 10)).normal(0.0, 0.7, size=2)
            measurement = VisionMeasurement(
                timestamp_s=t,
                position_xy_enu_m=est_pos + noise,
                yaw_rad=filt.yaw_rad + np.deg2rad(0.5),
                covariance=np.diag(np.array([0.7**2, 0.7**2, np.deg2rad(0.8) ** 2], dtype=np.float64)),
                source="lightglue",
                processing_latency_s=0.2,
                scale=1.0,
                inlier_ratio=0.85,
                vision_dof=3,
                yaw_valid=True,
                model="SE2_PLANAR",
            )
            result = filt.apply_historical_vision_update(measurement)
            if result.accepted:
                accepted += 1
            else:
                rejected += 1
            next_vision_t += 1.0 / vision_hz

        cov = filt.covariance_16x16
        times.append(t)
        trace_xy.append(float(cov[0, 0] + cov[1, 1]))
        t += dt

    return SimulationResult(
        time_s=np.asarray(times, dtype=np.float64),
        trace_xy=np.asarray(trace_xy, dtype=np.float64),
        accepted_updates=accepted,
        rejected_updates=rejected,
    )


def detect_sawtooth(trace_xy: np.ndarray, min_drop_events: int = 5) -> bool:
    if trace_xy.size < 20:
        return False

    diff = np.diff(trace_xy)
    drop_events = int(np.sum(diff < -1e-4))
    rise_events = int(np.sum(diff > 1e-5))
    return drop_events >= min_drop_events and rise_events > drop_events


def main() -> None:
    result = run_synthetic_flight()
    sawtooth = detect_sawtooth(result.trace_xy)

    print(f"accepted_updates={result.accepted_updates}")
    print(f"rejected_updates={result.rejected_updates}")
    print(f"trace_xy_start={result.trace_xy[0]:.6f}")
    print(f"trace_xy_end={result.trace_xy[-1]:.6f}")
    print(f"sawtooth_detected={sawtooth}")


if __name__ == "__main__":
    main()
