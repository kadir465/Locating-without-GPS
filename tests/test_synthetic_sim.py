from __future__ import annotations

from simulation.synthetic_flight import detect_sawtooth, run_synthetic_flight


def test_synthetic_flight_shows_covariance_sawtooth() -> None:
    result = run_synthetic_flight(duration_s=30.0, imu_hz=100.0, vision_hz=1.0)
    assert result.accepted_updates >= 5
    assert detect_sawtooth(result.trace_xy)
