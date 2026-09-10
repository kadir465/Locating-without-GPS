from __future__ import annotations

import numpy as np

from advanced_localization.vision.roi_selector import ShiftedRoiSelector


def test_roi_selector_applies_kinematic_center_prediction() -> None:
    selector = ShiftedRoiSelector(
        base_window_size_m=20.0,
        min_speed_mps=0.5,
        max_window_size_m=80.0,
        padding_sigma=3.0,
    )

    roi = selector.select(
        position_xy_enu_m=np.array([0.0, 0.0], dtype=np.float64),
        velocity_xy_mps=np.array([10.0, 0.0], dtype=np.float64),
        acceleration_xy_mps2=np.array([2.0, 0.0], dtype=np.float64),
        covariance_xy_m2=np.array([4.0, 9.0], dtype=np.float64),
        latency_s=0.5,
    )

    # x = x + v*t + 0.5*a*t^2
    assert abs(float(roi.center_xy_enu_m[0]) - 5.25) < 1e-6
    assert abs(float(roi.center_xy_enu_m[1])) < 1e-6

    # Direction is +X, so major axis projects to width.
    # forward_extra = 5.25, footprint_major = 25.25, footprint_minor = 20
    # padding_x = 3*sqrt(4)=6, padding_y = 3*sqrt(9)=9
    assert abs(float(roi.width_m) - 31.25) < 1e-6
    assert abs(float(roi.height_m) - 29.0) < 1e-6
    assert abs(float(roi.size_m) - 31.25) < 1e-6


def test_roi_selector_stationary_uses_base_plus_covariance_padding() -> None:
    selector = ShiftedRoiSelector(
        base_window_size_m=20.0,
        min_speed_mps=1.0,
        max_window_size_m=80.0,
        padding_sigma=3.0,
    )

    roi = selector.select(
        position_xy_enu_m=np.array([12.0, -5.0], dtype=np.float64),
        velocity_xy_mps=np.array([0.2, 0.0], dtype=np.float64),
        acceleration_xy_mps2=np.array([0.0, 0.0], dtype=np.float64),
        covariance_xy_m2=np.array([1.0, 1.0], dtype=np.float64),
        latency_s=0.4,
    )

    assert np.allclose(roi.center_xy_enu_m, np.array([12.08, -5.0], dtype=np.float64), atol=1e-6)
    assert abs(float(roi.width_m) - 23.0) < 1e-6
    assert abs(float(roi.height_m) - 23.0) < 1e-6
    assert abs(float(roi.predicted_latency_s) - 0.4) < 1e-6
