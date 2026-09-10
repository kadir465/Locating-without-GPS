from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RoiWindow:
    center_xy_enu_m: NDArray[np.float64]
    size_m: float
    width_m: float
    height_m: float
    predicted_latency_s: float
    padding_x_m: float
    padding_y_m: float


class ShiftedRoiSelector:
    def __init__(
        self,
        base_window_size_m: float = 20.0,
        min_speed_mps: float = 1.0,
        max_window_size_m: float = 80.0,
        padding_sigma: float = 3.0,
    ) -> None:
        self._base_window_size_m = base_window_size_m
        self._min_speed_mps = min_speed_mps
        self._max_window_size_m = max_window_size_m
        self._padding_sigma = padding_sigma

    def select(
        self,
        position_xy_enu_m: NDArray[np.float64],
        velocity_xy_mps: NDArray[np.float64],
        acceleration_xy_mps2: NDArray[np.float64] | None = None,
        covariance_xy_m2: NDArray[np.float64] | None = None,
        latency_s: float = 0.0,
    ) -> RoiWindow:
        position_xy_enu_m = np.asarray(position_xy_enu_m, dtype=np.float64)
        velocity_xy_mps = np.asarray(velocity_xy_mps, dtype=np.float64)
        acceleration_xy_mps2 = (
            np.zeros(2, dtype=np.float64)
            if acceleration_xy_mps2 is None
            else np.asarray(acceleration_xy_mps2, dtype=np.float64)
        )

        if covariance_xy_m2 is None:
            covariance_xy_m2 = np.array([self._base_window_size_m**2, self._base_window_size_m**2], dtype=np.float64)
        else:
            covariance_xy_m2 = np.asarray(covariance_xy_m2, dtype=np.float64)
            if covariance_xy_m2.shape == (2, 2):
                covariance_xy_m2 = np.array([covariance_xy_m2[0, 0], covariance_xy_m2[1, 1]], dtype=np.float64)
            elif covariance_xy_m2.shape != (2,):
                raise ValueError("covariance_xy_m2 must have shape (2,) or (2,2)")

        latency_s = float(max(0.0, latency_s))

        speed = float(np.linalg.norm(velocity_xy_mps))
        center = position_xy_enu_m + velocity_xy_mps * latency_s + 0.5 * acceleration_xy_mps2 * (latency_s**2)

        base_major_m = float(self._base_window_size_m)
        base_minor_m = float(self._base_window_size_m)

        if speed >= self._min_speed_mps:
            direction = velocity_xy_mps / speed
            forward_shift_m = speed * latency_s + 0.5 * max(0.0, float(np.dot(acceleration_xy_mps2, direction))) * (latency_s**2)
            forward_shift_m = float(np.clip(forward_shift_m, 0.0, 0.8 * self._base_window_size_m))

            major_m = base_major_m + forward_shift_m
            minor_m = base_minor_m

            lateral = np.array([-direction[1], direction[0]], dtype=np.float64)
            footprint_width_m = abs(direction[0]) * major_m + abs(lateral[0]) * minor_m
            footprint_height_m = abs(direction[1]) * major_m + abs(lateral[1]) * minor_m
        else:
            footprint_width_m = base_major_m
            footprint_height_m = base_minor_m

        p_xx = float(max(covariance_xy_m2[0], 1e-6))
        p_yy = float(max(covariance_xy_m2[1], 1e-6))
        padding_x_m = float(self._padding_sigma * np.sqrt(p_xx))
        padding_y_m = float(self._padding_sigma * np.sqrt(p_yy))

        width_m = float(np.clip(footprint_width_m + padding_x_m, self._base_window_size_m, self._max_window_size_m))
        height_m = float(np.clip(footprint_height_m + padding_y_m, self._base_window_size_m, self._max_window_size_m))
        size_m = float(max(width_m, height_m))

        return RoiWindow(
            center_xy_enu_m=center,
            size_m=size_m,
            width_m=width_m,
            height_m=height_m,
            predicted_latency_s=latency_s,
            padding_x_m=padding_x_m,
            padding_y_m=padding_y_m,
        )
