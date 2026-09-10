from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from ..config import SystemConfig
from ..types import FlowMeasurement


class OpticalFlowLayer:

    def get_flow_norm_and_feature_count(
        self,
        frame_gray: NDArray[np.uint8],
        timestamp_s: float,
        altitude_m: float,
        yaw_rad: float,
    ) -> Optional[tuple[float, int]]:
        """
        Returns (flow_norm, feature_count) for ZUPT triggering.
        """
        result = self.run(frame_gray, timestamp_s, altitude_m, yaw_rad)
        if result is not None:
            flow_norm = np.linalg.norm(result.velocity_xy_mps)
            return flow_norm, result.feature_count
        return None

    def __init__(self, config: SystemConfig, camera_matrix: NDArray[np.float64]) -> None:
        self._config = config
        self._camera_matrix = np.asarray(camera_matrix, dtype=np.float64)

        self._prev_frame: Optional[NDArray[np.uint8]] = None
        self._prev_timestamp_s: Optional[float] = None
        self._prev_points: Optional[NDArray[np.float32]] = None

        self._lk_win_size = (21, 21)
        self._lk_max_level = 3
        self._lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    def run(
        self,
        frame_gray: NDArray[np.uint8],
        timestamp_s: float,
        altitude_m: float,
        yaw_rad: float,
    ) -> Optional[FlowMeasurement]:
        if self._prev_frame is None or self._prev_timestamp_s is None:
            self._bootstrap(frame_gray, timestamp_s)
            return None

        dt = float(timestamp_s - self._prev_timestamp_s)
        if dt <= 1e-6:
            return None

        if self._prev_points is None or len(self._prev_points) < self._config.flow.min_features:
            self._prev_points = self._detect_features(self._prev_frame)
            if self._prev_points is None or len(self._prev_points) < 4:
                self._bootstrap(frame_gray, timestamp_s)
                return None

        next_points, status_fw, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_frame,
            frame_gray,
            self._prev_points,
            None,
            winSize=self._lk_win_size,
            maxLevel=self._lk_max_level,
            criteria=self._lk_criteria,
        )
        if next_points is None or status_fw is None:
            self._bootstrap(frame_gray, timestamp_s)
            return None

        back_points, status_bw, _ = cv2.calcOpticalFlowPyrLK(
            frame_gray,
            self._prev_frame,
            next_points,
            None,
            winSize=self._lk_win_size,
            maxLevel=self._lk_max_level,
            criteria=self._lk_criteria,
        )
        if back_points is None or status_bw is None:
            self._bootstrap(frame_gray, timestamp_s)
            return None

        fw_ok = status_fw.reshape(-1) == 1
        bw_ok = status_bw.reshape(-1) == 1
        bidirectional_err = np.linalg.norm(back_points - self._prev_points, axis=2).reshape(-1)
        valid = fw_ok & bw_ok & (bidirectional_err < self._config.flow.max_bidirectional_error_px)

        prev_valid = self._prev_points[valid][:, 0, :]
        next_valid = next_points[valid][:, 0, :]

        if prev_valid.shape[0] < self._config.flow.min_features:
            self._bootstrap(frame_gray, timestamp_s)
            return None

        flow_px = np.median(next_valid - prev_valid, axis=0)
        velocity_xy_enu_mps = self._pixel_flow_to_velocity_enu(
            flow_px=flow_px,
            dt=dt,
            altitude_m=altitude_m,
            yaw_rad=yaw_rad,
        )

        feature_scale = np.clip(self._config.flow.min_features / max(float(prev_valid.shape[0]), 1.0), 0.5, 4.0)
        sigma = self._config.flow.noise_std_mps * feature_scale
        covariance = np.zeros((2, 2), dtype=np.float64)
        covariance[0, 0] = sigma * sigma
        covariance[1, 1] = sigma * sigma

        self._prev_frame = frame_gray.copy()
        self._prev_timestamp_s = timestamp_s
        self._prev_points = next_valid.reshape(-1, 1, 2).astype(np.float32)

        return FlowMeasurement(
            timestamp_s=timestamp_s,
            velocity_xy_mps=velocity_xy_enu_mps,
            covariance_2x2=covariance,
            feature_count=int(prev_valid.shape[0]),
        )

    def _bootstrap(self, frame_gray: NDArray[np.uint8], timestamp_s: float) -> None:
        self._prev_frame = frame_gray.copy()
        self._prev_timestamp_s = timestamp_s
        self._prev_points = self._detect_features(frame_gray)

    def _detect_features(self, frame_gray: NDArray[np.uint8]) -> Optional[NDArray[np.float32]]:
        return cv2.goodFeaturesToTrack(
            frame_gray,
            maxCorners=800,
            qualityLevel=0.01,
            minDistance=6.0,
            blockSize=7,
            useHarrisDetector=False,
        )

    def _pixel_flow_to_velocity_enu(
        self,
        flow_px: NDArray[np.float32],
        dt: float,
        altitude_m: float,
        yaw_rad: float,
    ) -> NDArray[np.float64]:
        fx = float(self._camera_matrix[0, 0])
        fy = float(self._camera_matrix[1, 1])

        alt = max(1.0, abs(altitude_m))
        # Body-frame convention (shared with video testing):
        # forward velocity from v-flow, right velocity from -u-flow.
        vx_body = float(flow_px[1]) * alt / max(fy, 1e-6) / dt
        vy_body = -float(flow_px[0]) * alt / max(fx, 1e-6) / dt

        c = float(np.cos(yaw_rad))
        s = float(np.sin(yaw_rad))
        return np.array(
            [
                c * vx_body - s * vy_body,
                s * vx_body + c * vy_body,
            ],
            dtype=np.float64,
        )
