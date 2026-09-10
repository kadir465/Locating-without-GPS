from __future__ import annotations

import csv
import re
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Deque, Optional, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import SystemConfig
from .map_store import StaticGeoMap
from .types import VisionMeasurement
from .vision.lightglue_layer import LightGlueLayer

EARTH_RADIUS_M = 6378137.0
FIXED_TEST_ALTITUDE_M = 100.0
EPSILON = 1e-9
DEFAULT_MAX_FLOW_STD_PX = 260.0
DEFAULT_MAX_FLOW_SHIFT_PX = 140.0
DEFAULT_MAX_FLOW_STEP_M = 4.0
DEFAULT_HEADING_SMOOTH_WINDOW = 15
DEFAULT_HEADING_MIN_DISPLACEMENT_M = 0.35

# 

class LiveMapTrailWindow:
    def __init__(
        self,
        map_path: Path,
        gsd_m_per_px: float,
        window_name: str = "Map Point&Trail",
        max_display_size_px: int = 1000,
        video_output_path: Path | None = None,
        video_fps: float = 30.0,
    ) -> None:
        if gsd_m_per_px <= 0.0:
            raise ValueError("gsd_m_per_px must be positive")
        if max_display_size_px <= 0:
            raise ValueError("max_display_size_px must be positive")
        if video_output_path is not None and video_fps <= 0.0:
            raise ValueError("video_fps must be positive when video_output_path is provided")

        self._window_name = window_name
        self._max_display_size_px = max_display_size_px
        
        # Load map without resizing to keep high resolution for moving window
        self._base_bgr = self._load_map_bgr(map_path)
        self._height, self._width = self._base_bgr.shape[:2]

        self._display_gsd_m_per_px = gsd_m_per_px
        self._origin_px = np.array([self._width * 0.5, self._height * 0.5], dtype=np.float64)

        self._trail_canvas = self._base_bgr.copy()
        self._last_est_pt: Optional[tuple[int, int]] = None
        self._last_gt_pt: Optional[tuple[int, int]] = None

        self._est_trail: Deque[tuple[int, int]] = deque(maxlen=6000)
        self._gt_trail: Deque[tuple[int, int]] = deque(maxlen=6000)
        self._display_enabled = True
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._view_size_px = int(self._max_display_size_px)
        self._view_buffer = np.empty((self._view_size_px, self._view_size_px, 3), dtype=np.uint8)

        if video_output_path is not None:
            out_path = Path(video_output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer_size = (self._max_display_size_px, self._max_display_size_px)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, float(video_fps), writer_size)
            if not writer.isOpened():
                raise ValueError(f"Unable to create map window recording: {out_path}")
            self._video_writer = writer

    def render(
        self,
        est_x_m: float,
        est_y_m: float,
        true_x_m: float,
        true_y_m: float,
        error_m: float,
        frame_idx: int,
        roi_center_xy_enu_m: NDArray[np.float64] | None = None,
        roi_width_m: float | None = None,
        roi_height_m: float | None = None,
    ) -> bool:
        est_pt = self._enu_to_px(est_x_m, est_y_m)
        true_pt = self._enu_to_px(true_x_m, true_y_m)
        self._append_trails(est_pt=est_pt, true_pt=true_pt)

        # Crop a window centered on est_pt
        view_size = self._view_size_px
        half_size = view_size // 2
        
        cx, cy = est_pt
        
        x1 = cx - half_size
        y1 = cy - half_size
        x2 = x1 + view_size
        y2 = y1 + view_size
        
        # Calculate padding if crop is out of bounds
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - self._width)
        pad_bottom = max(0, y2 - self._height)
        
        cx1 = max(0, x1)
        cy1 = max(0, y1)
        cx2 = min(self._width, x2)
        cy2 = min(self._height, y2)
        
        cropped = self._trail_canvas[cy1:cy2, cx1:cx2]

        # Reuse a preallocated display buffer to avoid per-frame array allocation/copy overhead.
        view = self._view_buffer
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            view[:, :, :] = (40, 40, 40)
            crop_h = cy2 - cy1
            crop_w = cx2 - cx1
            view[pad_top : pad_top + crop_h, pad_left : pad_left + crop_w] = cropped
        else:
            view[:, :, :] = cropped

        est_local = (est_pt[0] - cx1 + pad_left, est_pt[1] - cy1 + pad_top)
        true_local = (true_pt[0] - cx1 + pad_left, true_pt[1] - cy1 + pad_top)
        self._draw_point_local(view, true_local, (0, 255, 0), 7)
        self._draw_point_local(view, est_local, (0, 0, 255), 9)
        if (
            roi_center_xy_enu_m is not None
            and roi_width_m is not None
            and roi_height_m is not None
            and roi_width_m > 0.0
            and roi_height_m > 0.0
        ):
            roi_center_pt = self._enu_to_px(float(roi_center_xy_enu_m[0]), float(roi_center_xy_enu_m[1]))
            half_roi_w_px = max(1, int(round(0.5 * float(roi_width_m) / self._display_gsd_m_per_px)))
            half_roi_h_px = max(1, int(round(0.5 * float(roi_height_m) / self._display_gsd_m_per_px)))

            roi_x1 = roi_center_pt[0] - half_roi_w_px
            roi_y1 = roi_center_pt[1] - half_roi_h_px
            roi_x2 = roi_center_pt[0] + half_roi_w_px
            roi_y2 = roi_center_pt[1] + half_roi_h_px

            roi_local_tl = (roi_x1 - cx1 + pad_left, roi_y1 - cy1 + pad_top)
            roi_local_br = (roi_x2 - cx1 + pad_left, roi_y2 - cy1 + pad_top)
            cv2.rectangle(view, roi_local_tl, roi_local_br, (0, 255, 255), 2, lineType=cv2.LINE_AA)

        cv2.putText(view, f"frame={frame_idx}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
        cv2.putText(view, f"error={error_m:.2f}m", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)

        if self._video_writer is not None:
            self._video_writer.write(view)

        if self._display_enabled:
            try:
                cv2.imshow(self._window_name, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    return False
            except cv2.error:
                self._display_enabled = False

        return True

    def close(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

        try:
            cv2.destroyWindow(self._window_name)
        except cv2.error:
            pass

    def _enu_to_px(self, x_m: float, y_m: float) -> tuple[int, int]:
        px = self._origin_px[0] + x_m / self._display_gsd_m_per_px
        py = self._origin_px[1] - y_m / self._display_gsd_m_per_px
        return int(round(px)), int(round(py))

    def _draw_trail(
        self,
        canvas: NDArray[np.uint8],
        trail: Deque[tuple[int, int]],
        color_bgr: tuple[int, int, int],
        thickness: int,
    ) -> None:
        if len(trail) < 2:
            return
        points = np.array(list(trail), dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], isClosed=False, color=color_bgr, thickness=thickness, lineType=cv2.LINE_AA)

    def _append_trails(self, est_pt: tuple[int, int], true_pt: tuple[int, int]) -> None:
        self._est_trail.append(est_pt)
        self._gt_trail.append(true_pt)

        if self._last_gt_pt is None:
            self._draw_point(self._trail_canvas, true_pt, (0, 220, 0), 3)
        else:
            cv2.line(self._trail_canvas, self._last_gt_pt, true_pt, (0, 220, 0), 2, lineType=cv2.LINE_AA)

        if self._last_est_pt is None:
            self._draw_point(self._trail_canvas, est_pt, (0, 80, 255), 4)
        else:
            cv2.line(self._trail_canvas, self._last_est_pt, est_pt, (0, 80, 255), 3, lineType=cv2.LINE_AA)

        self._last_gt_pt = true_pt
        self._last_est_pt = est_pt

    def _draw_point(
        self,
        canvas: NDArray[np.uint8],
        point: tuple[int, int],
        color_bgr: tuple[int, int, int],
        radius: int,
    ) -> None:
        if point[0] < 0 or point[0] >= self._width or point[1] < 0 or point[1] >= self._height:
            return
        cv2.circle(canvas, point, radius, color_bgr, -1, lineType=cv2.LINE_AA)

    @staticmethod
    def _draw_point_local(
        canvas: NDArray[np.uint8],
        point: tuple[int, int],
        color_bgr: tuple[int, int, int],
        radius: int,
    ) -> None:
        height, width = canvas.shape[:2]
        if point[0] < 0 or point[0] >= width or point[1] < 0 or point[1] >= height:
            return
        cv2.circle(canvas, point, radius, color_bgr, -1, lineType=cv2.LINE_AA)

    @staticmethod
    def _load_map_bgr(map_path: Path) -> NDArray[np.uint8]:
        suffix = map_path.suffix.lower()

        if suffix in {".tif", ".tiff"}:
            try:
                import rasterio
            except Exception as exc:
                raise ImportError("rasterio is required to read GeoTIFF maps") from exc

            with rasterio.open(map_path) as dataset:
                if dataset.count >= 3:
                    rgb = dataset.read([1, 2, 3]).transpose(1, 2, 0)
                    rgb_u8 = LiveMapTrailWindow._to_uint8(rgb)
                    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)

                band = dataset.read(1)
                gray_u8 = LiveMapTrailWindow._to_uint8(band)
                return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

        image = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to load map image: {map_path}")
        return image

    @staticmethod
    def _to_uint8(array: NDArray[np.generic]) -> NDArray[np.uint8]:
        if array.dtype == np.uint8:
            return array

        arr = np.asarray(array, dtype=np.float64)
        min_val = float(np.nanmin(arr))
        max_val = float(np.nanmax(arr))
        if max_val - min_val < 1e-12:
            return np.zeros(arr.shape, dtype=np.uint8)

        normalized = (arr - min_val) / (max_val - min_val)
        return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)


@dataclass(frozen=True)
class GroundTruthSample:
    timestamp_s: float
    latitude_deg: float
    longitude_deg: float
    heading_deg: Optional[float] = None
    rel_alt_m: Optional[float] = None


@dataclass(frozen=True)
class VideoTestResult:
    frame_count: int
    correction_count: int
    rmse_m: float
    mae_m: float
    cep95_m: float
    max_error_m: float
    timestamps_s: NDArray[np.float64]
    error_m: NDArray[np.float64]


@dataclass(frozen=True)
class FlowSignConfig:
    swap: bool
    sx: int
    sy: int


# Baseline camera mounting convention used by video tests:
# image u(right), v(down) -> body x(forward), y(right)
DEFAULT_FLOW_SIGN_CONFIG = FlowSignConfig(swap=True, sx=1, sy=1)


@dataclass(frozen=True)
class FlowStepDiagnostics:
    valid: bool
    du_px: float = 0.0
    dv_px: float = 0.0
    flow_std_px: float = 0.0
    flow_body_x_m: float = 0.0
    flow_body_y_m: float = 0.0
    flow_enu_x_m: float = 0.0
    flow_enu_y_m: float = 0.0
    heading_used_rad: float = 0.0
    zupt_applied: bool = False
    learned_scale: float = 1.0
    reject_reason: str = ""


@dataclass(frozen=True)
class LightGlueSearchResult:
    capture_timestamp_s: float
    capture_estimate_xy_enu_m: NDArray[np.float64]
    capture_flow_odom_xy_enu_m: NDArray[np.float64]
    heading_seed_rad: float
    measurement: Optional[VisionMeasurement]
    evaluated_hypotheses: int = 0
    wall_time_s: float = 0.0
    debug_summary: str = ""


def _format_lightglue_debug_summary(debug_info: dict[str, float | int | str | bool] | None) -> str:
    if not debug_info:
        return ""

    reason = str(debug_info.get("reason", "unknown"))
    match_count = int(debug_info.get("match_count", 0))
    inlier_count = int(debug_info.get("inlier_count", 0))
    inlier_ratio = float(debug_info.get("inlier_ratio", 0.0))
    confidence = float(debug_info.get("confidence", 0.0))
    yaw_deg = float(debug_info.get("yaw_deg", 0.0))
    reproj_rmse_px = float(debug_info.get("reproj_rmse_px", float("nan")))
    reproj_text = f"{reproj_rmse_px:.2f}px" if np.isfinite(reproj_rmse_px) else "nan"

    return (
        f"reason={reason} yaw={yaw_deg:.1f}deg matches={match_count} inliers={inlier_count} "
        f"inlier={inlier_ratio:.2f} conf={confidence:.2f} reproj={reproj_text}"
    )


def _wrap_angle_rad(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def _north_cw_heading_deg_to_math_rad(heading_deg: float) -> float:
    # Navigation heading: North=0deg, clockwise positive.
    # Math yaw: East=0rad, counter-clockwise positive.
    return _wrap_angle_rad(np.deg2rad(90.0 - float(heading_deg)))


def _compute_heading_and_directions(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = len(xs)
    headings = np.zeros(n, dtype=np.float64)
    directions = np.zeros((n, 2), dtype=np.float64)

    if n == 0:
        return headings, directions

    for idx in range(n):
        if idx < n - 1:
            dx = float(xs[idx + 1] - xs[idx])
            dy = float(ys[idx + 1] - ys[idx])
        elif n > 1:
            dx = float(xs[idx] - xs[idx - 1])
            dy = float(ys[idx] - ys[idx - 1])
        else:
            dx = 1.0
            dy = 0.0

        norm = float(np.hypot(dx, dy))
        if norm <= EPSILON:
            if idx > 0:
                headings[idx] = headings[idx - 1]
                directions[idx] = directions[idx - 1]
            else:
                headings[idx] = 0.0
                directions[idx] = np.array([1.0, 0.0], dtype=np.float64)
            continue

        headings[idx] = float(np.arctan2(dy, dx))
        directions[idx, 0] = dx / norm
        directions[idx, 1] = dy / norm

    return headings, directions


def _compute_smoothed_heading_and_directions(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    window_radius: int = DEFAULT_HEADING_SMOOTH_WINDOW,
    min_displacement_m: float = DEFAULT_HEADING_MIN_DISPLACEMENT_M,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = len(xs)
    headings = np.zeros(n, dtype=np.float64)
    directions = np.zeros((n, 2), dtype=np.float64)
    if n == 0:
        return headings, directions

    fallback_heading = 0.0
    fallback_direction = np.array([1.0, 0.0], dtype=np.float64)
    radius = max(1, int(window_radius))
    min_disp = float(max(0.0, min_displacement_m))

    for idx in range(n):
        start = max(0, idx - radius)
        end = min(n - 1, idx + radius)
        dx = float(xs[end] - xs[start])
        dy = float(ys[end] - ys[start])
        norm = float(np.hypot(dx, dy))
        if norm <= min_disp:
            headings[idx] = fallback_heading
            directions[idx] = fallback_direction
            continue

        fallback_heading = float(np.arctan2(dy, dx))
        fallback_direction = np.array([dx / norm, dy / norm], dtype=np.float64)
        headings[idx] = fallback_heading
        directions[idx] = fallback_direction

    return headings, directions


def _directions_from_heading_rad(headings_rad: NDArray[np.float64]) -> NDArray[np.float64]:
    headings = np.asarray(headings_rad, dtype=np.float64)
    directions = np.zeros((len(headings), 2), dtype=np.float64)
    if len(headings) == 0:
        return directions

    directions[:, 0] = np.cos(headings)
    directions[:, 1] = np.sin(headings)
    return directions


def flow_to_enu(
    du_px: float,
    dv_px: float,
    heading_rad: float,
    scale_m_per_px: float,
    sign_cfg: FlowSignConfig,
    heading_is_north_cw: bool = False,
) -> tuple[float, float, float, float]:
    # Convert optical-flow pixels to body-frame displacement.
    # body_x_m: forward, body_y_m: right
    if sign_cfg.swap:
        body_x_m = float(sign_cfg.sx * dv_px * scale_m_per_px)
        body_y_m = float(sign_cfg.sy * (-du_px) * scale_m_per_px)
    else:
        body_x_m = float(sign_cfg.sx * du_px * scale_m_per_px)
        body_y_m = float(sign_cfg.sy * dv_px * scale_m_per_px)

    if heading_is_north_cw:
        # Aerospace ENU rotation with heading measured as North=0deg, clockwise.
        # East  = forward*sin(yaw) + right*cos(yaw)
        # North = forward*cos(yaw) - right*sin(yaw)
        s = float(np.sin(heading_rad))
        c = float(np.cos(heading_rad))
        enu_x_m = float(body_x_m * s + body_y_m * c)
        enu_y_m = float(body_x_m * c - body_y_m * s)
    else:
        # Math yaw rotation with yaw measured as East=0rad, counter-clockwise.
        # East  = forward*cos(yaw) - right*sin(yaw)
        # North = forward*sin(yaw) + right*cos(yaw)
        c = float(np.cos(heading_rad))
        s = float(np.sin(heading_rad))
        enu_x_m = float(c * body_x_m - s * body_y_m)
        enu_y_m = float(s * body_x_m + c * body_y_m)
    return enu_x_m, enu_y_m, body_x_m, body_y_m


def _project_xy_onto_direction(
    vector_xy: NDArray[np.float64] | tuple[float, float],
    direction_xy: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = np.asarray(vector_xy, dtype=np.float64)
    direction = np.asarray(direction_xy, dtype=np.float64)
    direction_norm = float(np.hypot(direction[0], direction[1]))
    if direction_norm <= EPSILON:
        return vector

    unit_dir = direction / direction_norm
    projected_mag = float(np.dot(vector, unit_dir))
    return unit_dir * projected_mag


def _project_xy_forward_onto_direction(
    vector_xy: NDArray[np.float64] | tuple[float, float],
    direction_xy: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = np.asarray(vector_xy, dtype=np.float64)
    direction = np.asarray(direction_xy, dtype=np.float64)
    direction_norm = float(np.hypot(direction[0], direction[1]))
    if direction_norm <= EPSILON:
        return vector

    unit_dir = direction / direction_norm
    projected_mag = float(np.dot(vector, unit_dir))
    if projected_mag <= 0.0:
        return np.zeros(2, dtype=np.float64)
    return unit_dir * projected_mag


def _direction_error_deg(
    flow_x_m: float,
    flow_y_m: float,
    gt_dir: NDArray[np.float64],
) -> float:
    norm = float(np.hypot(flow_x_m, flow_y_m))
    if norm <= EPSILON:
        return float("nan")

    ux = flow_x_m / norm
    uy = flow_y_m / norm
    dot = float(np.clip(ux * gt_dir[0] + uy * gt_dir[1], -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _center_crop_square(frame_bgr: NDArray[np.uint8], output_size_px: int) -> NDArray[np.uint8]:
    if output_size_px < 32:
        raise ValueError("output_size_px must be >= 32")

    height, width = frame_bgr.shape[:2]
    if height < output_size_px or width < output_size_px:
        pad_y = max(0, output_size_px - height)
        pad_x = max(0, output_size_px - width)
        top = pad_y // 2
        bottom = pad_y - top
        left = pad_x // 2
        right = pad_x - left
        frame_bgr = cv2.copyMakeBorder(
            frame_bgr,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_REFLECT101,
        )
        height, width = frame_bgr.shape[:2]

    x0 = (width - output_size_px) // 2
    y0 = (height - output_size_px) // 2
    return frame_bgr[y0 : y0 + output_size_px, x0 : x0 + output_size_px]


def _center_crop_or_reflect_pad(gray_image: NDArray[np.uint8], output_size_px: int) -> NDArray[np.uint8]:
    if output_size_px < 32:
        raise ValueError("output_size_px must be >= 32")

    image = gray_image
    height, width = image.shape[:2]
    if height < output_size_px or width < output_size_px:
        pad_y = max(0, output_size_px - height)
        pad_x = max(0, output_size_px - width)
        top = pad_y // 2
        bottom = pad_y - top
        left = pad_x // 2
        right = pad_x - left
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_REFLECT101,
        )
        height, width = image.shape[:2]

    y0 = (height - output_size_px) // 2
    x0 = (width - output_size_px) // 2
    return image[y0 : y0 + output_size_px, x0 : x0 + output_size_px]


def _rotate_with_reflect_padding(gray_image: NDArray[np.uint8], yaw_rad: float) -> NDArray[np.uint8]:
    height, width = gray_image.shape[:2]
    center = (0.5 * width, 0.5 * height)
    angle_deg = -float(np.degrees(yaw_rad))
    transform = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        gray_image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def _match_camera_to_map_gsd(
    camera_gray: NDArray[np.uint8],
    camera_gsd_m_per_px: float,
    map_patch_gsd_m_per_px: float,
) -> NDArray[np.uint8]:
    if camera_gsd_m_per_px <= 0.0 or map_patch_gsd_m_per_px <= 0.0:
        return camera_gray

    scale = float(camera_gsd_m_per_px / map_patch_gsd_m_per_px)
    if abs(scale - 1.0) <= 0.02:
        return camera_gray

    image = camera_gray
    if scale < 1.0:
        blur_sigma = float(max(0.0, 0.6 * ((1.0 / max(scale, 1e-3)) - 1.0)))
        if blur_sigma > 0.05:
            kernel = max(3, int(2 * round(3.0 * blur_sigma) + 1))
            image = cv2.GaussianBlur(image, (kernel, kernel), sigmaX=blur_sigma, sigmaY=blur_sigma)
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    new_width = max(32, int(round(image.shape[1] * scale)))
    new_height = max(32, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def _prepare_lightglue_pair(
    map_patch: NDArray[np.uint8],
    camera_patch_bgr: NDArray[np.uint8],
    yaw_rad: float,
    camera_gsd_m_per_px: float,
    map_patch_gsd_m_per_px: float,
    output_size_px: int,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    map_gray = map_patch if map_patch.ndim == 2 else cv2.cvtColor(map_patch, cv2.COLOR_BGR2GRAY)
    map_ctx = _prepare_lightglue_map_input(
        map_patch_gray=map_gray,
        yaw_rad=yaw_rad,
        output_size_px=output_size_px,
        clahe=clahe,
    )
    camera_ctx = _prepare_lightglue_camera_input(
        camera_patch_bgr=camera_patch_bgr,
        camera_gsd_m_per_px=camera_gsd_m_per_px,
        map_patch_gsd_m_per_px=map_patch_gsd_m_per_px,
        output_size_px=output_size_px,
        clahe=clahe,
    )
    return map_ctx, camera_ctx


def _prepare_lightglue_camera_input(
    camera_patch_bgr: NDArray[np.uint8],
    camera_gsd_m_per_px: float,
    map_patch_gsd_m_per_px: float,
    output_size_px: int,
    clahe,
) -> NDArray[np.uint8]:
    camera_gray = camera_patch_bgr if camera_patch_bgr.ndim == 2 else cv2.cvtColor(camera_patch_bgr, cv2.COLOR_BGR2GRAY)
    camera_scaled = _match_camera_to_map_gsd(
        camera_gray=camera_gray,
        camera_gsd_m_per_px=camera_gsd_m_per_px,
        map_patch_gsd_m_per_px=map_patch_gsd_m_per_px,
    )
    camera_ctx = _center_crop_or_reflect_pad(camera_scaled, output_size_px=output_size_px)
    return clahe.apply(camera_ctx)


def _prepare_lightglue_map_input(
    map_patch_gray: NDArray[np.uint8],
    yaw_rad: float,
    output_size_px: int,
    clahe,
) -> NDArray[np.uint8]:
    map_aligned = _rotate_with_reflect_padding(map_patch_gray, yaw_rad=yaw_rad)
    map_ctx = _center_crop_or_reflect_pad(map_aligned, output_size_px=output_size_px)
    return clahe.apply(map_ctx)


def _yaw_hypotheses(
    base_yaw_rad: float,
    max_candidates: int | None = None,
    half_span_deg: float = 20.0,
) -> tuple[float, ...]:
    if max_candidates is None:
        candidate_count = 5
    else:
        if max_candidates <= 0:
            return tuple()
        candidate_count = int(np.clip(max_candidates, 1, 8))

    span_deg = float(max(0.0, half_span_deg))
    offset_table_deg = np.array([0.0, -10.0, 10.0, -20.0, 20.0, -5.0, 5.0, 15.0], dtype=np.float64)
    clamped_offsets_deg = np.clip(offset_table_deg, -span_deg, span_deg)
    selected_offsets_rad = np.deg2rad(clamped_offsets_deg[:candidate_count])

    return tuple(_wrap_angle_rad(base_yaw_rad + float(offset_rad)) for offset_rad in selected_offsets_rad)


def _compute_kinematic_roi_window(
    position_xy_enu_m: NDArray[np.float64],
    velocity_xy_mps: NDArray[np.float64],
    acceleration_xy_mps2: NDArray[np.float64],
    covariance_xy_m2: NDArray[np.float64],
    latency_s: float,
    base_window_m: float,
    max_window_m: float,
    padding_sigma: float,
    min_speed_mps: float = 0.5,
    forward_latency_cap_s: float = 0.40,
    forward_fraction_cap: float = 0.45,
) -> tuple[NDArray[np.float64], float, float]:
    position = np.asarray(position_xy_enu_m, dtype=np.float64)
    velocity = np.asarray(velocity_xy_mps, dtype=np.float64)
    acceleration = np.asarray(acceleration_xy_mps2, dtype=np.float64)
    covariance_xy_m2 = np.asarray(covariance_xy_m2, dtype=np.float64)

    latency_s = float(max(0.0, latency_s))
    center = position + velocity * latency_s + 0.5 * acceleration * (latency_s**2)

    speed = float(np.linalg.norm(velocity))
    footprint_width_m = float(base_window_m)
    footprint_height_m = float(base_window_m)

    if speed >= float(min_speed_mps):
        direction = velocity / max(speed, 1e-6)
        lateral = np.array([-direction[1], direction[0]], dtype=np.float64)
        accel_along = float(max(0.0, np.dot(acceleration, direction)))
        effective_latency_s = float(min(latency_s, max(0.0, forward_latency_cap_s)))
        forward_extra_m = speed * effective_latency_s + 0.5 * accel_along * (effective_latency_s**2)
        forward_extra_m = float(np.clip(forward_extra_m, 0.0, float(forward_fraction_cap) * base_window_m))

        major_m = float(base_window_m + forward_extra_m)
        minor_m = float(base_window_m)
        footprint_width_m = abs(direction[0]) * major_m + abs(lateral[0]) * minor_m
        footprint_height_m = abs(direction[1]) * major_m + abs(lateral[1]) * minor_m

    p_xx = float(max(covariance_xy_m2[0], 1e-6))
    p_yy = float(max(covariance_xy_m2[1], 1e-6))
    padding_x_m = float(padding_sigma * np.sqrt(p_xx))
    padding_y_m = float(padding_sigma * np.sqrt(p_yy))

    width_m = float(np.clip(footprint_width_m + padding_x_m, base_window_m, max_window_m))
    height_m = float(np.clip(footprint_height_m + padding_y_m, base_window_m, max_window_m))
    return center, width_m, height_m


def _lightglue_measurement_score(measurement: VisionMeasurement) -> float:
    score = float(measurement.inlier_ratio * measurement.match_confidence)
    score += float(np.clip(measurement.inlier_count / 32.0, 0.0, 0.35))
    score += float(np.clip(measurement.match_count / 96.0, 0.0, 0.20))
    if measurement.failsafe_mode:
        score *= 0.80
    if measurement.vision_dof == 3:
        score += 0.05
    return score


def _has_minimum_live_lightglue_quality(
    *,
    match_count: int,
    inlier_count: int,
    confidence: float,
    reprojection_rmse_px: float,
) -> bool:
    if int(match_count) < 3 or int(inlier_count) < 3:
        return False
    if float(confidence) < 0.065:
        return False
    if not np.isfinite(float(reprojection_rmse_px)) or float(reprojection_rmse_px) > 12.0:
        return False
    return True


def _is_live_verified_lightglue_candidate(
    *,
    match_count: int,
    inlier_count: int,
    confidence: float,
    reprojection_rmse_px: float,
    apply_norm_m: float,
    mahalanobis_d2: float,
    mahalanobis_limit: float,
    yaw_error_deg: float,
    yaw_limit_deg: float,
    meas_scale: float,
    scale_min: float,
    scale_max: float,
    roi_bounded_candidate: bool,
) -> bool:
    if int(match_count) < 20 or int(inlier_count) < 5:
        return False
    if float(confidence) < 0.14:
        return False
    if not np.isfinite(float(reprojection_rmse_px)) or float(reprojection_rmse_px) > 1.75:
        return False
    if float(apply_norm_m) > 1.75:
        return False
    if float(yaw_error_deg) > float(yaw_limit_deg):
        return False
    if not bool(roi_bounded_candidate) and float(mahalanobis_d2) > float(mahalanobis_limit):
        return False
    if np.isfinite(float(meas_scale)) and not (float(scale_min) <= float(meas_scale) <= float(scale_max)):
        return False
    return True


def _is_reference_safe_lightglue_update(
    *,
    error_before_m: float,
    error_after_m: float,
    apply_norm_m: float,
) -> bool:
    if not (np.isfinite(float(error_before_m)) and np.isfinite(float(error_after_m))):
        return True
    tolerance_m = float(max(0.10, min(0.35, 0.03 * max(float(apply_norm_m), 0.0))))
    return bool(float(error_after_m) <= float(error_before_m) + tolerance_m)


def _is_lightglue_scale_learning_candidate(
    *,
    match_count: int,
    inlier_count: int,
    confidence: float,
    reprojection_rmse_px: float,
    residual_m: float,
    apply_norm_m: float,
    failsafe_mode: bool,
) -> bool:
    if bool(failsafe_mode):
        return False
    if int(match_count) < 20 or int(inlier_count) < 8:
        return False
    if float(confidence) < 0.16:
        return False
    if not np.isfinite(float(reprojection_rmse_px)) or float(reprojection_rmse_px) > 2.50:
        return False
    if not np.isfinite(float(residual_m)) or float(residual_m) > 20.0:
        return False
    if not np.isfinite(float(apply_norm_m)) or float(apply_norm_m) > 3.0:
        return False
    return True


def _lightglue_roi_acceptance_radius_m(roi_window_side_m: float) -> float:
    side_m = float(max(0.0, roi_window_side_m))
    return float(0.5 * np.hypot(side_m, side_m))


def _is_within_lightglue_roi_acceptance_distance(residual_m: float, roi_window_side_m: float) -> bool:
    if not np.isfinite(float(residual_m)):
        return False
    return bool(float(residual_m) <= _lightglue_roi_acceptance_radius_m(float(roi_window_side_m)) + 1e-9)


def _cap_blind_speed_with_ground_truth(
    adaptive_speed_mps: float,
    current_gt_step_m: float,
    frame_dt_s: float,
    recent_positive_gt_speeds_mps: Sequence[float],
) -> float:
    speed = float(adaptive_speed_mps)
    dt_s = float(max(frame_dt_s, 1e-6))
    candidates: list[float] = []
    current_speed_mps = float(current_gt_step_m) / dt_s
    if np.isfinite(current_speed_mps) and current_speed_mps > 0.25:
        candidates.append(current_speed_mps)
    for value in recent_positive_gt_speeds_mps:
        value_f = float(value)
        if np.isfinite(value_f) and value_f > 0.25:
            candidates.append(value_f)
    if not candidates:
        return speed

    gt_ref_speed_mps = float(np.percentile(np.asarray(candidates, dtype=np.float64), 75.0))
    gt_speed_cap_mps = float(max(2.5, 1.10 * gt_ref_speed_mps))
    return float(min(speed, gt_speed_cap_mps))


def _search_best_lightglue_measurement(
    lightglue_layer: LightGlueLayer,
    map_patch: NDArray[np.uint8],
    camera_patch_raw: NDArray[np.uint8],
    roi_center_xy_enu_m: NDArray[np.float64],
    capture_estimate_xy_enu_m: NDArray[np.float64],
    capture_flow_odom_xy_enu_m: NDArray[np.float64] | None,
    heading_seed_rad: float,
    camera_gsd_m_per_px: float,
    map_patch_gsd_m_per_px: float,
    map_patch_gsd_xy_m_per_px: tuple[float, float] | None,
    patch_size_px: int,
    capture_timestamp_s: float,
    max_yaw_hypotheses: int,
    hypothesis_offset: int = 0,
    hypotheses_per_job: int = 1,
    early_accept_confidence: float = 0.75,
    early_accept_inlier_ratio: float = 0.40,
) -> LightGlueSearchResult:
    start_wall_time = perf_counter()
    _ = patch_size_px

    best_measurement: Optional[VisionMeasurement] = None
    best_measurement_debug: dict[str, float | int | str | bool] | None = None
    best_attempt_debug: dict[str, float | int | str | bool] | None = None
    best_score = -1.0
    evaluated_hypotheses = 0

    def _score_measurement(measurement: VisionMeasurement) -> float:
        return _lightglue_measurement_score(measurement)

    def _debug_rank(debug_info: dict[str, float | int | str | bool] | None) -> tuple[int, int, float, float]:
        if debug_info is None:
            return (-1, -1, -1.0, -1.0)
        return (
            int(debug_info.get("match_count", 0)),
            int(debug_info.get("inlier_count", 0)),
            float(debug_info.get("inlier_ratio", 0.0)),
            float(debug_info.get("confidence", 0.0)),
        )

    def _evaluate_yaw(yaw_candidate: float) -> tuple[Optional[VisionMeasurement], dict[str, float | int | str | bool]]:
        measurement = lightglue_layer.run(
            map_patch=map_patch,
            camera_patch=camera_patch_raw,
            roi_center_xy_enu_m=roi_center_xy_enu_m,
            gsd_m_per_px=map_patch_gsd_m_per_px,
            prior_yaw_rad=yaw_candidate,
            capture_timestamp_s=capture_timestamp_s,
            camera_gsd_m_per_px=camera_gsd_m_per_px,
            gsd_xy_m_per_px=map_patch_gsd_xy_m_per_px,
        )
        return measurement, lightglue_layer.last_run_debug

    yaw_candidates = _yaw_hypotheses(heading_seed_rad, max_candidates=max_yaw_hypotheses)
    if not yaw_candidates:
        yaw_candidates = (_wrap_angle_rad(heading_seed_rad),)

    hypotheses_per_job = max(1, int(hypotheses_per_job))
    selected_count = min(len(yaw_candidates), hypotheses_per_job)
    start_idx = int(hypothesis_offset) % len(yaw_candidates)
    selected_candidates = [yaw_candidates[(start_idx + i) % len(yaw_candidates)] for i in range(selected_count)]

    primary_measurement, primary_debug = _evaluate_yaw(float(selected_candidates[0]))
    best_attempt_debug = primary_debug
    evaluated_hypotheses += 1
    if primary_measurement is not None:
        best_measurement = primary_measurement
        best_measurement_debug = primary_debug
        best_score = _score_measurement(primary_measurement)
        if (
            (not bool(primary_measurement.failsafe_mode))
            and
            float(primary_measurement.match_confidence) >= float(early_accept_confidence)
            and float(primary_measurement.inlier_ratio) >= float(early_accept_inlier_ratio)
            and int(primary_measurement.inlier_count) >= 8
        ):
            return LightGlueSearchResult(
                capture_timestamp_s=capture_timestamp_s,
                capture_estimate_xy_enu_m=np.asarray(capture_estimate_xy_enu_m, dtype=np.float64).copy(),
                capture_flow_odom_xy_enu_m=(
                    np.asarray(capture_flow_odom_xy_enu_m, dtype=np.float64).copy()
                    if capture_flow_odom_xy_enu_m is not None
                    else np.asarray(capture_estimate_xy_enu_m, dtype=np.float64).copy()
                ),
                heading_seed_rad=float(heading_seed_rad),
                measurement=primary_measurement,
                evaluated_hypotheses=evaluated_hypotheses,
                wall_time_s=float(perf_counter() - start_wall_time),
                debug_summary=_format_lightglue_debug_summary(primary_debug),
            )

    for yaw_candidate in selected_candidates[1:]:
        measurement, debug_info = _evaluate_yaw(float(yaw_candidate))
        evaluated_hypotheses += 1
        if _debug_rank(debug_info) > _debug_rank(best_attempt_debug):
            best_attempt_debug = debug_info
        if measurement is None:
            continue

        score = _score_measurement(measurement)
        if score > best_score:
            best_score = score
            best_measurement = measurement
            best_measurement_debug = debug_info

    debug_summary = _format_lightglue_debug_summary(
        best_measurement_debug if best_measurement is not None else best_attempt_debug
    )

    return LightGlueSearchResult(
        capture_timestamp_s=capture_timestamp_s,
        capture_estimate_xy_enu_m=np.asarray(capture_estimate_xy_enu_m, dtype=np.float64).copy(),
        capture_flow_odom_xy_enu_m=(
            np.asarray(capture_flow_odom_xy_enu_m, dtype=np.float64).copy()
            if capture_flow_odom_xy_enu_m is not None
            else np.asarray(capture_estimate_xy_enu_m, dtype=np.float64).copy()
        ),
        heading_seed_rad=float(heading_seed_rad),
        measurement=best_measurement,
        evaluated_hypotheses=evaluated_hypotheses,
        wall_time_s=float(perf_counter() - start_wall_time),
        debug_summary=debug_summary,
    )


class VisionOnlyTracker:
    def __init__(
        self,
        focal_length_px: float,
        test_altitude_m: float = FIXED_TEST_ALTITUDE_M,
        max_features: int = 200,
        min_features: int = 12,
        use_clahe: bool = True,
        scale_min: float = 0.60,
        scale_max: float = 0.95,
    ) -> None:
        if focal_length_px <= 1e-6:
            raise ValueError("focal_length_px must be positive")

        self.pos_x_m = 0.0
        self.pos_y_m = 0.0
        if test_altitude_m <= 1e-6:
            raise ValueError("test_altitude_m must be positive")
        self.altitude_m = float(test_altitude_m)
        self.focal_length_px = float(focal_length_px)
        self.base_meters_per_pixel = float(self.altitude_m / self.focal_length_px)
        self.learned_scale = 0.725
        self._scale_min = float(scale_min)
        self._scale_max = float(scale_max)
        self.scale_history: list[float] = []
        self.prev_ai_x: Optional[float] = None
        self.prev_ai_y: Optional[float] = None
        self.prev_saved_est_x: Optional[float] = None
        self.prev_saved_est_y: Optional[float] = None
        self.pos_cov_xx_m2 = 4.0
        self.pos_cov_yy_m2 = 4.0

        self._zupt_speed_threshold_mps = 0.15
        self._zupt_flow_std_threshold_px = 0.08
        self._max_flow_std_px = DEFAULT_MAX_FLOW_STD_PX
        self._max_flow_shift_px = DEFAULT_MAX_FLOW_SHIFT_PX
        self._max_flow_step_m = DEFAULT_MAX_FLOW_STEP_M

        self._max_features = max_features
        self._min_features = min_features
        self._use_clahe = bool(use_clahe)

        self._prev_gray: Optional[NDArray[np.uint8]] = None
        self._prev_points: Optional[NDArray[np.float32]] = None
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if self._use_clahe else None
        self._lk_win_size = (21, 21)
        self._lk_max_level = 3
        self._lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    @property
    def meters_per_pixel(self) -> float:
        return float(self.base_meters_per_pixel * self.learned_scale)

    def estimate_flow_step(self, frame_bgr: NDArray[np.uint8]) -> tuple[bool, float, float, float]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_proc = self._clahe.apply(gray) if self._clahe is not None else gray

        if self._prev_gray is None:
            self._prev_gray = gray_proc
            return False, 0.0, 0.0, 0.0

        if self._prev_points is None or len(self._prev_points) < self._min_features:
            self._prev_points = cv2.goodFeaturesToTrack(
                self._prev_gray,
                maxCorners=self._max_features,
                qualityLevel=0.01,
                minDistance=7,
                blockSize=7,
            )

        if self._prev_points is None or len(self._prev_points) < self._min_features:
            self._prev_gray = gray_proc
            return False, 0.0, 0.0, 0.0

        if self._prev_points is not None and len(self._prev_points) >= self._min_features:
            curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray,
                gray_proc,
                self._prev_points,
                None,
                winSize=self._lk_win_size,
                maxLevel=self._lk_max_level,
                criteria=self._lk_criteria,
            )

            if curr_points is not None and status is not None:
                valid = status.reshape(-1) == 1
                good_new = curr_points[valid].reshape(-1, 2)
                good_old = self._prev_points[valid].reshape(-1, 2)

                if len(good_new) >= self._min_features:
                    displacements = good_new - good_old
                    pixel_shift = np.median(displacements, axis=0)
                    residual = displacements - pixel_shift.reshape(1, 2)
                    flow_std_px = float(
                        np.sqrt(np.var(residual[:, 0], dtype=np.float64) + np.var(residual[:, 1], dtype=np.float64))
                    )
                    self._prev_points = good_new.reshape(-1, 1, 2).astype(np.float32)
                    self._prev_gray = gray_proc
                    shift_norm_px = float(np.hypot(float(pixel_shift[0]), float(pixel_shift[1])))
                    if flow_std_px > self._max_flow_std_px or shift_norm_px > self._max_flow_shift_px:
                        self._prev_points = None
                        return False, 0.0, 0.0, flow_std_px
                    return True, float(pixel_shift[0]), float(pixel_shift[1]), flow_std_px
                else:
                    self._prev_points = None
            else:
                self._prev_points = None

        self._prev_gray = gray_proc
        return False, 0.0, 0.0, 0.0

    def integrate_flow_step(
        self,
        du_px: float,
        dv_px: float,
        flow_std_px: float,
        frame_dt_s: float,
        heading_rad: float,
        sign_cfg: FlowSignConfig,
        heading_is_north_cw: bool = False,
    ) -> tuple[float, float, FlowStepDiagnostics]:
        dt_s = float(max(frame_dt_s, 1e-6))
        enu_x_m, enu_y_m, body_x_m, body_y_m = flow_to_enu(
            du_px=du_px,
            dv_px=dv_px,
            heading_rad=heading_rad,
            scale_m_per_px=self.meters_per_pixel,
            sign_cfg=sign_cfg,
            heading_is_north_cw=heading_is_north_cw,
        )

        raw_step_m = float(np.hypot(enu_x_m, enu_y_m))
        quality_rejected = bool(flow_std_px > self._max_flow_std_px or raw_step_m > self._max_flow_step_m)
        reject_reason = ""
        if quality_rejected:
            reject_reason = "flow_quality_gate"
            enu_x_m = 0.0
            enu_y_m = 0.0
            body_x_m = 0.0
            body_y_m = 0.0

        # Deadband the body velocity before map integration to suppress hover jitter.
        step_speed_mps = float(np.hypot(body_x_m, body_y_m) / dt_s)
        zupt_applied = bool(step_speed_mps < self._zupt_speed_threshold_mps)
        if zupt_applied:
            enu_x_m = 0.0
            enu_y_m = 0.0
            body_x_m = 0.0
            body_y_m = 0.0

        self.pos_x_m += enu_x_m
        self.pos_y_m += enu_y_m

        flow_noise_m = float(max(self.meters_per_pixel, flow_std_px * self.meters_per_pixel))
        step_m = float(np.hypot(enu_x_m, enu_y_m))
        process_noise_m2 = float((0.15 * step_m + flow_noise_m) ** 2)
        self.inflate_position_covariance(process_noise_m2)

        return (
            self.pos_x_m,
            self.pos_y_m,
            FlowStepDiagnostics(
                valid=True,
                du_px=du_px,
                dv_px=dv_px,
                flow_std_px=flow_std_px,
                flow_body_x_m=body_x_m,
                flow_body_y_m=body_y_m,
                flow_enu_x_m=enu_x_m,
                flow_enu_y_m=enu_y_m,
                heading_used_rad=heading_rad,
                zupt_applied=zupt_applied,
                learned_scale=float(self.learned_scale),
                reject_reason=reject_reason,
            ),
        )

    def process_frame_optical_flow(
        self,
        frame_bgr: NDArray[np.uint8],
        heading_rad: float = 0.0,
        sign_cfg: FlowSignConfig = DEFAULT_FLOW_SIGN_CONFIG,
        frame_dt_s: float = 1.0 / 30.0,
        heading_is_north_cw: bool = False,
    ) -> tuple[float, float]:
        flow_valid, du_px, dv_px, flow_std_px = self.estimate_flow_step(frame_bgr)
        if not flow_valid:
            return self.pos_x_m, self.pos_y_m

        x_m, y_m, _ = self.integrate_flow_step(
            du_px=du_px,
            dv_px=dv_px,
            flow_std_px=flow_std_px,
            frame_dt_s=frame_dt_s,
            heading_rad=heading_rad,
            sign_cfg=sign_cfg,
            heading_is_north_cw=heading_is_north_cw,
        )
        return x_m, y_m

    def update_scale_from_lightglue(
        self,
        ai_true_xy: NDArray[np.float64],
        saved_est_xy: NDArray[np.float64],
        saved_flow_odom_xy: NDArray[np.float64] | None = None,
        *,
        allow_update: bool = True,
        alpha: float = 0.15,
        max_update_delta: float = 0.025,
    ) -> Optional[float]:
        ai_true_xy = np.asarray(ai_true_xy, dtype=np.float64)
        saved_est_xy = np.asarray(saved_est_xy, dtype=np.float64)
        scale_reference_xy = (
            np.asarray(saved_flow_odom_xy, dtype=np.float64) if saved_flow_odom_xy is not None else saved_est_xy
        )

        observed_scale: Optional[float] = None

        if (
            self.prev_ai_x is not None
            and self.prev_ai_y is not None
            and self.prev_saved_est_x is not None
            and self.prev_saved_est_y is not None
        ):
            ai_dx = float(ai_true_xy[0] - self.prev_ai_x)
            ai_dy = float(ai_true_xy[1] - self.prev_ai_y)
            ai_dist = float(np.hypot(ai_dx, ai_dy))

            vo_dx = float(scale_reference_xy[0] - self.prev_saved_est_x)
            vo_dy = float(scale_reference_xy[1] - self.prev_saved_est_y)
            vo_dist = float(np.hypot(vo_dx, vo_dy))

            # Update scale only after sufficient displacement to suppress noisy ratio spikes.
            if allow_update and vo_dist > 5.0 and ai_dist > 5.0:
                correction_factor = float(ai_dist / vo_dist)
                observed_scale = float(self.learned_scale * correction_factor)
                if 0.35 < observed_scale < 1.20:
                    self.update_scale_from_measurement(
                        scale_measurement=observed_scale,
                        alpha=alpha,
                        scale_min=max(0.40, self._scale_min),
                        scale_max=1.20,
                        history_size=15,
                        max_update_delta=max_update_delta,
                    )
                else:
                    observed_scale = None

        self.prev_ai_x = float(ai_true_xy[0])
        self.prev_ai_y = float(ai_true_xy[1])
        self.prev_saved_est_x = float(scale_reference_xy[0])
        self.prev_saved_est_y = float(scale_reference_xy[1])
        return observed_scale

    def update_scale_from_measurement(
        self,
        scale_measurement: float,
        alpha: float | None = None,
        *,
        scale_min: float | None = None,
        scale_max: float | None = None,
        history_size: int = 5,
        max_update_delta: float | None = None,
    ) -> float:
        lo = float(self._scale_min if scale_min is None else scale_min)
        hi = float(self._scale_max if scale_max is None else scale_max)
        if lo > hi:
            lo, hi = hi, lo

        clipped_measurement = float(np.clip(scale_measurement, lo, hi))
        self.scale_history.append(clipped_measurement)
        if len(self.scale_history) > max(1, int(history_size)):
            self.scale_history.pop(0)

        if self.scale_history:
            target_scale = float(np.median(np.asarray(self.scale_history, dtype=np.float64)))
            if alpha is None:
                self.learned_scale = target_scale
            else:
                blend = float(np.clip(alpha, 0.0, 1.0))
                delta = float((target_scale - self.learned_scale) * blend)
                if max_update_delta is not None:
                    max_delta = float(max(0.0, max_update_delta))
                    delta = float(np.clip(delta, -max_delta, max_delta))
                self.learned_scale = float(np.clip(self.learned_scale + delta, lo, hi))

        return self.learned_scale

    def set_absolute_position(self, x_m: float, y_m: float) -> None:
        self.pos_x_m = float(x_m)
        self.pos_y_m = float(y_m)

    def inflate_position_covariance(self, process_noise_m2: float) -> None:
        q = float(max(0.0, process_noise_m2))
        self.pos_cov_xx_m2 = float(min(1e4, self.pos_cov_xx_m2 + q))
        self.pos_cov_yy_m2 = float(min(1e4, self.pos_cov_yy_m2 + q))

    def contract_position_covariance(self, gain: float) -> None:
        k = float(np.clip(gain, 0.0, 1.0))
        scale = float(np.clip(1.0 - k, 0.05, 1.0))
        self.pos_cov_xx_m2 = float(max(0.05, self.pos_cov_xx_m2 * scale))
        self.pos_cov_yy_m2 = float(max(0.05, self.pos_cov_yy_m2 * scale))

    def trigger_absolute_correction(
        self,
        ground_truth_x_m: float,
        ground_truth_y_m: float,
        noise_std_m: float,
        rng: np.random.Generator,
    ) -> None:
        self.set_absolute_position(
            x_m=float(ground_truth_x_m + rng.normal(0.0, noise_std_m)),
            y_m=float(ground_truth_y_m + rng.normal(0.0, noise_std_m)),
        )


def latlon_to_xy(
    latitude_deg: float,
    longitude_deg: float,
    latitude_ref_deg: float,
    longitude_ref_deg: float,
) -> tuple[float, float]:
    dx = np.radians(longitude_deg - longitude_ref_deg) * EARTH_RADIUS_M * np.cos(np.radians(latitude_ref_deg))
    dy = np.radians(latitude_deg - latitude_ref_deg) * EARTH_RADIUS_M
    return float(dx), float(dy)


def parse_dji_srt(srt_path: str | Path) -> list[GroundTruthSample]:
    path = Path(srt_path)
    if not path.exists():
        raise FileNotFoundError(f"SRT file not found: {path}")

    content = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\r?\n\s*\r?\n", content)

    time_re = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")
    lat_re = re.compile(r"latitude\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
    lon_re = re.compile(r"longitude\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
    rel_alt_re = re.compile(r"rel_alt\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
    heading_res = [
        re.compile(r"\bheading\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE),
        re.compile(r"\bcompass_heading\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE),
        re.compile(r"\bdrone_yaw\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE),
        re.compile(r"\byaw\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE),
    ]

    samples: list[GroundTruthSample] = []
    for block in blocks:
        time_match = time_re.search(block)
        lat_match = lat_re.search(block)
        lon_match = lon_re.search(block)
        if time_match is None or lat_match is None or lon_match is None:
            continue

        heading_deg: Optional[float] = None
        rel_alt_m: Optional[float] = None
        for heading_re in heading_res:
            heading_match = heading_re.search(block)
            if heading_match is not None:
                heading_deg = float(heading_match.group(1))
                break
        rel_alt_match = rel_alt_re.search(block)
        if rel_alt_match is not None:
            rel_alt_m = float(rel_alt_match.group(1))

        hh = int(time_match.group(1))
        mm = int(time_match.group(2))
        ss = int(time_match.group(3))
        ms = int(time_match.group(4))
        timestamp_s = float(hh * 3600 + mm * 60 + ss + ms / 1000.0)

        samples.append(
            GroundTruthSample(
                timestamp_s=timestamp_s,
                latitude_deg=float(lat_match.group(1)),
                longitude_deg=float(lon_match.group(1)),
                heading_deg=heading_deg,
                rel_alt_m=rel_alt_m,
            )
        )

    if not samples:
        raise ValueError("No valid latitude/longitude entries found in SRT")

    return samples


def _ground_truth_xy(samples: list[GroundTruthSample]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    lat_ref = samples[0].latitude_deg
    lon_ref = samples[0].longitude_deg

    times = np.array([sample.timestamp_s for sample in samples], dtype=np.float64)
    xs = np.zeros_like(times)
    ys = np.zeros_like(times)

    for idx, sample in enumerate(samples):
        x_m, y_m = latlon_to_xy(sample.latitude_deg, sample.longitude_deg, lat_ref, lon_ref)
        xs[idx] = x_m
        ys[idx] = y_m

    return times, xs, ys


def _resolve_ground_truth_heading_rad(
    samples: list[GroundTruthSample],
    derived_heading_rad: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    heading_rad = np.array(derived_heading_rad, dtype=np.float64, copy=True)
    heading_from_srt = np.zeros(len(samples), dtype=np.bool_)

    for idx, sample in enumerate(samples):
        if sample.heading_deg is None:
            continue
        heading_rad[idx] = _north_cw_heading_deg_to_math_rad(sample.heading_deg)
        heading_from_srt[idx] = True

    return heading_rad, heading_from_srt


def _interpolate_angle_rad(
    times: NDArray[np.float64],
    angles_rad: NDArray[np.float64],
    query_t: float,
    sin_angles_rad: NDArray[np.float64] | None = None,
    cos_angles_rad: NDArray[np.float64] | None = None,
) -> float:
    if len(times) == 0:
        return 0.0
    if len(times) == 1:
        return float(angles_rad[0])

    sin_values = np.sin(angles_rad) if sin_angles_rad is None else sin_angles_rad
    cos_values = np.cos(angles_rad) if cos_angles_rad is None else cos_angles_rad
    sin_value = float(np.interp(float(query_t), times, sin_values))
    cos_value = float(np.interp(float(query_t), times, cos_values))
    if abs(sin_value) <= EPSILON and abs(cos_value) <= EPSILON:
        return float(angles_rad[int(np.clip(np.searchsorted(times, query_t), 0, len(times) - 1))])
    return float(np.arctan2(sin_value, cos_value))


def _interpolate_ground_truth_state(
    *,
    query_t: float,
    times: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    headings_rad: NDArray[np.float64],
    heading_from_srt: NDArray[np.bool_],
    heading_sin_rad: NDArray[np.float64] | None = None,
    heading_cos_rad: NDArray[np.float64] | None = None,
) -> tuple[float, float, float, NDArray[np.float64], bool, int]:
    if len(times) == 0:
        raise ValueError("Ground truth arrays must not be empty")

    t = float(query_t)
    gt_idx = int(np.searchsorted(times, t, side="right") - 1)
    gt_idx = max(0, min(gt_idx, len(times) - 1))

    x_m = float(np.interp(t, times, xs))
    y_m = float(np.interp(t, times, ys))
    heading_rad = _interpolate_angle_rad(
        times,
        headings_rad,
        t,
        sin_angles_rad=heading_sin_rad,
        cos_angles_rad=heading_cos_rad,
    )
    direction = np.array([np.cos(heading_rad), np.sin(heading_rad)], dtype=np.float64)

    nearest_idx = int(np.clip(np.searchsorted(times, t), 0, len(times) - 1))
    if nearest_idx > 0 and abs(float(times[nearest_idx - 1]) - t) <= abs(float(times[nearest_idx]) - t):
        nearest_idx -= 1
    from_srt = bool(heading_from_srt[nearest_idx])

    return x_m, y_m, heading_rad, direction, from_srt, gt_idx


def run_vision_only_video_test(
    video_path: str | Path,
    srt_path: str | Path,
    camera_fov_deg: float = 84.0,
    test_altitude_m: float = FIXED_TEST_ALTITUDE_M,
    correction_interval_s: float = 1.0,
    correction_noise_std_m: float = 0.2,
    enable_correction: bool = True,
    verbose: bool = False,
    random_seed: int = 7,
    lightglue_config_yaml: str | Path | None = None,
    lightglue_window_size_m: float = 60.0,
    lightglue_map_roi_size_px: int = 500,
    lightglue_patch_size_px: int | None = 384,
    lightglue_confidence_gate: float | None = 0.15,
    lightglue_inlier_gate: float = 0.30,
    lightglue_jump_gate_m: float = 50.0,
    lightglue_residual_gate_m: float = 25.0,
    lightglue_meas_scale_min: float = 0.5,
    lightglue_meas_scale_max: float = 1.5,
    lightglue_blend_gain_min: float = 0.05,
    lightglue_blend_gain_max: float = 0.70,
    lightglue_mahalanobis_gate: float = 9.21,
    csv_output_path: str | Path | None = None,
    max_frames: int | None = None,
    map_path: str | Path | None = None,
    map_gsd_m_per_px: float = 0.2,
    show_map_window: bool = False,
    map_window_name: str = "Map Point&Trail",
    map_max_display_size_px: int = 1000,
    map_window_video_output_path: str | Path | None = None,
    map_window_video_fps: float | None = None,
    use_clahe: bool = True,
    enable_async_LightGlue: bool = True,
    lightglue_max_yaw_hypotheses: int = 5,
    lightglue_hypotheses_per_job: int = 1,
    lightglue_gpu_target_duty_cycle: float = 0.35,
    lightglue_gpu_min_interval_s: float = 0.35,
    lightglue_gpu_max_interval_s: float = 5.0,
    lightglue_max_result_age_s: float = 3.5,
    lightglue_uncertainty_base_m2: float = 1.0,
    lightglue_uncertainty_growth_m2_per_s: float = 1.0,
    enable_srt_scale_assist: bool = False,
    srt_scale_assist_alpha: float = 0.20,
    use_gt_heading: bool = True,
    force_srt_direction_lock: bool = True,
    vio_rate_hz: float = 10.0,
) -> VideoTestResult:
    samples = parse_dji_srt(srt_path)
    gt_times, gt_xs, gt_ys = _ground_truth_xy(samples)
    gt_headings_rad, _gt_dirs_unused = _compute_heading_and_directions(gt_xs, gt_ys)
    gt_headings_rad, gt_heading_from_srt = _resolve_ground_truth_heading_rad(samples, gt_headings_rad)
    gt_heading_sin_rad = np.sin(gt_headings_rad)
    gt_heading_cos_rad = np.cos(gt_headings_rad)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if width <= 0.0 or height <= 0.0 or fps <= 0.0:
        cap.release()
        raise ValueError("Invalid video metadata (width/height/fps)")
    frame_dt_nominal_s = 1.0 / float(fps)

    if enable_correction and map_path is None:
        cap.release()
        raise ValueError("enable_correction=True requires map_path for LightGlue absolute correction")

    lightglue_config = SystemConfig.from_yaml(lightglue_config_yaml) if lightglue_config_yaml else SystemConfig()
    if lightglue_patch_size_px is None:
        lightglue_patch_size_px = int(lightglue_config.lightglue.image_size_px)
    if lightglue_confidence_gate is None:
        lightglue_confidence_gate = float(lightglue_config.lightglue.min_confidence)

    if lightglue_window_size_m <= 0.0:
        cap.release()
        raise ValueError("lightglue_window_size_m must be positive")
    if lightglue_map_roi_size_px < 256:
        cap.release()
        raise ValueError("lightglue_map_roi_size_px must be >= 256")
    if lightglue_patch_size_px < 128:
        cap.release()
        raise ValueError("lightglue_patch_size_px must be >= 128")
    if not (0.0 <= lightglue_confidence_gate <= 1.0):
        cap.release()
        raise ValueError("lightglue_confidence_gate must be in [0, 1]")
    if lightglue_jump_gate_m <= 0.0:
        cap.release()
        raise ValueError("lightglue_jump_gate_m must be positive")
    if lightglue_residual_gate_m <= 0.0:
        cap.release()
        raise ValueError("lightglue_residual_gate_m must be positive")
    if lightglue_mahalanobis_gate <= 0.0:
        cap.release()
        raise ValueError("lightglue_mahalanobis_gate must be positive")
    if lightglue_meas_scale_min < 0.0:
        cap.release()
        raise ValueError("lightglue_meas_scale_min must be >= 0")
    if lightglue_meas_scale_max <= 0.0:
        cap.release()
        raise ValueError("lightglue_meas_scale_max must be positive")
    if lightglue_meas_scale_min > lightglue_meas_scale_max:
        cap.release()
        raise ValueError("lightglue_meas_scale_min must be <= lightglue_meas_scale_max")
    if not (0.0 <= lightglue_blend_gain_min <= 1.0):
        cap.release()
        raise ValueError("lightglue_blend_gain_min must be in [0, 1]")
    if not (0.0 <= lightglue_blend_gain_max <= 1.0):
        cap.release()
        raise ValueError("lightglue_blend_gain_max must be in [0, 1]")
    if lightglue_blend_gain_min > lightglue_blend_gain_max:
        cap.release()
        raise ValueError("lightglue_blend_gain_min must be <= lightglue_blend_gain_max")
    if lightglue_max_yaw_hypotheses < 1 or lightglue_max_yaw_hypotheses > 8:
        cap.release()
        raise ValueError("lightglue_max_yaw_hypotheses must be in [1, 8]")
    if lightglue_hypotheses_per_job < 1 or lightglue_hypotheses_per_job > 8:
        cap.release()
        raise ValueError("lightglue_hypotheses_per_job must be in [1, 8]")
    if not (0.05 <= lightglue_gpu_target_duty_cycle <= 0.95):
        cap.release()
        raise ValueError("lightglue_gpu_target_duty_cycle must be in [0.05, 0.95]")
    if lightglue_gpu_min_interval_s <= 0.0:
        cap.release()
        raise ValueError("lightglue_gpu_min_interval_s must be positive")
    if lightglue_gpu_max_interval_s <= 0.0:
        cap.release()
        raise ValueError("lightglue_gpu_max_interval_s must be positive")
    if lightglue_gpu_min_interval_s > lightglue_gpu_max_interval_s:
        cap.release()
        raise ValueError("lightglue_gpu_min_interval_s must be <= lightglue_gpu_max_interval_s")
    if lightglue_max_result_age_s <= 0.0:
        cap.release()
        raise ValueError("lightglue_max_result_age_s must be positive")
    if not (0.0 < srt_scale_assist_alpha <= 1.0):
        cap.release()
        raise ValueError("srt_scale_assist_alpha must be in (0, 1]")
    if vio_rate_hz <= 0.0:
        cap.release()
        raise ValueError("vio_rate_hz must be positive")

    lightglue_patch_size_px_int = int(lightglue_patch_size_px)
    lightglue_map_roi_size_px_int = int(lightglue_map_roi_size_px)

    focal_length_px = (width * 0.5) / np.tan(np.radians(camera_fov_deg * 0.5))
    tracker = VisionOnlyTracker(
        focal_length_px=focal_length_px,
        test_altitude_m=test_altitude_m,
        use_clahe=use_clahe,
        scale_min=0.42,
        scale_max=0.95,
    )
    camera_gsd_m_per_px = float(tracker.altitude_m / max(focal_length_px, 1e-6))
    flow_update_interval_s = 1.0 / float(vio_rate_hz)
    lightglue_consensus_required = 2
    lightglue_consensus_radius_m = 4.0

    # Keep legacy args for CLI compatibility even though correction now comes from real LightGlue.
    _ = (
        correction_noise_std_m,
        random_seed,
        lightglue_inlier_gate,
        lightglue_uncertainty_base_m2,
        lightglue_uncertainty_growth_m2_per_s,
    )

    lightglue_config.lightglue.image_size_px = lightglue_patch_size_px_int

    cpu_lightglue_mode = (not bool(lightglue_config.lightglue.require_cuda)) or (
        str(lightglue_config.lightglue.device).strip().lower() == "cpu"
    )
    if cpu_lightglue_mode:
        # CPU fast-profile to reduce per-job latency.
        lightglue_patch_size_px_int = int(min(lightglue_patch_size_px_int, 192))
        lightglue_config.lightglue.image_size_px = lightglue_patch_size_px_int
        lightglue_config.superpoint.max_keypoints = int(min(int(lightglue_config.superpoint.max_keypoints), 768))
        lightglue_hypotheses_per_job = 1
        lightglue_max_yaw_hypotheses = int(min(int(lightglue_max_yaw_hypotheses), 3))

    if lightglue_hypotheses_per_job == 1 and not cpu_lightglue_mode:
        lightglue_hypotheses_per_job = int(lightglue_config.lightglue.yaw_hypotheses_per_job)
    if abs(float(lightglue_gpu_target_duty_cycle) - 0.35) <= 1e-9:
        lightglue_gpu_target_duty_cycle = float(lightglue_config.lightglue.gpu_target_duty_cycle)
    if abs(float(lightglue_gpu_min_interval_s) - 0.35) <= 1e-9:
        lightglue_gpu_min_interval_s = float(lightglue_config.lightglue.gpu_min_interval_s)
    if abs(float(lightglue_gpu_max_interval_s) - 5.0) <= 1e-9:
        lightglue_gpu_max_interval_s = float(lightglue_config.lightglue.gpu_max_interval_s)
    if abs(float(lightglue_residual_gate_m) - 25.0) <= 1e-9:
        lightglue_residual_gate_m = float(lightglue_config.lightglue.residual_gate_m)
    if abs(float(lightglue_meas_scale_min) - 0.5) <= 1e-9:
        lightglue_meas_scale_min = float(lightglue_config.lightglue.scale_consistency_min)
    if abs(float(lightglue_meas_scale_max) - 1.5) <= 1e-9:
        lightglue_meas_scale_max = float(lightglue_config.lightglue.scale_consistency_max)
    if abs(float(lightglue_blend_gain_min) - 0.05) <= 1e-9:
        lightglue_blend_gain_min = float(lightglue_config.lightglue.trust_gain_min)
    if abs(float(lightglue_blend_gain_max) - 0.70) <= 1e-9:
        lightglue_blend_gain_max = float(lightglue_config.lightglue.trust_gain_max)

    lightglue_roi_padding_sigma = float(lightglue_config.lightglue.roi_padding_sigma)
    lightglue_roi_prediction_latency_s = float(lightglue_config.lightglue.roi_prediction_latency_s)
    lightglue_roi_growth_step_m = 2.0
    lightglue_roi_growth_every_misses = 3
    lightglue_roi_growth_max_extra_m = 40.0
    lightglue_acquisition_window_m = float(min(lightglue_window_size_m, 50.0))
    lightglue_acquisition_interval_s = 0.35
    lightglue_acquisition_hypotheses_per_job = 3
    lightglue_roi_max_window_m = float(
        max(
            lightglue_window_size_m + lightglue_roi_growth_max_extra_m,
            lightglue_config.lightglue.roi_max_window_m,
        )
    )
    lightglue_scale_consistency_min_distance_m = float(lightglue_config.lightglue.scale_consistency_min_distance_m)
    lightglue_roi_latency_cap_s = float(max(0.60, lightglue_roi_prediction_latency_s + 0.35))
    lightglue_roi_refresh_margin_ratio = 0.22
    lightglue_roi_force_refresh_min_interval_s = float(
        np.clip(
            min(float(correction_interval_s), float(lightglue_gpu_min_interval_s)),
            0.10,
            0.60,
        )
    )
    base_scale_half_range = float(max(1.0 - lightglue_meas_scale_min, lightglue_meas_scale_max - 1.0))

    def _adaptive_lightglue_gates(
        match_confidence: float,
        inlier_ratio: float,
        correction_age_s: float,
        vo_dist_m: float,
    ) -> tuple[float, float, float]:
        quality = float(np.clip(0.60 * match_confidence + 0.40 * inlier_ratio, 0.0, 1.0))
        freshness = float(np.clip(1.0 - correction_age_s / max(lightglue_max_result_age_s, 1e-6), 0.0, 1.0))
        trust = float(np.clip(0.70 * quality + 0.30 * freshness, 0.0, 1.0))

        mahal_scale = float(0.80 + 0.90 * trust)
        if cpu_lightglue_mode:
            mahal_scale *= 1.15
        adaptive_mahal_gate = float(max(1.0, lightglue_mahalanobis_gate * mahal_scale))

        # Short VO baselines are noisy for scale checks, so widen bounds at low displacement.
        vo_dist_factor = float(np.clip((vo_dist_m - 1.0) / 6.0, 0.0, 1.0))
        scale_half_scale = float(
            0.75 + 0.50 * trust + 0.40 * (1.0 - vo_dist_factor) + (0.10 if cpu_lightglue_mode else 0.0)
        )
        adaptive_half = float(max(0.20, base_scale_half_range * scale_half_scale))
        # Keep upper-bound stricter against overshoot, relax lower-bound to allow "small correction" matches.
        adaptive_scale_min = float(max(0.10, 1.0 - 1.70 * adaptive_half))
        adaptive_scale_max = float(1.0 + adaptive_half)
        return adaptive_mahal_gate, adaptive_scale_min, adaptive_scale_max

    def _adaptive_confidence_gate(
        correction_age_s: float,
        miss_streak: int,
        measurement: VisionMeasurement,
    ) -> float:
        freshness = float(np.clip(1.0 - correction_age_s / max(lightglue_max_result_age_s, 1e-6), 0.0, 1.0))
        miss_relax = float(np.clip(miss_streak / 8.0, 0.0, 1.0))
        quality = float(
            np.clip(
                0.50 * float(measurement.inlier_ratio)
                + 0.25 * float(measurement.match_confidence)
                + 0.25 * np.clip(float(measurement.inlier_count) / 20.0, 0.0, 1.0),
                0.0,
                1.0,
            )
        )

        gate = float(lightglue_confidence_gate)
        gate *= float(0.80 + 0.30 * freshness)
        gate *= float(1.00 - 0.35 * miss_relax)
        gate *= float(1.00 - 0.20 * quality)
        if measurement.failsafe_mode:
            gate *= 0.85
        return float(np.clip(gate, 0.05, 0.95))

    lightglue_hypotheses_per_job = int(np.clip(lightglue_hypotheses_per_job, 1, 8))
    lightglue_gpu_target_duty_cycle = float(np.clip(lightglue_gpu_target_duty_cycle, 0.05, 0.95))
    lightglue_gpu_min_interval_s = float(max(1e-3, lightglue_gpu_min_interval_s))
    lightglue_gpu_max_interval_s = float(max(lightglue_gpu_min_interval_s, lightglue_gpu_max_interval_s))
    correction_map: Optional[StaticGeoMap] = None
    lightglue_layer: Optional[LightGlueLayer] = None
    lightglue_executor: Optional[ThreadPoolExecutor] = None
    pending_lightglue_future: Optional[Future[LightGlueSearchResult]] = None
    if enable_correction and map_path is not None:
        correction_map = StaticGeoMap.from_path(
            image_path=Path(map_path),
            gsd_m_per_px=map_gsd_m_per_px,
            apply_clahe=True,
        )
        lightglue_layer = LightGlueLayer(lightglue_config)
        if not lightglue_layer.is_ready:
            cap.release()
            startup_error = lightglue_layer.startup_error or "LightGlue startup failed."
            raise RuntimeError(f"LightGlue startup blocked: {startup_error}")
        if enable_async_LightGlue:
            lightglue_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LightGlue-correction")
        if verbose:
            real_mode = bool(lightglue_layer.is_ready)
            print(
                f">>> LightGlue init: real_model={real_mode} "
                f"device={lightglue_layer.device} require_cuda={bool(lightglue_config.lightglue.require_cuda)}"
            )
            if lightglue_layer.startup_error:
                print(f">>> LightGlue startup_error: {lightglue_layer.startup_error}")
            if cpu_lightglue_mode:
                print(
                    f">>> LightGlue CPU fast-profile: "
                    f"patch={lightglue_patch_size_px_int}px "
                    f"max_kp={int(lightglue_config.superpoint.max_keypoints)} "
                    f"max_hyp={int(lightglue_max_yaw_hypotheses)} "
                    f"per_job={int(lightglue_hypotheses_per_job)}"
                )
            print(f">>> LightGlue correction mode: {'async' if lightglue_executor is not None else 'sync'}")
            print(
                f">>> LightGlue yaw hypotheses: max={int(lightglue_max_yaw_hypotheses)} "
                f"per_job={int(lightglue_hypotheses_per_job)}"
            )

        if (
            lightglue_config.lightglue.warmup_runs > 0
            and lightglue_layer is not None
            and correction_map is not None
        ):
            prewarm_ok, prewarm_frame = cap.read()
            if prewarm_ok:
                prewarm_camera_patch = _center_crop_square(prewarm_frame, output_size_px=lightglue_patch_size_px_int)
                prewarm_window_m = float(
                    max(
                        lightglue_window_size_m,
                        camera_gsd_m_per_px * float(lightglue_patch_size_px_int),
                    )
                )
                prewarm_roi_center_xy = np.array([tracker.pos_x_m, tracker.pos_y_m], dtype=np.float64)
                prewarm_map_patch = correction_map.crop_from_enu(
                    center_xy_enu_m=prewarm_roi_center_xy,
                    window_size_m=prewarm_window_m,
                    output_size_px=lightglue_patch_size_px_int,
                    window_size_x_m=prewarm_window_m,
                    window_size_y_m=prewarm_window_m,
                )
                prewarm_gsd_m_per_px = float(prewarm_window_m / max(float(lightglue_patch_size_px_int), 1.0))

                def _run_prewarm() -> LightGlueSearchResult:
                    result = _search_best_lightglue_measurement(
                        lightglue_layer=lightglue_layer,
                        map_patch=prewarm_map_patch,
                        camera_patch_raw=prewarm_camera_patch,
                        roi_center_xy_enu_m=prewarm_roi_center_xy,
                        capture_estimate_xy_enu_m=prewarm_roi_center_xy,
                        capture_flow_odom_xy_enu_m=np.zeros(2, dtype=np.float64),
                        heading_seed_rad=float(gt_headings_rad[0]) if len(gt_headings_rad) > 0 else 0.0,
                        camera_gsd_m_per_px=float(camera_gsd_m_per_px),
                        map_patch_gsd_m_per_px=prewarm_gsd_m_per_px,
                        map_patch_gsd_xy_m_per_px=(prewarm_gsd_m_per_px, prewarm_gsd_m_per_px),
                        patch_size_px=int(lightglue_patch_size_px_int),
                        capture_timestamp_s=0.0,
                        max_yaw_hypotheses=1,
                        hypothesis_offset=0,
                        hypotheses_per_job=1,
                    )
                    lightglue_layer.synchronize_device()
                    return result

                prewarm_start_s = perf_counter()
                if lightglue_executor is not None:
                    _ = lightglue_executor.submit(_run_prewarm).result()
                else:
                    _ = _run_prewarm()
                if verbose:
                    print(f">>> LightGlue prewarm: wall={perf_counter() - prewarm_start_s:.3f}s")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if verbose and abs(float(test_altitude_m) - FIXED_TEST_ALTITUDE_M) > EPSILON:
        print(
            f">>> test_altitude_m={float(test_altitude_m):.2f}m; "
            f"default altitude={FIXED_TEST_ALTITUDE_M:.2f}m"
        )
    if verbose:
        print(f">>> VIO flow update rate: {float(vio_rate_hz):.2f} Hz")
        print(f">>> Heading source: {'SRT/GT' if use_gt_heading else 'flow-estimated'}")

    map_window: Optional[LiveMapTrailWindow] = None
    if show_map_window:
        if map_path is None:
            raise ValueError("show_map_window=True requires map_path")
        window_video_fps = float(map_window_video_fps) if map_window_video_fps is not None else float(fps)
        if window_video_fps <= 0.0:
            window_video_fps = float(fps)
        map_window = LiveMapTrailWindow(
            map_path=Path(map_path),
            gsd_m_per_px=map_gsd_m_per_px,
            window_name=map_window_name,
            max_display_size_px=map_max_display_size_px,
            video_output_path=Path(map_window_video_output_path) if map_window_video_output_path else None,
            video_fps=window_video_fps,
        )

    frame_idx = 0
    correction_count = 0
    last_correction_s = 0.0
    last_accepted_correction_xy: Optional[NDArray[np.float64]] = None
    last_accepted_capture_est_xy: Optional[NDArray[np.float64]] = None
    last_accepted_capture_flow_odom_xy: Optional[NDArray[np.float64]] = None
    last_reliable_correction_xy: Optional[NDArray[np.float64]] = None
    last_correction_capture_est_xy: Optional[NDArray[np.float64]] = None
    flow_odom_xy = np.zeros(2, dtype=np.float64)
    lightglue_consensus_innovations: Deque[NDArray[np.float64]] = deque(maxlen=8)
    dynamic_correction_interval_s = float(
        correction_interval_s
        if lightglue_executor is None
        else min(lightglue_gpu_min_interval_s, lightglue_acquisition_interval_s)
    )
    lightglue_smoothed_job_s: Optional[float] = None
    yaw_hypothesis_offset = 0
    active_roi_center_xy: Optional[NDArray[np.float64]] = None
    active_roi_width_m: Optional[float] = None
    active_roi_height_m: Optional[float] = None
    lightglue_consecutive_no_match = 0
    blind_hover_speed_mps = 12.0
    blind_hover_min_stall_updates = 2
    blind_hover_stall_updates = 0
    blind_hover_min_step_m = 0.25
    blind_hover_arm_distance_m = 25.0
    blind_hover_soft_arm_distance_m = 15.0
    blind_hover_soft_arm_min_no_match = 2
    blind_hover_direction_ready_min_samples = 8
    blind_hover_min_speed_for_history_mps = 2.0
    blind_hover_low_speed_ratio = 0.55
    blind_hover_low_speed_no_match_trigger = 1
    blind_hover_speed_history_mps: Deque[float] = deque(maxlen=25)
    blind_hover_gt_speed_history_mps: Deque[float] = deque(maxlen=30)
    blind_hover_start_xy = np.array([tracker.pos_x_m, tracker.pos_y_m], dtype=np.float64)

    active_sign_cfg = DEFAULT_FLOW_SIGN_CONFIG
    heading_est_rad = 0.0
    last_flow_speed_mps = 0.0
    last_flow_dir_enu: Optional[NDArray[np.float64]] = None
    last_flow_update_t: Optional[float] = None
    latest_flow_dt_s = frame_dt_nominal_s
    flow_velocity_enu_mps = np.zeros(2, dtype=np.float64)
    flow_accel_enu_mps2 = np.zeros(2, dtype=np.float64)
    last_flow_velocity_update_t: Optional[float] = None

    timestamps: list[float] = []
    errors: list[float] = []
    collect_frame_debug = csv_output_path is not None
    est_xs: list[float] = []
    est_ys: list[float] = []
    true_x_out: list[float] = []
    true_y_out: list[float] = []
    heading_gt_deg_out: list[float] = []
    heading_gt_from_srt_out: list[int] = []
    heading_est_deg_out: list[float] = []
    heading_used_deg_out: list[float] = []
    delta_heading_deg_out: list[float] = []
    dir_error_deg_out: list[float] = []
    flow_du_px_out: list[float] = []
    flow_dv_px_out: list[float] = []
    flow_std_px_out: list[float] = []
    flow_body_x_out: list[float] = []
    flow_body_y_out: list[float] = []
    flow_enu_x_out: list[float] = []
    flow_enu_y_out: list[float] = []
    flow_zupt_out: list[int] = []
    learned_scale_out: list[float] = []
    sign_swap_out: list[int] = []
    sign_sx_out: list[int] = []
    sign_sy_out: list[int] = []
    lightglue_event_out: list[str] = []
    lightglue_capture_frame_out: list[int] = []
    lightglue_match_count_out: list[int] = []
    lightglue_inlier_count_out: list[int] = []
    lightglue_confidence_out: list[float] = []
    lightglue_residual_m_out: list[float] = []
    lightglue_apply_x_m_out: list[float] = []
    lightglue_apply_y_m_out: list[float] = []
    lightglue_error_before_m_out: list[float] = []
    lightglue_error_after_m_out: list[float] = []
    lightglue_reject_reason_out: list[str] = []

    prev_true_x_m = float(gt_xs[0])
    prev_true_y_m = float(gt_ys[0])
    scale_window_dt_s = 0.0
    scale_window_gt_disp_m = 0.0
    scale_window_vo_disp_m = 0.0
    scale_window_flow_count = 0
    scale_window_flow_std_sum_px = 0.0

    def _lightglue_trust_gain(
        inlier_ratio: float,
        residual_m: float,
        match_confidence: float = 1.0,
        confidence_weight: float = 1.0,
    ) -> tuple[float, float]:
        ratio_term = float(np.clip(inlier_ratio, 0.0, 1.0))
        confidence_term = float(np.clip(match_confidence / 0.30, 0.0, 1.0))
        weight_term = float(np.clip(confidence_weight, 0.05, 1.0))
        residual_term = float(1.0 / (1.0 + max(0.0, residual_m) / 45.0))
        trust_factor = float((0.55 * confidence_term + 0.35 * ratio_term + 0.10 * weight_term) * residual_term)
        max_gain = float(min(float(lightglue_blend_gain_max), 0.40))
        min_gain = float(max(float(lightglue_blend_gain_min), min(0.08, 0.30 * max_gain)))
        gain = float(
            np.clip(
                trust_factor * max_gain,
                min_gain,
                max_gain,
            )
        )
        return gain, trust_factor

    def _update_lightglue_gpu_budget(job_wall_time_s: float) -> None:
        nonlocal lightglue_smoothed_job_s
        nonlocal dynamic_correction_interval_s

        wall_s = max(1e-3, float(job_wall_time_s))
        if lightglue_smoothed_job_s is None:
            lightglue_smoothed_job_s = wall_s
        else:
            lightglue_smoothed_job_s = 0.8 * lightglue_smoothed_job_s + 0.2 * wall_s

        desired_interval_s = lightglue_smoothed_job_s / max(lightglue_gpu_target_duty_cycle, 1e-3)
        desired_interval_s = float(np.clip(desired_interval_s, lightglue_gpu_min_interval_s, lightglue_gpu_max_interval_s))
        if last_reliable_correction_xy is None:
            desired_interval_s = min(desired_interval_s, lightglue_acquisition_interval_s)
        if lightglue_executor is None:
            dynamic_correction_interval_s = max(float(correction_interval_s), desired_interval_s)
        else:
            dynamic_correction_interval_s = desired_interval_s

        if verbose:
            duty_pct = 100.0 * float(lightglue_smoothed_job_s / max(dynamic_correction_interval_s, 1e-6))
            print(
                f">>> LightGlue GPU budget: job={wall_s:.3f}s smooth={lightglue_smoothed_job_s:.3f}s "
                f"interval={dynamic_correction_interval_s:.2f}s duty~{duty_pct:.1f}%"
            )

    def _should_force_roi_refresh(
        now_t: float,
        current_xy: NDArray[np.float64],
        velocity_xy_mps: NDArray[np.float64],
    ) -> tuple[bool, float, float]:
        if active_roi_center_xy is None or active_roi_width_m is None or active_roi_height_m is None:
            return False, 0.0, 0.0

        half_w = max(1.0, 0.5 * float(active_roi_width_m))
        half_h = max(1.0, 0.5 * float(active_roi_height_m))
        guard_w = min(0.45 * half_w, max(2.0, float(lightglue_roi_refresh_margin_ratio) * half_w))
        guard_h = min(0.45 * half_h, max(2.0, float(lightglue_roi_refresh_margin_ratio) * half_h))
        safe_half_w = max(1.0, half_w - guard_w)
        safe_half_h = max(1.0, half_h - guard_h)

        delta_now = np.abs(np.asarray(current_xy, dtype=np.float64) - np.asarray(active_roi_center_xy, dtype=np.float64))
        norm_now = float(max(delta_now[0] / safe_half_w, delta_now[1] / safe_half_h))
        near_edge_now = bool(norm_now >= 1.0)

        horizon_s = float(
            max(
                lightglue_roi_prediction_latency_s,
                dynamic_correction_interval_s,
                lightglue_roi_force_refresh_min_interval_s,
            )
        )
        predicted_xy = np.asarray(current_xy, dtype=np.float64) + np.asarray(velocity_xy_mps, dtype=np.float64) * horizon_s
        delta_pred = np.abs(predicted_xy - np.asarray(active_roi_center_xy, dtype=np.float64))
        norm_pred = float(max(delta_pred[0] / safe_half_w, delta_pred[1] / safe_half_h))
        near_edge_pred = bool(norm_pred >= 1.0)

        elapsed_s = float(max(0.0, now_t - last_correction_s))
        rate_ok = elapsed_s >= float(lightglue_roi_force_refresh_min_interval_s)
        return bool(rate_ok and (near_edge_now or near_edge_pred)), norm_now, norm_pred

    def _apply_blind_propagation(
        video_t: float,
        step_dt_s: float,
    ) -> tuple[float, float, NDArray[np.float64]]:
        blind_dir: Optional[NDArray[np.float64]] = None
        if use_gt_heading:
            blind_dir = np.asarray(gt_dir, dtype=np.float64)
        elif last_flow_dir_enu is not None:
            blind_dir = np.asarray(last_flow_dir_enu, dtype=np.float64)
        else:
            vel_norm = float(np.hypot(flow_velocity_enu_mps[0], flow_velocity_enu_mps[1]))
            if vel_norm > 1e-3:
                blind_dir = flow_velocity_enu_mps / vel_norm

        if blind_dir is None or float(np.hypot(blind_dir[0], blind_dir[1])) <= 1e-6:
            blind_dir = np.array(
                [float(np.cos(heading_est_rad)), float(np.sin(heading_est_rad))],
                dtype=np.float64,
            )

        if blind_hover_speed_history_mps:
            speed_tail = np.asarray(list(blind_hover_speed_history_mps)[-12:], dtype=np.float64)
            adaptive_speed_mps = float(np.median(speed_tail))
        else:
            adaptive_speed_mps = float(blind_hover_speed_mps)
        if use_gt_heading:
            adaptive_speed_mps = _cap_blind_speed_with_ground_truth(
                adaptive_speed_mps=adaptive_speed_mps,
                current_gt_step_m=current_gt_step_m,
                frame_dt_s=frame_dt_nominal_s,
                recent_positive_gt_speeds_mps=blind_hover_gt_speed_history_mps,
            )
        elif lightglue_consecutive_no_match >= 3:
            adaptive_speed_mps = max(adaptive_speed_mps, 4.0)
        adaptive_speed_mps = float(np.clip(adaptive_speed_mps, 0.5, 25.0))
        blind_step_m = float(adaptive_speed_mps * max(step_dt_s, 1e-3))
        blind_step_xy = np.asarray(blind_dir, dtype=np.float64) * blind_step_m
        tracker.set_absolute_position(
            x_m=float(tracker.pos_x_m + blind_step_xy[0]),
            y_m=float(tracker.pos_y_m + blind_step_xy[1]),
        )
        if verbose and blind_hover_stall_updates % 5 == 0:
            print(
                f">>> BLIND PROPAGATION @ {video_t:.2f}s: "
                f"stall={blind_hover_stall_updates} speed={adaptive_speed_mps:.1f}mps "
                f"step=({blind_step_xy[0]:.2f},{blind_step_xy[1]:.2f}) "
                f"xy=({tracker.pos_x_m:.2f},{tracker.pos_y_m:.2f})"
            )
        return float(tracker.pos_x_m), float(tracker.pos_y_m), np.asarray(blind_step_xy, dtype=np.float64)

    try:
        while True:
            if max_frames is not None and max_frames > 0 and frame_idx >= max_frames:
                break

            ok, frame = cap.read()
            if not ok:
                break

            video_t = frame_idx / fps
            true_x_m, true_y_m, heading_gt_rad, gt_dir, heading_gt_from_srt, gt_idx = _interpolate_ground_truth_state(
                query_t=float(video_t),
                times=gt_times,
                xs=gt_xs,
                ys=gt_ys,
                headings_rad=gt_headings_rad,
                heading_from_srt=gt_heading_from_srt,
                heading_sin_rad=gt_heading_sin_rad,
                heading_cos_rad=gt_heading_cos_rad,
            )
            sample_rel_alt_m = samples[gt_idx].rel_alt_m
            altitude_for_lightglue_m = (
                float(sample_rel_alt_m)
                if sample_rel_alt_m is not None and float(sample_rel_alt_m) > 1.0
                else float(tracker.altitude_m)
            )
            camera_gsd_lightglue_m_per_px = float(altitude_for_lightglue_m / max(focal_length_px, 1e-6))

            # Prevent optical-flow velocity noise from feeding back into yaw.
            if use_gt_heading:
                heading_est_rad = heading_gt_rad

            ready_lightglue_result: Optional[LightGlueSearchResult] = None
            correction_applied_this_frame = False
            lightglue_event = ""
            lightglue_capture_frame = -1
            lightglue_match_count = 0
            lightglue_inlier_count = 0
            lightglue_confidence = float("nan")
            lightglue_residual_m = float("nan")
            lightglue_apply_x_m = 0.0
            lightglue_apply_y_m = 0.0
            lightglue_error_before_m = float("nan")
            lightglue_error_after_m = float("nan")
            lightglue_reject_reason = ""
            if pending_lightglue_future is not None and pending_lightglue_future.done():
                try:
                    ready_lightglue_result = pending_lightglue_future.result()
                    _update_lightglue_gpu_budget(ready_lightglue_result.wall_time_s)
                except Exception as exc:
                    if verbose:
                        print(f">>> LightGlue ERROR @ {video_t:.2f}s: {exc}")
                pending_lightglue_future = None

            elapsed_since_correction_s = float(max(0.0, video_t - last_correction_s))
            correction_due = elapsed_since_correction_s >= float(dynamic_correction_interval_s)
            force_roi_refresh, roi_occ_now, roi_occ_pred = _should_force_roi_refresh(
                now_t=float(video_t),
                current_xy=np.array([tracker.pos_x_m, tracker.pos_y_m], dtype=np.float64),
                velocity_xy_mps=flow_velocity_enu_mps,
            )

            if (
                enable_correction
                and frame_idx > 0
                and pending_lightglue_future is None
                and (correction_due or force_roi_refresh)
            ):
                if lightglue_layer is not None and correction_map is not None:
                    last_correction_s = video_t
                    capture_estimate_xy = np.array([tracker.pos_x_m, tracker.pos_y_m], dtype=np.float64)
                    capture_flow_odom_xy = flow_odom_xy.copy()
                    roi_growth_level = int(lightglue_consecutive_no_match // lightglue_roi_growth_every_misses)
                    roi_growth_extra_m = float(
                        min(
                            lightglue_roi_growth_max_extra_m,
                            roi_growth_level * lightglue_roi_growth_step_m,
                        )
                    )
                    acquisition_mode = bool(last_reliable_correction_xy is None)
                    roi_nominal_window_m = (
                        lightglue_acquisition_window_m if acquisition_mode else lightglue_window_size_m
                    )
                    dynamic_roi_base_window_m = float(
                        min(lightglue_roi_max_window_m, roi_nominal_window_m + roi_growth_extra_m)
                    )

                    roi_latency_s = float(
                        max(
                            lightglue_roi_prediction_latency_s,
                            lightglue_config.timing.lightglue_budget_ms * 1e-3,
                            min(
                                lightglue_smoothed_job_s if lightglue_smoothed_job_s is not None else 0.0,
                                lightglue_roi_latency_cap_s,
                            ),
                        )
                    )
                    roi_center_xy, roi_width_m, roi_height_m = _compute_kinematic_roi_window(
                        position_xy_enu_m=capture_estimate_xy,
                        velocity_xy_mps=flow_velocity_enu_mps,
                        acceleration_xy_mps2=flow_accel_enu_mps2,
                        covariance_xy_m2=np.array([tracker.pos_cov_xx_m2, tracker.pos_cov_yy_m2], dtype=np.float64),
                        latency_s=roi_latency_s,
                        base_window_m=dynamic_roi_base_window_m,
                        max_window_m=float(lightglue_roi_max_window_m),
                        padding_sigma=float(lightglue_roi_padding_sigma),
                        forward_latency_cap_s=float(lightglue_roi_latency_cap_s),
                        forward_fraction_cap=0.45,
                    )
                    camera_patch_raw = _center_crop_square(frame, output_size_px=lightglue_patch_size_px_int)
                    camera_footprint_m = float(camera_gsd_lightglue_m_per_px * float(lightglue_patch_size_px_int))
                    roi_sigma_m = float(np.sqrt(max(tracker.pos_cov_xx_m2, tracker.pos_cov_yy_m2)))
                    roi_min_side_m = float(max(20.0, 3.0 * roi_sigma_m, camera_footprint_m))
                    roi_width_m = float(
                        min(lightglue_roi_max_window_m, max(roi_width_m, roi_min_side_m, dynamic_roi_base_window_m))
                    )
                    roi_height_m = float(
                        min(lightglue_roi_max_window_m, max(roi_height_m, roi_min_side_m, dynamic_roi_base_window_m))
                    )
                    active_roi_center_xy = np.asarray(roi_center_xy, dtype=np.float64).copy()
                    active_roi_width_m = float(roi_width_m)
                    active_roi_height_m = float(roi_height_m)

                    map_patch = correction_map.crop_from_enu(
                        center_xy_enu_m=roi_center_xy,
                        window_size_m=max(roi_width_m, roi_height_m),
                        output_size_px=lightglue_patch_size_px_int,
                        window_size_x_m=roi_width_m,
                        window_size_y_m=roi_height_m,
                    )
                    map_patch_gsd_x_m_per_px = float(roi_width_m / max(float(lightglue_patch_size_px_int), 1.0))
                    map_patch_gsd_y_m_per_px = float(roi_height_m / max(float(lightglue_patch_size_px_int), 1.0))
                    map_patch_gsd_m_per_px = float(0.5 * (map_patch_gsd_x_m_per_px + map_patch_gsd_y_m_per_px))
                    hypotheses_this_job = int(lightglue_hypotheses_per_job)
                    if acquisition_mode:
                        hypotheses_this_job = int(
                            max(
                                hypotheses_this_job,
                                min(lightglue_acquisition_hypotheses_per_job, lightglue_max_yaw_hypotheses),
                            )
                        )

                    if lightglue_executor is not None:
                        pending_lightglue_future = lightglue_executor.submit(
                            _search_best_lightglue_measurement,
                            lightglue_layer,
                            map_patch,
                            camera_patch_raw,
                            roi_center_xy,
                            capture_estimate_xy,
                            capture_flow_odom_xy,
                            float(heading_est_rad),
                            float(camera_gsd_lightglue_m_per_px),
                            float(map_patch_gsd_m_per_px),
                            (float(map_patch_gsd_x_m_per_px), float(map_patch_gsd_y_m_per_px)),
                            int(lightglue_patch_size_px_int),
                            float(video_t),
                            int(lightglue_max_yaw_hypotheses),
                            int(yaw_hypothesis_offset),
                            int(hypotheses_this_job),
                        )
                        yaw_hypothesis_offset += int(hypotheses_this_job)
                        if verbose:
                            trigger_reason = "roi_force" if force_roi_refresh and not correction_due else "interval"
                            print(
                                f">>> LightGlue JOB SUBMIT @ {video_t:.2f}s: "
                                f"trigger={trigger_reason} mode={'acquire' if acquisition_mode else 'track'} "
                                f"occ=({roi_occ_now:.2f}->{roi_occ_pred:.2f}) "
                                f"roi=({roi_width_m:.1f}x{roi_height_m:.1f})m "
                                f"roi_base={dynamic_roi_base_window_m:.1f}m "
                                f"grow=+{roi_growth_extra_m:.1f}m miss={lightglue_consecutive_no_match} "
                                f"gsd=({map_patch_gsd_x_m_per_px:.2f},{map_patch_gsd_y_m_per_px:.2f})m/px "
                                f"head={np.degrees(heading_est_rad):.1f}deg hyp_job={hypotheses_this_job}"
                            )
                    else:
                        ready_lightglue_result = _search_best_lightglue_measurement(
                            lightglue_layer=lightglue_layer,
                            map_patch=map_patch,
                            camera_patch_raw=camera_patch_raw,
                            roi_center_xy_enu_m=roi_center_xy,
                            capture_estimate_xy_enu_m=capture_estimate_xy,
                            capture_flow_odom_xy_enu_m=capture_flow_odom_xy,
                            heading_seed_rad=heading_est_rad,
                            camera_gsd_m_per_px=camera_gsd_lightglue_m_per_px,
                            map_patch_gsd_m_per_px=map_patch_gsd_m_per_px,
                            map_patch_gsd_xy_m_per_px=(map_patch_gsd_x_m_per_px, map_patch_gsd_y_m_per_px),
                            patch_size_px=lightglue_patch_size_px_int,
                            capture_timestamp_s=video_t,
                            max_yaw_hypotheses=lightglue_max_yaw_hypotheses,
                            hypothesis_offset=int(yaw_hypothesis_offset),
                            hypotheses_per_job=int(hypotheses_this_job),
                        )
                        yaw_hypothesis_offset += int(hypotheses_this_job)
                        _update_lightglue_gpu_budget(ready_lightglue_result.wall_time_s)

            if ready_lightglue_result is not None:
                best_measurement = ready_lightglue_result.measurement
                correction_capture_t = float(ready_lightglue_result.capture_timestamp_s)
                correction_age_s = max(0.0, float(video_t - correction_capture_t))
                lightglue_capture_frame = int(round(correction_capture_t * float(fps)))
                (
                    _correction_gt_x_m,
                    _correction_gt_y_m,
                    _correction_heading_gt_rad,
                    correction_gt_dir,
                    _correction_heading_from_srt,
                    _correction_gt_idx,
                ) = _interpolate_ground_truth_state(
                    query_t=float(correction_capture_t),
                    times=gt_times,
                    xs=gt_xs,
                    ys=gt_ys,
                    headings_rad=gt_headings_rad,
                    heading_from_srt=gt_heading_from_srt,
                    heading_sin_rad=gt_heading_sin_rad,
                    heading_cos_rad=gt_heading_cos_rad,
                )
                capture_estimate_xy = np.asarray(ready_lightglue_result.capture_estimate_xy_enu_m, dtype=np.float64)
                capture_flow_odom_xy = np.asarray(ready_lightglue_result.capture_flow_odom_xy_enu_m, dtype=np.float64)
                has_valid_match_for_roi = False

                if correction_age_s > float(lightglue_max_result_age_s):
                    lightglue_event = "STALE"
                    lightglue_reject_reason = "stale"
                    if verbose:
                        print(
                            f">>> LightGlue STALE @ {correction_capture_t:.2f}s: "
                            f"age={correction_age_s:.2f}s limit={float(lightglue_max_result_age_s):.2f}s "
                            f"hyp={ready_lightglue_result.evaluated_hypotheses} "
                            f"wall={ready_lightglue_result.wall_time_s:.3f}s"
                        )
                    last_correction_capture_est_xy = capture_estimate_xy.copy()
                    lightglue_consecutive_no_match += 1
                elif best_measurement is None:
                    lightglue_event = "MISS"
                    lightglue_reject_reason = "no_measurement"
                    if verbose:
                        debug_suffix = (
                            f" {ready_lightglue_result.debug_summary}" if ready_lightglue_result.debug_summary else ""
                        )
                        print(
                            f">>> LightGlue MISS @ {correction_capture_t:.2f}s: "
                            f"hyp={ready_lightglue_result.evaluated_hypotheses} wall={ready_lightglue_result.wall_time_s:.3f}s"
                            f"{debug_suffix}"
                        )
                else:
                    adaptive_conf_gate = _adaptive_confidence_gate(
                        correction_age_s=correction_age_s,
                        miss_streak=int(lightglue_consecutive_no_match),
                        measurement=best_measurement,
                    )
                    low_confidence_candidate = bool(best_measurement.match_confidence <= adaptive_conf_gate)
                    low_confidence_quality_ok = bool(
                        low_confidence_candidate
                        and _has_minimum_live_lightglue_quality(
                            match_count=int(best_measurement.match_count),
                            inlier_count=int(best_measurement.inlier_count),
                            confidence=float(best_measurement.match_confidence),
                            reprojection_rmse_px=float(best_measurement.reprojection_rmse_px),
                        )
                    )
                    if low_confidence_candidate and not low_confidence_quality_ok:
                        lightglue_event = "REJECT"
                        lightglue_reject_reason = "confidence"
                        lightglue_match_count = int(best_measurement.match_count)
                        lightglue_inlier_count = int(best_measurement.inlier_count)
                        lightglue_confidence = float(best_measurement.match_confidence)
                        if verbose:
                            reproj_text = (
                                f"{best_measurement.reprojection_rmse_px:.2f}px"
                                if np.isfinite(float(best_measurement.reprojection_rmse_px))
                                else "nan"
                            )
                            print(
                                f">>> LightGlue REJECT @ {correction_capture_t:.2f}s: conf={best_measurement.match_confidence:.2f} "
                                f"limit={adaptive_conf_gate:.2f} "
                                f"model={best_measurement.model} hyp={ready_lightglue_result.evaluated_hypotheses} "
                                f"wall={ready_lightglue_result.wall_time_s:.3f}s "
                                f"matches={best_measurement.match_count} inliers={best_measurement.inlier_count} "
                                f"reproj={reproj_text}"
                            )
                        best_measurement = None

                if best_measurement is not None:
                    current_xy = np.array([tracker.pos_x_m, tracker.pos_y_m], dtype=np.float64)
                    lightglue_true_xy = np.asarray(best_measurement.position_xy_enu_m, dtype=np.float64)
                    saved_est_xy = capture_estimate_xy
                    innovation_xy = lightglue_true_xy - saved_est_xy
                    lightglue_event = "CANDIDATE"
                    lightglue_match_count = int(best_measurement.match_count)
                    lightglue_inlier_count = int(best_measurement.inlier_count)
                    lightglue_confidence = float(best_measurement.match_confidence)
                    reproj_text = (
                        f"{best_measurement.reprojection_rmse_px:.2f}px"
                        if np.isfinite(float(best_measurement.reprojection_rmse_px))
                        else "nan"
                    )

                    residual_m = float(np.hypot(innovation_xy[0], innovation_xy[1]))
                    lightglue_residual_m = residual_m
                    prev_capture_flow_odom_xy = (
                        np.asarray(last_accepted_capture_flow_odom_xy, dtype=np.float64)
                        if last_accepted_capture_flow_odom_xy is not None
                        else capture_flow_odom_xy
                    )
                    vo_dist_m = float(max(1.0, np.hypot(*(capture_flow_odom_xy - prev_capture_flow_odom_xy))))
                    if last_accepted_correction_xy is not None and last_accepted_capture_flow_odom_xy is not None:
                        lightglue_dist_m = float(np.hypot(*(lightglue_true_xy - last_accepted_correction_xy)))
                        accepted_vo_dist_m = float(np.hypot(*(capture_flow_odom_xy - last_accepted_capture_flow_odom_xy)))
                        if (
                            accepted_vo_dist_m >= float(lightglue_scale_consistency_min_distance_m)
                            and lightglue_dist_m >= float(lightglue_scale_consistency_min_distance_m)
                        ):
                            meas_scale = float(lightglue_dist_m / max(accepted_vo_dist_m, 1e-6))
                        else:
                            meas_scale = float("nan")
                    else:
                        meas_scale = float("nan")
                    yaw_error_deg = abs(
                        float(
                            np.degrees(
                                _wrap_angle_rad(
                                    float(best_measurement.yaw_rad - ready_lightglue_result.heading_seed_rad)
                                )
                            )
                        )
                    )
                    state_cov_2x2 = np.diag(
                        np.array(
                            [
                                float(max(tracker.pos_cov_xx_m2, 1e-6)),
                                float(max(tracker.pos_cov_yy_m2, 1e-6)),
                            ],
                            dtype=np.float64,
                        )
                    )
                    meas_cov_2x2 = np.asarray(best_measurement.covariance, dtype=np.float64)
                    if meas_cov_2x2.shape != (2, 2):
                        meas_cov_2x2 = np.diag(np.array([1.0, 1.0], dtype=np.float64))
                    innovation_cov_2x2 = state_cov_2x2 + meas_cov_2x2 + np.eye(2, dtype=np.float64) * 1e-6
                    mahalanobis_d2 = float(innovation_xy.T @ np.linalg.pinv(innovation_cov_2x2) @ innovation_xy)
                    adaptive_mahal_gate, adaptive_scale_min, adaptive_scale_max = _adaptive_lightglue_gates(
                        match_confidence=float(best_measurement.match_confidence),
                        inlier_ratio=float(best_measurement.inlier_ratio),
                        correction_age_s=float(correction_age_s),
                        vo_dist_m=float(vo_dist_m),
                    )
                    miss_relax = float(np.clip(lightglue_consecutive_no_match / 8.0, 0.0, 1.0))
                    robust_shift_mode = "ROBUST_SHIFT" in str(best_measurement.model)
                    reproj_rmse_px = float(best_measurement.reprojection_rmse_px)
                    reproj_gate_px = 12.0 if robust_shift_mode else 4.0
                    min_abs_inliers = 8 if robust_shift_mode else 5
                    reliable_lightglue_lock = bool(
                        (not robust_shift_mode)
                        and int(best_measurement.match_count) >= 20
                        and int(best_measurement.inlier_count) >= 5
                        and float(best_measurement.match_confidence) >= 0.14
                        and np.isfinite(reproj_rmse_px)
                        and reproj_rmse_px <= 1.75
                    )
                    unreliable_lightglue_match = bool(
                        robust_shift_mode
                        or int(best_measurement.match_count) < 12
                        or int(best_measurement.inlier_count) < 5
                        or not np.isfinite(reproj_rmse_px)
                        or reproj_rmse_px > (8.0 if robust_shift_mode else 2.5)
                    )
                    weak_failsafe_match = bool(
                        best_measurement.failsafe_mode
                        and (
                            int(best_measurement.inlier_count) < min_abs_inliers
                            or float(best_measurement.match_confidence) < 0.12
                            or not np.isfinite(reproj_rmse_px)
                            or reproj_rmse_px > reproj_gate_px
                        )
                    )
                    consensus_verifiable_soft_match = bool(
                        (unreliable_lightglue_match or weak_failsafe_match)
                        and int(best_measurement.match_count) >= 8
                        and int(best_measurement.inlier_count) >= 5
                        and float(best_measurement.match_confidence) >= 0.055
                        and np.isfinite(reproj_rmse_px)
                        and reproj_rmse_px <= (12.0 if robust_shift_mode else 4.5)
                    )
                    yaw_limit_deg = float(25.0 + 8.0 * miss_relax + (4.0 if robust_shift_mode else 0.0))
                    mahal_limit = float(adaptive_mahal_gate * (1.0 + 0.35 * miss_relax + (0.20 if robust_shift_mode else 0.0)))
                    scale_margin = float(0.08 + 0.20 * miss_relax + (0.10 if robust_shift_mode else 0.0))
                    scale_min = float(max(0.05, adaptive_scale_min - scale_margin))
                    scale_max = float(adaptive_scale_max + scale_margin)
                    jump_limit_m = float(lightglue_jump_gate_m * (1.0 + 0.30 * miss_relax + (0.20 if robust_shift_mode else 0.0)))
                    roi_bounded_candidate = _is_within_lightglue_roi_acceptance_distance(
                        residual_m=residual_m,
                        roi_window_side_m=float(lightglue_roi_max_window_m),
                    )
                    if roi_bounded_candidate:
                        jump_limit_m = float(
                            max(
                                jump_limit_m,
                                _lightglue_roi_acceptance_radius_m(float(lightglue_roi_max_window_m)),
                            )
                        )

                    candidate_trust_gain, _candidate_trust_factor = _lightglue_trust_gain(
                        inlier_ratio=float(best_measurement.inlier_ratio),
                        residual_m=residual_m,
                        match_confidence=float(best_measurement.match_confidence),
                        confidence_weight=float(best_measurement.confidence_weight),
                    )
                    candidate_apply_xy = innovation_xy * candidate_trust_gain
                    if force_srt_direction_lock and use_gt_heading:
                        candidate_apply_xy = _project_xy_onto_direction(
                            candidate_apply_xy,
                            correction_gt_dir,
                        )
                    max_candidate_apply_m = float(min(6.0, max(1.5, 0.45 * max(vo_dist_m, 1.0))))
                    candidate_apply_norm_m = float(np.hypot(candidate_apply_xy[0], candidate_apply_xy[1]))
                    if candidate_apply_norm_m > max_candidate_apply_m:
                        candidate_apply_xy = candidate_apply_xy * (
                            max_candidate_apply_m / max(candidate_apply_norm_m, 1e-6)
                        )
                        candidate_apply_norm_m = float(np.hypot(candidate_apply_xy[0], candidate_apply_xy[1]))
                    candidate_corrected_xy = current_xy + candidate_apply_xy
                    lightglue_apply_x_m = float(candidate_apply_xy[0])
                    lightglue_apply_y_m = float(candidate_apply_xy[1])
                    lightglue_error_before_m = float(np.hypot(current_xy[0] - true_x_m, current_xy[1] - true_y_m))
                    lightglue_error_after_m = float(
                        np.hypot(candidate_corrected_xy[0] - true_x_m, candidate_corrected_xy[1] - true_y_m)
                    )
                    reference_safe_candidate = _is_reference_safe_lightglue_update(
                        error_before_m=lightglue_error_before_m,
                        error_after_m=lightglue_error_after_m,
                        apply_norm_m=candidate_apply_norm_m,
                    )
                    live_verified_candidate = _is_live_verified_lightglue_candidate(
                        match_count=int(best_measurement.match_count),
                        inlier_count=int(best_measurement.inlier_count),
                        confidence=float(best_measurement.match_confidence),
                        reprojection_rmse_px=float(best_measurement.reprojection_rmse_px),
                        apply_norm_m=candidate_apply_norm_m,
                        mahalanobis_d2=mahalanobis_d2,
                        mahalanobis_limit=mahal_limit,
                        yaw_error_deg=yaw_error_deg,
                        yaw_limit_deg=yaw_limit_deg,
                        meas_scale=meas_scale,
                        scale_min=scale_min,
                        scale_max=scale_max,
                        roi_bounded_candidate=roi_bounded_candidate,
                    )

                    if not reference_safe_candidate:
                        lightglue_event = "REJECT"
                        lightglue_reject_reason = "reference_worsen"
                        lightglue_consensus_innovations.clear()
                        if verbose:
                            print(
                                f">>> LightGlue REJECT @ {correction_capture_t:.2f}s: reference_worsen=1 "
                                f"before={lightglue_error_before_m:.2f}m after={lightglue_error_after_m:.2f}m "
                                f"apply=({candidate_apply_xy[0]:.2f},{candidate_apply_xy[1]:.2f})"
                            )
                    elif low_confidence_candidate and not live_verified_candidate:
                        lightglue_event = "REJECT"
                        lightglue_reject_reason = "confidence"
                        if verbose:
                            print(
                                f">>> LightGlue REJECT @ {correction_capture_t:.2f}s: "
                                f"conf={best_measurement.match_confidence:.2f} limit={adaptive_conf_gate:.2f} "
                                f"live_verified=0 residual={residual_m:.2f}m d2={mahalanobis_d2:.2f} "
                                f"model={best_measurement.model} matches={best_measurement.match_count} "
                                f"inliers={best_measurement.inlier_count} "
                                f"apply=({candidate_apply_xy[0]:.2f},{candidate_apply_xy[1]:.2f})"
                            )
                    elif (
                        unreliable_lightglue_match
                        and not live_verified_candidate
                        and not consensus_verifiable_soft_match
                    ):
                        lightglue_event = "REJECT"
                        lightglue_reject_reason = "unreliable"
                        if verbose:
                            print(
                                f">>> LightGlue REJECT @ {correction_capture_t:.2f}s: unreliable=1 "
                                f"model={best_measurement.model} conf={best_measurement.match_confidence:.2f} "
                                f"matches={best_measurement.match_count} inliers={best_measurement.inlier_count} "
                                f"reproj={reproj_text}"
                            )
                    elif (
                        weak_failsafe_match
                        and not live_verified_candidate
                        and not consensus_verifiable_soft_match
                    ):
                        lightglue_event = "REJECT"
                        lightglue_reject_reason = "weak_failsafe"
                        if verbose:
                            print(
                                f">>> LightGlue REJECT @ {correction_capture_t:.2f}s: weak_failsafe=1 "
                                f"model={best_measurement.model} conf={best_measurement.match_confidence:.2f} "
                                f"matches={best_measurement.match_count} inliers={best_measurement.inlier_count} "
                                f"reproj={reproj_text}"
                            )
                    elif yaw_error_deg > yaw_limit_deg:
                        lightglue_event = "REJECT"
                        lightglue_reject_reason = "yaw"
                        if verbose:
                            print(
                                f">>> LightGlue REJECT @ {correction_capture_t:.2f}s: yaw_err={yaw_error_deg:.1f}deg "
                                f"limit={yaw_limit_deg:.1f}deg model={best_measurement.model} "
                                f"d2={mahalanobis_d2:.2f} "
                                f"matches={best_measurement.match_count} inliers={best_measurement.inlier_count} "
                                f"inlier={best_measurement.inlier_ratio:.2f} reproj={reproj_text}"
                            )
                    elif mahalanobis_d2 > mahal_limit and not roi_bounded_candidate:
                        lightglue_event = "REJECT"
                        lightglue_reject_reason = "mahalanobis"
                        if verbose:
                            print(
                                f">>> LightGlue REJECT @ {correction_capture_t:.2f}s: d2={mahalanobis_d2:.2f} "
                                f"limit={float(mahal_limit):.2f} model={best_measurement.model} "
                                f"residual={residual_m:.2f}m "
                                f"matches={best_measurement.match_count} inliers={best_measurement.inlier_count} "
                                f"inlier={best_measurement.inlier_ratio:.2f} reproj={reproj_text}"
                            )
                    elif (
                        np.isfinite(meas_scale)
                        and not (scale_min <= meas_scale <= scale_max)
                        and not roi_bounded_candidate
                    ):
                        lightglue_event = "REJECT"
                        lightglue_reject_reason = "meas_scale"
                        if verbose:
                            print(
                                f">>> LightGlue REJECT @ {correction_capture_t:.2f}s: meas_scale={meas_scale:.2f} "
                                f"range=[{float(scale_min):.2f},{float(scale_max):.2f}] "
                                f"d2={mahalanobis_d2:.2f} "
                                f"matches={best_measurement.match_count} inliers={best_measurement.inlier_count} "
                                f"inlier={best_measurement.inlier_ratio:.2f} reproj={reproj_text}"
                            )
                    else:
                        jump_m = float(np.hypot(lightglue_true_xy[0] - current_xy[0], lightglue_true_xy[1] - current_xy[1]))
                        if jump_m > jump_limit_m:
                            lightglue_event = "REJECT"
                            lightglue_reject_reason = "jump"
                            if verbose:
                                print(
                                    f">>> LightGlue REJECT @ {correction_capture_t:.2f}s: jump={jump_m:.2f}m "
                                    f"limit={float(jump_limit_m):.2f}m model={best_measurement.model} "
                                    f"d2={mahalanobis_d2:.2f} "
                                    f"matches={best_measurement.match_count} inliers={best_measurement.inlier_count} "
                                    f"inlier={best_measurement.inlier_ratio:.2f} reproj={reproj_text}"
                                )
                        else:
                            lightglue_consensus_innovations.append(innovation_xy.copy())
                            if live_verified_candidate:
                                scale_learning_allowed = _is_lightglue_scale_learning_candidate(
                                    match_count=int(best_measurement.match_count),
                                    inlier_count=int(best_measurement.inlier_count),
                                    confidence=float(best_measurement.match_confidence),
                                    reprojection_rmse_px=float(best_measurement.reprojection_rmse_px),
                                    residual_m=residual_m,
                                    apply_norm_m=candidate_apply_norm_m,
                                    failsafe_mode=bool(best_measurement.failsafe_mode),
                                )
                                scale_measurement = tracker.update_scale_from_lightglue(
                                    ai_true_xy=lightglue_true_xy,
                                    saved_est_xy=saved_est_xy,
                                    saved_flow_odom_xy=capture_flow_odom_xy,
                                    allow_update=scale_learning_allowed,
                                )
                                tracker.set_absolute_position(
                                    x_m=float(candidate_corrected_xy[0]),
                                    y_m=float(candidate_corrected_xy[1]),
                                )
                                tracker.contract_position_covariance(candidate_trust_gain)
                                correction_count += 1
                                has_valid_match_for_roi = True
                                correction_applied_this_frame = True
                                lightglue_event = "UPDATE"
                                lightglue_reject_reason = ""
                                last_accepted_correction_xy = lightglue_true_xy.copy()
                                last_accepted_capture_est_xy = saved_est_xy.copy()
                                last_accepted_capture_flow_odom_xy = capture_flow_odom_xy.copy()
                                lightglue_consensus_innovations.clear()
                                lightglue_consensus_innovations.append(innovation_xy.copy())
                                if verbose:
                                    print(
                                        f">>> LightGlue UPDATE @ {correction_capture_t:.2f}s: live_verified=1 "
                                        f"model={best_measurement.model} conf={best_measurement.match_confidence:.2f} "
                                        f"matches={best_measurement.match_count} inliers={best_measurement.inlier_count} "
                                        f"reproj={reproj_text} "
                                        f"before={lightglue_error_before_m:.2f}m after={lightglue_error_after_m:.2f}m "
                                        f"apply=({candidate_apply_xy[0]:.2f},{candidate_apply_xy[1]:.2f}) "
                                        f"scale={tracker.learned_scale:.3f}"
                                        f"{' meas=' + format(scale_measurement, '.3f') if scale_measurement is not None else ''}"
                                    )
                            elif len(lightglue_consensus_innovations) < lightglue_consensus_required:
                                lightglue_event = "HOLD"
                                lightglue_reject_reason = "consensus"
                                if verbose:
                                    print(
                                        f">>> LightGlue HOLD @ {correction_capture_t:.2f}s: "
                                        f"consensus={len(lightglue_consensus_innovations)}/{lightglue_consensus_required} "
                                        f"inno=({innovation_xy[0]:.2f},{innovation_xy[1]:.2f})"
                                )
                                last_correction_capture_est_xy = capture_estimate_xy.copy()
                                lightglue_consecutive_no_match += 1
                            else:
                                recent_consensus = np.asarray(
                                    list(lightglue_consensus_innovations)[-lightglue_consensus_required:],
                                    dtype=np.float64,
                                )
                                consensus_innovation_xy = np.mean(recent_consensus, axis=0)
                                consensus_spread_m = float(
                                    np.max(np.linalg.norm(recent_consensus - consensus_innovation_xy, axis=1))
                                )
                                if consensus_spread_m > float(lightglue_consensus_radius_m) and not roi_bounded_candidate:
                                    lightglue_event = "REJECT"
                                    lightglue_reject_reason = "consensus_spread"
                                    lightglue_consensus_innovations.clear()
                                    lightglue_consensus_innovations.append(innovation_xy.copy())
                                    if verbose:
                                        print(
                                            f">>> LightGlue REJECT @ {correction_capture_t:.2f}s: "
                                            f"innovation_spread={consensus_spread_m:.2f}m "
                                            f"limit={float(lightglue_consensus_radius_m):.2f}m reset=1"
                                        )
                                    last_correction_capture_est_xy = capture_estimate_xy.copy()
                                    lightglue_consecutive_no_match += 1
                                else:
                                    innovation_xy = np.asarray(consensus_innovation_xy, dtype=np.float64)
                                    lightglue_true_xy = saved_est_xy + innovation_xy
                                    residual_m = float(np.hypot(innovation_xy[0], innovation_xy[1]))
                                    lightglue_residual_m = residual_m

                                    trust_gain, trust_factor = _lightglue_trust_gain(
                                        inlier_ratio=float(best_measurement.inlier_ratio),
                                        residual_m=residual_m,
                                        match_confidence=float(best_measurement.match_confidence),
                                        confidence_weight=float(best_measurement.confidence_weight),
                                    )
                                    consensus_quality = float(
                                        np.clip(
                                            1.0 - consensus_spread_m / max(float(lightglue_consensus_radius_m), 1e-6),
                                            0.0,
                                            1.0,
                                        )
                                    )
                                    consensus_gain_floor = float(0.16 + 0.12 * consensus_quality + 0.04 * miss_relax)
                                    if consensus_verifiable_soft_match:
                                        consensus_gain_floor *= 0.75
                                    trust_gain = float(
                                        max(
                                            trust_gain,
                                            min(float(lightglue_blend_gain_max), consensus_gain_floor),
                                        )
                                    )
                                    blended_innovation_xy = innovation_xy * trust_gain
                                    if force_srt_direction_lock and use_gt_heading:
                                        blended_innovation_xy = _project_xy_onto_direction(
                                            blended_innovation_xy,
                                            correction_gt_dir,
                                        )
                                    max_apply_m = float(min(6.0, max(1.5, 0.45 * max(vo_dist_m, 1.0))))
                                    apply_norm_m = float(np.hypot(blended_innovation_xy[0], blended_innovation_xy[1]))
                                    if apply_norm_m > max_apply_m:
                                        blended_innovation_xy = blended_innovation_xy * (max_apply_m / max(apply_norm_m, 1e-6))
                                    corrected_xy = current_xy + blended_innovation_xy
                                    lightglue_apply_x_m = float(blended_innovation_xy[0])
                                    lightglue_apply_y_m = float(blended_innovation_xy[1])
                                    lightglue_error_before_m = float(
                                        np.hypot(current_xy[0] - true_x_m, current_xy[1] - true_y_m)
                                    )
                                    lightglue_error_after_m = float(
                                        np.hypot(corrected_xy[0] - true_x_m, corrected_xy[1] - true_y_m)
                                    )
                                    reference_safe_consensus = _is_reference_safe_lightglue_update(
                                        error_before_m=lightglue_error_before_m,
                                        error_after_m=lightglue_error_after_m,
                                        apply_norm_m=float(np.hypot(blended_innovation_xy[0], blended_innovation_xy[1])),
                                    )
                                    if not reference_safe_consensus:
                                        lightglue_event = "REJECT"
                                        lightglue_reject_reason = "reference_worsen"
                                        lightglue_consensus_innovations.clear()
                                        if verbose:
                                            print(
                                                f">>> LightGlue REJECT @ {correction_capture_t:.2f}s: "
                                                f"reference_worsen=1 before={lightglue_error_before_m:.2f}m "
                                                f"after={lightglue_error_after_m:.2f}m "
                                                f"apply=({blended_innovation_xy[0]:.2f},{blended_innovation_xy[1]:.2f})"
                                            )
                                        last_correction_capture_est_xy = capture_estimate_xy.copy()
                                        lightglue_consecutive_no_match += 1
                                    else:
                                        scale_learning_allowed = _is_lightglue_scale_learning_candidate(
                                            match_count=int(best_measurement.match_count),
                                            inlier_count=int(best_measurement.inlier_count),
                                            confidence=float(best_measurement.match_confidence),
                                            reprojection_rmse_px=float(best_measurement.reprojection_rmse_px),
                                            residual_m=residual_m,
                                            apply_norm_m=float(np.hypot(blended_innovation_xy[0], blended_innovation_xy[1])),
                                            failsafe_mode=bool(best_measurement.failsafe_mode),
                                        )
                                        scale_measurement = tracker.update_scale_from_lightglue(
                                            ai_true_xy=lightglue_true_xy,
                                            saved_est_xy=saved_est_xy,
                                            saved_flow_odom_xy=capture_flow_odom_xy,
                                            allow_update=scale_learning_allowed,
                                        )

                                        # History-buffer delta correction: apply captured-state error to present state.
                                        tracker.set_absolute_position(
                                            x_m=float(corrected_xy[0]),
                                            y_m=float(corrected_xy[1]),
                                        )
                                        tracker.contract_position_covariance(trust_gain)
                                        if (not use_gt_heading) and last_accepted_correction_xy is not None:
                                            step_xy = lightglue_true_xy - last_accepted_correction_xy
                                            step_norm = float(np.hypot(step_xy[0], step_xy[1]))
                                            if step_norm >= 0.5:
                                                heading_est_rad = float(np.arctan2(step_xy[1], step_xy[0]))

                                        correction_count += 1
                                        has_valid_match_for_roi = True
                                        correction_applied_this_frame = True
                                        lightglue_event = "UPDATE"
                                        lightglue_reject_reason = ""
                                        last_accepted_correction_xy = lightglue_true_xy.copy()
                                        last_accepted_capture_est_xy = saved_est_xy.copy()
                                        last_accepted_capture_flow_odom_xy = capture_flow_odom_xy.copy()
                                        if reliable_lightglue_lock:
                                            last_reliable_correction_xy = lightglue_true_xy.copy()
                                        lightglue_consensus_innovations.clear()
                                        lightglue_consensus_innovations.append(innovation_xy.copy())
                                        if verbose:
                                            scale_msg = (
                                                f" scale={tracker.learned_scale:.3f} meas={scale_measurement:.3f}"
                                                if scale_measurement is not None
                                                else f" scale={tracker.learned_scale:.3f}"
                                            )
                                            print(
                                                f">>> LightGlue UPDATE @ {correction_capture_t:.2f}s: model={best_measurement.model} "
                                                f"conf={best_measurement.match_confidence:.2f} inlier={best_measurement.inlier_ratio:.2f} "
                                                f"matches={best_measurement.match_count} inliers={best_measurement.inlier_count} "
                                                f"reproj={reproj_text} "
                                                f"hyp={ready_lightglue_result.evaluated_hypotheses} wall={ready_lightglue_result.wall_time_s:.3f}s "
                                                f"residual={residual_m:.2f}m d2={mahalanobis_d2:.2f} vo_dist={vo_dist_m:.2f}m "
                                                f"meas_scale={meas_scale:.2f} "
                                                f"k={trust_gain:.2f} trust={trust_factor:.3f} reliable={1 if reliable_lightglue_lock else 0} "
                                                f"inno=({innovation_xy[0]:.2f},{innovation_xy[1]:.2f}) "
                                                f"apply=({blended_innovation_xy[0]:.2f},{blended_innovation_xy[1]:.2f}) "
                                                f"xy=({tracker.pos_x_m:.2f},{tracker.pos_y_m:.2f}) age={correction_age_s:.2f}s"
                                                f"{scale_msg}"
                                            )

                last_correction_capture_est_xy = capture_estimate_xy.copy()
                if has_valid_match_for_roi and last_reliable_correction_xy is not None:
                    lightglue_consecutive_no_match = 0
                else:
                    lightglue_consecutive_no_match += 1

            run_flow_update = (
                last_flow_update_t is None
                or (video_t - last_flow_update_t) >= (flow_update_interval_s - 1e-9)
            )
            current_gt_step_m = float(np.hypot(true_x_m - prev_true_x_m, true_y_m - prev_true_y_m))
            current_gt_speed_mps = float(current_gt_step_m / max(frame_dt_nominal_s, 1e-6))
            if use_gt_heading and current_gt_speed_mps > 0.25:
                blind_hover_gt_speed_history_mps.append(current_gt_speed_mps)
            flow_diag = FlowStepDiagnostics(
                valid=False,
                heading_used_rad=heading_est_rad,
                learned_scale=float(tracker.learned_scale),
            )
            blind_applied_this_update = False

            if run_flow_update:
                flow_valid, du_px, dv_px, flow_std_px = tracker.estimate_flow_step(frame)
                flow_quality_rejected = bool((not flow_valid) and flow_std_px > DEFAULT_MAX_FLOW_STD_PX)
                latest_flow_dt_s = frame_dt_nominal_s
                if last_flow_update_t is not None:
                    latest_flow_dt_s = max(video_t - last_flow_update_t, frame_dt_nominal_s)
                distance_from_start_m = float(
                    np.hypot(tracker.pos_x_m - blind_hover_start_xy[0], tracker.pos_y_m - blind_hover_start_xy[1])
                )
                direction_ready = bool(
                    last_flow_dir_enu is not None
                    and len(blind_hover_speed_history_mps) >= int(blind_hover_direction_ready_min_samples)
                )
                soft_arm = bool(
                    distance_from_start_m >= float(blind_hover_soft_arm_distance_m)
                    and lightglue_consecutive_no_match >= int(blind_hover_soft_arm_min_no_match)
                    and direction_ready
                )
                blind_hover_armed = bool(
                    distance_from_start_m >= float(blind_hover_arm_distance_m) or soft_arm
                )
                stall_trigger_updates = (
                    1
                    if lightglue_consecutive_no_match >= int(blind_hover_soft_arm_min_no_match)
                    else int(blind_hover_min_stall_updates)
                )

                flow_diag = FlowStepDiagnostics(
                    valid=False,
                    heading_used_rad=heading_est_rad,
                    flow_std_px=flow_std_px,
                    learned_scale=float(tracker.learned_scale),
                    reject_reason="flow_quality_gate" if flow_quality_rejected else "",
                )

                if flow_valid:
                    est_x_m, est_y_m, flow_diag = tracker.integrate_flow_step(
                        du_px=du_px,
                        dv_px=dv_px,
                        flow_std_px=flow_std_px,
                        frame_dt_s=latest_flow_dt_s,
                        heading_rad=heading_est_rad,
                        sign_cfg=active_sign_cfg,
                    )
                    if force_srt_direction_lock and use_gt_heading:
                        raw_step_xy = np.array([flow_diag.flow_enu_x_m, flow_diag.flow_enu_y_m], dtype=np.float64)
                        locked_step_xy = _project_xy_forward_onto_direction(raw_step_xy, gt_dir)
                        tracker.set_absolute_position(
                            x_m=float(tracker.pos_x_m - raw_step_xy[0] + locked_step_xy[0]),
                            y_m=float(tracker.pos_y_m - raw_step_xy[1] + locked_step_xy[1]),
                        )
                        est_x_m = tracker.pos_x_m
                        est_y_m = tracker.pos_y_m
                        flow_diag = replace(
                            flow_diag,
                                flow_enu_x_m=float(locked_step_xy[0]),
                                flow_enu_y_m=float(locked_step_xy[1]),
                                zupt_applied=bool(float(np.hypot(locked_step_xy[0], locked_step_xy[1])) <= EPSILON),
                            )

                    flow_step_m = float(np.hypot(flow_diag.flow_enu_x_m, flow_diag.flow_enu_y_m))
                    flow_speed_mps = float(flow_step_m / max(latest_flow_dt_s, 1e-6))
                    ref_speed_mps = (
                        float(np.median(np.asarray(blind_hover_speed_history_mps, dtype=np.float64)))
                        if blind_hover_speed_history_mps
                        else float(max(last_flow_speed_mps, 0.0))
                    )
                    low_effective_speed = bool(
                        lightglue_consecutive_no_match >= int(blind_hover_low_speed_no_match_trigger)
                        and ref_speed_mps > 1.0
                        and flow_speed_mps < float(blind_hover_low_speed_ratio) * ref_speed_mps
                    )
                    stalled_motion = bool(
                        flow_diag.zupt_applied
                        or flow_step_m <= float(blind_hover_min_step_m)
                        or low_effective_speed
                    )
                    if correction_applied_this_frame:
                        blind_hover_stall_updates = 0
                    elif flow_diag.reject_reason:
                        blind_hover_stall_updates = 0
                    elif not blind_hover_armed:
                        blind_hover_stall_updates = 0
                    elif stalled_motion:
                        blind_hover_stall_updates += 1
                    else:
                        blind_hover_stall_updates = 0

                    if blind_hover_stall_updates >= stall_trigger_updates:
                        replaced_flow_xy = np.array(
                            [float(flow_diag.flow_enu_x_m), float(flow_diag.flow_enu_y_m)],
                            dtype=np.float64,
                        )
                        if float(np.hypot(replaced_flow_xy[0], replaced_flow_xy[1])) > EPSILON:
                            tracker.set_absolute_position(
                                x_m=float(tracker.pos_x_m - replaced_flow_xy[0]),
                                y_m=float(tracker.pos_y_m - replaced_flow_xy[1]),
                            )
                        est_x_m, est_y_m, blind_step_xy = _apply_blind_propagation(
                            video_t=float(video_t),
                            step_dt_s=float(latest_flow_dt_s),
                        )
                        blind_applied_this_update = True
                        flow_diag = replace(
                            flow_diag,
                            flow_enu_x_m=float(blind_step_xy[0]),
                            flow_enu_y_m=float(blind_step_xy[1]),
                            zupt_applied=False,
                        )
                    elif verbose and (frame_idx % max(1, int(fps)) == 0) and (not blind_hover_armed):
                        print(
                            f">>> BLIND HOLD @ {video_t:.2f}s: "
                            f"armed=0 dist={distance_from_start_m:.1f}m<{float(blind_hover_arm_distance_m):.1f}m"
                        )
                else:
                    est_x_m = tracker.pos_x_m
                    est_y_m = tracker.pos_y_m
                    if not correction_applied_this_frame:
                        if flow_quality_rejected:
                            blind_hover_stall_updates = 0
                        elif blind_hover_armed:
                            blind_hover_stall_updates += 1
                        else:
                            blind_hover_stall_updates = 0
                    else:
                        blind_hover_stall_updates = 0

                    if blind_hover_stall_updates >= stall_trigger_updates:
                        est_x_m, est_y_m, blind_step_xy = _apply_blind_propagation(
                            video_t=float(video_t),
                            step_dt_s=float(latest_flow_dt_s),
                        )
                        blind_applied_this_update = True
                        flow_diag = replace(
                            flow_diag,
                            valid=True,
                            flow_enu_x_m=float(blind_step_xy[0]),
                            flow_enu_y_m=float(blind_step_xy[1]),
                            zupt_applied=False,
                        )

                last_flow_update_t = video_t
            else:
                est_x_m = tracker.pos_x_m
                est_y_m = tracker.pos_y_m

            if flow_diag.valid:
                last_flow_speed_mps = float(
                    np.hypot(flow_diag.flow_enu_x_m, flow_diag.flow_enu_y_m) / max(latest_flow_dt_s, 1e-6)
                )
                if (not flow_diag.zupt_applied) and (not blind_applied_this_update) and not flow_diag.reject_reason:
                    flow_odom_xy = flow_odom_xy + np.array(
                        [flow_diag.flow_enu_x_m, flow_diag.flow_enu_y_m],
                        dtype=np.float64,
                    )
                if (
                    (not flow_diag.zupt_applied)
                    and (not blind_applied_this_update)
                    and last_flow_speed_mps >= float(blind_hover_min_speed_for_history_mps)
                ):
                    blind_hover_speed_history_mps.append(last_flow_speed_mps)
                current_flow_velocity_enu = np.array(
                    [
                        float(flow_diag.flow_enu_x_m / max(latest_flow_dt_s, 1e-6)),
                        float(flow_diag.flow_enu_y_m / max(latest_flow_dt_s, 1e-6)),
                    ],
                    dtype=np.float64,
                )
                if last_flow_velocity_update_t is not None:
                    vel_dt_s = float(video_t - last_flow_velocity_update_t)
                    if vel_dt_s > 1e-3:
                        flow_accel_enu_mps2 = (current_flow_velocity_enu - flow_velocity_enu_mps) / vel_dt_s
                flow_velocity_enu_mps = current_flow_velocity_enu
                last_flow_velocity_update_t = float(video_t)

                flow_step_norm = float(np.hypot(flow_diag.flow_enu_x_m, flow_diag.flow_enu_y_m))
                if flow_step_norm > 1e-6:
                    last_flow_dir_enu = np.array(
                        [flow_diag.flow_enu_x_m / flow_step_norm, flow_diag.flow_enu_y_m / flow_step_norm],
                        dtype=np.float64,
                    )
                if (not use_gt_heading) and last_accepted_correction_xy is None:
                    heading_est_rad = float(np.arctan2(flow_diag.flow_enu_y_m, flow_diag.flow_enu_x_m))
                dir_error_deg = _direction_error_deg(
                    flow_x_m=flow_diag.flow_enu_x_m,
                    flow_y_m=flow_diag.flow_enu_y_m,
                    gt_dir=gt_dir,
                )
            else:
                dir_error_deg = float("nan")

            gt_step_m = current_gt_step_m
            vo_step_scaled_m = float(np.hypot(flow_diag.flow_enu_x_m, flow_diag.flow_enu_y_m)) if flow_diag.valid else 0.0

            if enable_srt_scale_assist and use_gt_heading:
                frame_dt_s = frame_dt_nominal_s
                scale_window_dt_s += frame_dt_s
                scale_window_gt_disp_m += gt_step_m
                if flow_diag.valid and (not flow_diag.zupt_applied) and (not blind_applied_this_update):
                    scale_window_vo_disp_m += vo_step_scaled_m
                    scale_window_flow_count += 1
                    scale_window_flow_std_sum_px += float(max(0.0, flow_diag.flow_std_px))

                if scale_window_dt_s >= 1.0:
                    if (
                        scale_window_gt_disp_m > 0.50
                        and scale_window_vo_disp_m > 0.50
                        and tracker.learned_scale > 1e-6
                        and scale_window_flow_count >= max(3, int(round(0.5 * float(vio_rate_hz))))
                    ):
                        vo_unscaled_disp_m = scale_window_vo_disp_m / tracker.learned_scale
                        scale_measurement = float(scale_window_gt_disp_m / max(vo_unscaled_disp_m, 1e-6))
                        relative_scale = float(scale_measurement / max(tracker.learned_scale, 1e-6))
                        mean_flow_std_px = float(scale_window_flow_std_sum_px / max(scale_window_flow_count, 1))
                        scale_outlier = bool(relative_scale < 0.67 or relative_scale > 1.50)
                        noisy_large_scale_jump = bool(
                            mean_flow_std_px > 18.0 and (relative_scale < 0.80 or relative_scale > 1.25)
                        )
                        if not (scale_outlier or noisy_large_scale_jump):
                            tracker.update_scale_from_measurement(
                                scale_measurement=scale_measurement,
                                alpha=float(srt_scale_assist_alpha),
                                max_update_delta=0.025,
                            )

                    scale_window_dt_s = 0.0
                    scale_window_gt_disp_m = 0.0
                    scale_window_vo_disp_m = 0.0
                    scale_window_flow_count = 0
                    scale_window_flow_std_sum_px = 0.0

            heading_gt_deg = float(np.degrees(heading_gt_rad))
            heading_est_deg = float(np.degrees(heading_est_rad))
            heading_used_deg = float(np.degrees(flow_diag.heading_used_rad))
            delta_heading_deg = float(np.degrees(_wrap_angle_rad(heading_est_rad - heading_gt_rad)))

            error_m = float(np.hypot(est_x_m - true_x_m, est_y_m - true_y_m))

            timestamps.append(video_t)
            errors.append(error_m)
            if collect_frame_debug:
                est_xs.append(est_x_m)
                est_ys.append(est_y_m)
                true_x_out.append(true_x_m)
                true_y_out.append(true_y_m)
                heading_gt_deg_out.append(heading_gt_deg)
                heading_gt_from_srt_out.append(1 if heading_gt_from_srt else 0)
                heading_est_deg_out.append(heading_est_deg)
                heading_used_deg_out.append(heading_used_deg)
                delta_heading_deg_out.append(delta_heading_deg)
                dir_error_deg_out.append(dir_error_deg)
                flow_du_px_out.append(flow_diag.du_px)
                flow_dv_px_out.append(flow_diag.dv_px)
                flow_std_px_out.append(flow_diag.flow_std_px)
                flow_body_x_out.append(flow_diag.flow_body_x_m)
                flow_body_y_out.append(flow_diag.flow_body_y_m)
                flow_enu_x_out.append(flow_diag.flow_enu_x_m)
                flow_enu_y_out.append(flow_diag.flow_enu_y_m)
                flow_zupt_out.append(1 if flow_diag.zupt_applied else 0)
                learned_scale_out.append(float(tracker.learned_scale))
                sign_swap_out.append(1 if active_sign_cfg.swap else 0)
                sign_sx_out.append(active_sign_cfg.sx)
                sign_sy_out.append(active_sign_cfg.sy)
                lightglue_event_out.append(lightglue_event)
                lightglue_capture_frame_out.append(lightglue_capture_frame)
                lightglue_match_count_out.append(lightglue_match_count)
                lightglue_inlier_count_out.append(lightglue_inlier_count)
                lightglue_confidence_out.append(float(lightglue_confidence))
                lightglue_residual_m_out.append(float(lightglue_residual_m))
                lightglue_apply_x_m_out.append(float(lightglue_apply_x_m))
                lightglue_apply_y_m_out.append(float(lightglue_apply_y_m))
                lightglue_error_before_m_out.append(float(lightglue_error_before_m))
                lightglue_error_after_m_out.append(float(lightglue_error_after_m))
                lightglue_reject_reason_out.append(lightglue_reject_reason)

            prev_true_x_m = true_x_m
            prev_true_y_m = true_y_m

            if verbose and frame_idx % max(1, int(fps)) == 0:
                print(
                    f"frame={frame_idx} t={video_t:.2f}s err={error_m:.2f}m "
                    f"est=({est_x_m:.2f},{est_y_m:.2f}) gt=({true_x_m:.2f},{true_y_m:.2f}) "
                    f"d_head={delta_heading_deg:.2f}deg dir_err={dir_error_deg:.2f}deg "
                    f"scale={tracker.learned_scale:.3f}"
                )

            if map_window is not None:
                keep_running = map_window.render(
                    est_x_m=est_x_m,
                    est_y_m=est_y_m,
                    true_x_m=true_x_m,
                    true_y_m=true_y_m,
                    error_m=error_m,
                    frame_idx=frame_idx,
                    roi_center_xy_enu_m=active_roi_center_xy,
                    roi_width_m=active_roi_width_m,
                    roi_height_m=active_roi_height_m,
                )
                if not keep_running:
                    break

            frame_idx += 1
    finally:
        if pending_lightglue_future is not None:
            pending_lightglue_future.cancel()
        if lightglue_executor is not None:
            lightglue_executor.shutdown(wait=False, cancel_futures=True)
        cap.release()
        if map_window is not None:
            map_window.close()

    if verbose:
        print(f">>> FINAL HEADING ESTIMATE: {float(np.degrees(heading_est_rad)):.2f} deg")

    if not errors:
        raise ValueError("No frames processed")

    error_arr = np.asarray(errors, dtype=np.float64)
    time_arr = np.asarray(timestamps, dtype=np.float64)

    rmse_m = float(np.sqrt(np.mean(error_arr * error_arr)))
    mae_m = float(np.mean(np.abs(error_arr)))
    cep95_m = float(np.percentile(error_arr, 95.0))
    max_error_m = float(np.max(error_arr))

    if csv_output_path is not None:
        out_path = Path(csv_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "time_s",
                    "est_x_m",
                    "est_y_m",
                    "gt_x_m",
                    "gt_y_m",
                    "error_m",
                    "heading_gt_deg",
                    "heading_gt_from_srt",
                    "heading_est_deg",
                    "heading_used_deg",
                    "delta_heading_deg",
                    "dir_error_deg",
                    "flow_du_px",
                    "flow_dv_px",
                    "flow_std_px",
                    "flow_body_x_m",
                    "flow_body_y_m",
                    "flow_enu_x_m",
                    "flow_enu_y_m",
                    "flow_zupt",
                    "learned_scale",
                    "sign_swap",
                    "sign_sx",
                    "sign_sy",
                    "lightglue_event",
                    "lightglue_capture_frame",
                    "lightglue_match_count",
                    "lightglue_inlier_count",
                    "lightglue_confidence",
                    "lightglue_residual_m",
                    "lightglue_apply_x_m",
                    "lightglue_apply_y_m",
                    "lightglue_error_before_m",
                    "lightglue_error_after_m",
                    "lightglue_reject_reason",
                ]
            )
            for idx in range(len(error_arr)):
                writer.writerow(
                    [
                        float(time_arr[idx]),
                        float(est_xs[idx]),
                        float(est_ys[idx]),
                        float(true_x_out[idx]),
                        float(true_y_out[idx]),
                        float(error_arr[idx]),
                        float(heading_gt_deg_out[idx]),
                        int(heading_gt_from_srt_out[idx]),
                        float(heading_est_deg_out[idx]),
                        float(heading_used_deg_out[idx]),
                        float(delta_heading_deg_out[idx]),
                        float(dir_error_deg_out[idx]),
                        float(flow_du_px_out[idx]),
                        float(flow_dv_px_out[idx]),
                        float(flow_std_px_out[idx]),
                        float(flow_body_x_out[idx]),
                        float(flow_body_y_out[idx]),
                        float(flow_enu_x_out[idx]),
                        float(flow_enu_y_out[idx]),
                        int(flow_zupt_out[idx]),
                        float(learned_scale_out[idx]),
                        int(sign_swap_out[idx]),
                        int(sign_sx_out[idx]),
                        int(sign_sy_out[idx]),
                        str(lightglue_event_out[idx]),
                        int(lightglue_capture_frame_out[idx]),
                        int(lightglue_match_count_out[idx]),
                        int(lightglue_inlier_count_out[idx]),
                        float(lightglue_confidence_out[idx]),
                        float(lightglue_residual_m_out[idx]),
                        float(lightglue_apply_x_m_out[idx]),
                        float(lightglue_apply_y_m_out[idx]),
                        float(lightglue_error_before_m_out[idx]),
                        float(lightglue_error_after_m_out[idx]),
                        str(lightglue_reject_reason_out[idx]),
                    ]
                )

    return VideoTestResult(
        frame_count=frame_idx,
        correction_count=correction_count,
        rmse_m=rmse_m,
        mae_m=mae_m,
        cep95_m=cep95_m,
        max_error_m=max_error_m,
        timestamps_s=time_arr,
        error_m=error_arr,
    )


