from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

from ..config import SystemConfig
from ..eskf import CascadedESKF
from ..failsafe import FailsafeManager
from ..map_store import StaticGeoMap
from ..types import FailsafeCode, ImuSample, UpdateResult, VisionMeasurement
from ..vision.lightglue_layer import LightGlueLayer
from ..vision.optical_flow_layer import OpticalFlowLayer
from ..vision.roi_selector import ShiftedRoiSelector

try:
    import rclpy
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import PoseStamped
    from mavros_msgs.srv import CommandLong, SetMode
    from rclpy.node import Node
    from sensor_msgs.msg import Image, Imu
    from std_msgs.msg import Bool, String
except Exception:
    rclpy = None
    Time = None
    PoseStamped = None
    CommandLong = None
    SetMode = None
    Node = object
    Image = None
    Imu = None
    Bool = None
    String = None


def _stamp_to_seconds(stamp: object) -> float:
    sec = getattr(stamp, "sec", 0)
    nanosec = getattr(stamp, "nanosec", 0)
    return float(sec) + float(nanosec) * 1e-9


def _seconds_to_stamp(timestamp_s: float) -> Any:
    if Time is None:
        raise RuntimeError("builtin_interfaces.msg.Time is unavailable")
    sec = int(timestamp_s)
    nanosec = int((timestamp_s - float(sec)) * 1e9)
    msg = Time()
    msg.sec = sec
    msg.nanosec = nanosec
    return msg


def _load_camera_calibration(
    yaml_path: Path,
    fallback_camera_matrix: NDArray[np.float64],
    fallback_distortion: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if not yaml_path.exists():
        return fallback_camera_matrix.copy(), fallback_distortion.copy()

    with yaml_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    camera_matrix = fallback_camera_matrix.copy()
    distortion = fallback_distortion.copy()

    if "camera_matrix" in raw:
        matrix_raw = raw["camera_matrix"]
        if isinstance(matrix_raw, dict) and "data" in matrix_raw:
            camera_matrix = np.asarray(matrix_raw["data"], dtype=np.float64).reshape(3, 3)
        else:
            camera_matrix = np.asarray(matrix_raw, dtype=np.float64).reshape(3, 3)

    if "distortion_coefficients" in raw:
        dist_raw = raw["distortion_coefficients"]
        if isinstance(dist_raw, dict) and "data" in dist_raw:
            distortion = np.asarray(dist_raw["data"], dtype=np.float64)
        else:
            distortion = np.asarray(dist_raw, dtype=np.float64)
    elif "distortion_coeffs" in raw:
        distortion = np.asarray(raw["distortion_coeffs"], dtype=np.float64)

    return camera_matrix, distortion


if rclpy is not None:

    class HierarchicalLocalizationNode(Node):
        def __init__(self) -> None:
            super().__init__("hierarchical_localization_node")

            self.declare_parameter("config_yaml", "")
            config_yaml_path = str(self.get_parameter("config_yaml").value)
            self._config = SystemConfig.from_yaml(config_yaml_path) if config_yaml_path else SystemConfig()

            self.declare_parameter("map_path", "")
            self.declare_parameter("map_gsd_m_per_px", 0.2)
            self.declare_parameter("start_x_enu_m", 0.0)
            self.declare_parameter("start_y_enu_m", 0.0)
            self.declare_parameter("start_z_enu_m", 0.0)
            self.declare_parameter("start_yaw_deg", 0.0)
            self.declare_parameter("camera_info_yaml", self._config.camera.info_yaml_path)

            map_path = str(self.get_parameter("map_path").value)
            map_gsd = float(self.get_parameter("map_gsd_m_per_px").value)

            camera_yaml = Path(str(self.get_parameter("camera_info_yaml").value))
            self._camera_matrix, self._distortion_coeffs = _load_camera_calibration(
                yaml_path=camera_yaml,
                fallback_camera_matrix=self._config.camera.camera_matrix,
                fallback_distortion=self._config.camera.distortion_coeffs,
            )

            self._map: Optional[StaticGeoMap] = None
            if map_path:
                try:
                    self._map = StaticGeoMap.from_path(map_path, gsd_m_per_px=map_gsd, apply_clahe=True)
                    self.get_logger().info(f"Loaded static map: {map_path}")
                except Exception as exc:
                    self.get_logger().error(f"Failed to load map '{map_path}': {exc}")
            else:
                self.get_logger().warn("Parameter map_path is empty. Vision matching will be skipped.")

            self._filter = CascadedESKF(self._config)
            self._failsafe = FailsafeManager(
                max_rejections=self._config.gate.max_consecutive_rejections,
                max_position_trace_m2=self._config.gate.max_position_trace_m2,
            )

            self._lightglue = LightGlueLayer(self._config)
            self._roi_selector = ShiftedRoiSelector(
                base_window_size_m=float(self._config.lightglue.roi_base_window_m),
                min_speed_mps=1.0,
                max_window_size_m=float(self._config.lightglue.roi_max_window_m),
                padding_sigma=float(self._config.lightglue.roi_padding_sigma),
            )
            self._flow_layer = OpticalFlowLayer(self._config, self._camera_matrix)

            self._start_position_enu_m = np.array(
                [
                    float(self.get_parameter("start_x_enu_m").value),
                    float(self.get_parameter("start_y_enu_m").value),
                    float(self.get_parameter("start_z_enu_m").value),
                ],
                dtype=np.float64,
            )
            self._start_yaw_rad = np.deg2rad(float(self.get_parameter("start_yaw_deg").value))

            self._frame_lock = threading.Lock()
            self._latest_frame: Optional[NDArray[np.uint8]] = None
            self._latest_frame_timestamp_s: Optional[float] = None
            self._last_lightglue_measurement: Optional[VisionMeasurement] = None
            self._latest_flow_velocity_xy_mps = np.zeros(2, dtype=np.float64)
            self._latest_flow_accel_xy_mps2 = np.zeros(2, dtype=np.float64)
            self._last_flow_velocity_timestamp_s: Optional[float] = None

            self._auto_excitation_sent = False
            self._last_failsafe_code = FailsafeCode.NONE

            self._vision_pose_pub = self.create_publisher(PoseStamped, "/mavros/vision_pose/pose", 20)
            self._status_pub = self.create_publisher(String, "/advanced_localization/status", 10)
            self._excitation_pub = self.create_publisher(Bool, "/advanced_localization/auto_excitation_flag", 10)

            self.create_subscription(Imu, "/mavros/imu/data_raw", self._on_imu, 100)
            self.create_subscription(Image, "/camera/image_mono", self._on_image, 10)

            self.create_timer(1.0 / 4.0, self._on_lightglue_timer)
            self.create_timer(1.0 / max(self._config.flow.rate_hz, 1.0), self._on_flow_timer)

            self._set_mode_client = self.create_client(SetMode, "/mavros/set_mode") if SetMode is not None else None
            self._command_long_client = self.create_client(CommandLong, "/mavros/cmd/command") if CommandLong is not None else None

            self.get_logger().info("Hierarchical localization node initialized")

        def _on_imu(self, msg: Any) -> None:
            timestamp_s = _stamp_to_seconds(msg.header.stamp)

            if not self._filter.is_initialized:
                self._filter.initialize(
                    position_enu_m=self._start_position_enu_m,
                    yaw_rad=self._start_yaw_rad,
                    timestamp_s=timestamp_s,
                )

            imu_sample = ImuSample(
                timestamp_s=timestamp_s,
                accel_mps2=np.array(
                    [
                        msg.linear_acceleration.x,
                        msg.linear_acceleration.y,
                        msg.linear_acceleration.z,
                    ],
                    dtype=np.float64,
                ),
                gyro_rps=np.array(
                    [
                        msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z,
                    ],
                    dtype=np.float64,
                ),
            )
            self._filter.predict_from_imu(imu_sample)
            self._publish_pose(timestamp_s)
            self._publish_status(latest_update=None)
            self._check_observability_and_trigger_excitation()

        def _on_image(self, msg: Any) -> None:
            try:
                gray = self._image_to_grayscale(msg)
            except ValueError as exc:
                self.get_logger().warn(str(exc))
                return

            undistorted = cv2.undistort(gray, self._camera_matrix, self._distortion_coeffs)
            timestamp_s = _stamp_to_seconds(msg.header.stamp)

            with self._frame_lock:
                self._latest_frame = undistorted
                self._latest_frame_timestamp_s = timestamp_s

        def _on_flow_timer(self) -> None:
            if not self._filter.is_initialized:
                return

            frame, capture_timestamp_s = self._get_latest_frame()
            if frame is None or capture_timestamp_s is None:
                return

            altitude_m = max(abs(float(self._filter.position_enu_m[2])), 1.0)
            flow_measurement = self._flow_layer.run(
                frame_gray=frame,
                timestamp_s=capture_timestamp_s,
                altitude_m=altitude_m,
                yaw_rad=self._filter.yaw_rad,
            )
            if flow_measurement is None:
                return

            velocity_xy_mps = np.asarray(flow_measurement.velocity_xy_mps, dtype=np.float64)
            if self._last_flow_velocity_timestamp_s is not None:
                dt_s = float(capture_timestamp_s - self._last_flow_velocity_timestamp_s)
                if dt_s > 1e-3:
                    self._latest_flow_accel_xy_mps2 = (velocity_xy_mps - self._latest_flow_velocity_xy_mps) / dt_s
            self._latest_flow_velocity_xy_mps = velocity_xy_mps
            self._last_flow_velocity_timestamp_s = float(capture_timestamp_s)

            flow_result = self._filter.update_from_flow(flow_measurement)
            self._publish_status(latest_update=flow_result)

        def _on_lightglue_timer(self) -> None:
            if self._map is None or not self._filter.is_initialized:
                return

            frame, capture_timestamp_s = self._get_latest_frame()
            if frame is None or capture_timestamp_s is None:
                return

            covariance_xy_m2 = self._filter.position_covariance_xy_m2
            expected_latency_s = float(
                max(
                    self._config.lightglue.roi_prediction_latency_s,
                    self._config.timing.lightglue_budget_ms * 1e-3,
                    self._filter.time_delay_s,
                )
            )

            roi_window = self._roi_selector.select(
                position_xy_enu_m=self._filter.position_enu_m[0:2],
                velocity_xy_mps=self._latest_flow_velocity_xy_mps,
                acceleration_xy_mps2=self._latest_flow_accel_xy_mps2,
                covariance_xy_m2=covariance_xy_m2,
                latency_s=expected_latency_s,
            )

            altitude_m = max(abs(float(self._filter.position_enu_m[2])), 1.0)
            focal_px_raw = float(max(self._camera_matrix[0, 0], 1e-6))
            camera_gsd_m_per_px = float(altitude_m / focal_px_raw)
            camera_footprint_m = float(camera_gsd_m_per_px * 256.0)

            map_patch = self._map.crop_from_enu(
                center_xy_enu_m=roi_window.center_xy_enu_m,
                window_size_m=camera_footprint_m,
                output_size_px=256,
                window_size_x_m=camera_footprint_m,
                window_size_y_m=camera_footprint_m,
            )
            camera_patch = self._center_crop(frame, output_size_px=256)
            patch_gsd_x_m_per_px = float(camera_gsd_m_per_px)
            patch_gsd_y_m_per_px = float(camera_gsd_m_per_px)
            patch_gsd_m_per_px = float(camera_gsd_m_per_px)

            measurement = self._lightglue.run(
                map_patch=map_patch,
                camera_patch=camera_patch,
                roi_center_xy_enu_m=roi_window.center_xy_enu_m,
                gsd_m_per_px=patch_gsd_m_per_px,
                prior_yaw_rad=self._filter.yaw_rad,
                capture_timestamp_s=capture_timestamp_s,
                camera_gsd_m_per_px=camera_gsd_m_per_px,
                gsd_xy_m_per_px=(patch_gsd_x_m_per_px, patch_gsd_y_m_per_px),
            )
            if measurement is None:
                self._filter.register_vision_loss()
                return

            saved_est_xy = np.asarray(self._filter.position_enu_m[0:2], dtype=np.float64)
            innovation_xy = np.asarray(measurement.position_xy_enu_m, dtype=np.float64) - saved_est_xy
            residual_m = float(np.hypot(innovation_xy[0], innovation_xy[1]))

            if residual_m > float(self._config.lightglue.residual_gate_m):
                result = self._filter.reject_vision_measurement(
                    source=measurement.source,
                    reason=f"residual_gate_rejected:{residual_m:.2f}m",
                )
                self._handle_vision_update(result)
                return

            prev_ai_xy = (
                np.asarray(self._last_lightglue_measurement.position_xy_enu_m, dtype=np.float64)
                if self._last_lightglue_measurement is not None
                else saved_est_xy
            )
            vo_dist_m = float(max(1.0, np.hypot(*(saved_est_xy - prev_ai_xy))))
            meas_scale = float(residual_m / vo_dist_m)
            if (
                vo_dist_m > float(self._config.lightglue.scale_consistency_min_distance_m)
                and not (
                    float(self._config.lightglue.scale_consistency_min)
                    <= meas_scale
                    <= float(self._config.lightglue.scale_consistency_max)
                )
            ):
                result = self._filter.reject_vision_measurement(
                    source=measurement.source,
                    reason=f"scale_consistency_rejected:{meas_scale:.2f}",
                )
                self._handle_vision_update(result)
                return

            trust_factor = float(measurement.inlier_ratio * (1.0 / (1.0 + residual_m)))
            trust_gain = float(
                np.clip(
                    trust_factor,
                    float(self._config.lightglue.trust_gain_min),
                    float(self._config.lightglue.trust_gain_max),
                )
            )
            trust_gain = float(max(trust_gain, 1e-3))

            pxx, pyy = self._filter.position_covariance_xy_m2
            min_var = float(self._config.gate.min_measurement_variance_m2)
            r_x = max(min_var, float(pxx * (1.0 - trust_gain) / trust_gain))
            r_y = max(min_var, float(pyy * (1.0 - trust_gain) / trust_gain))

            tuned_measurement = VisionMeasurement(
                timestamp_s=measurement.timestamp_s,
                position_xy_enu_m=measurement.position_xy_enu_m,
                yaw_rad=measurement.yaw_rad,
                covariance=np.diag(np.array([r_x, r_y], dtype=np.float64)),
                source=measurement.source,
                processing_latency_s=measurement.processing_latency_s,
                scale=measurement.scale,
                inlier_ratio=measurement.inlier_ratio,
                vision_dof=measurement.vision_dof,
                yaw_valid=measurement.yaw_valid,
                model=measurement.model,
                match_confidence=measurement.match_confidence,
            )

            result = self._filter.apply_historical_vision_update(tuned_measurement)
            if result.accepted:
                self._last_lightglue_measurement = tuned_measurement

            self._handle_vision_update(result)

        def _check_observability_and_trigger_excitation(self) -> None:
            if self._auto_excitation_sent and self._config.observability.trigger_once:
                return

            if self._filter.velocity_trace_m2ps2 <= self._config.observability.velocity_trace_limit_m2ps2:
                return

            flag_msg = Bool()
            flag_msg.data = True
            self._excitation_pub.publish(flag_msg)

            if self._command_long_client is not None and self._command_long_client.wait_for_service(timeout_sec=0.1):
                req = CommandLong.Request()
                req.broadcast = False
                req.command = int(self._config.observability.command_id)
                req.confirmation = 0
                req.param1 = 1.0
                req.param2 = 0.0
                req.param3 = 0.0
                req.param4 = 0.0
                req.param5 = 0.0
                req.param6 = 0.0
                req.param7 = 0.0
                self._command_long_client.call_async(req)

            self._auto_excitation_sent = True
            self.get_logger().warn("Observability guard requested one-shot S-maneuver excitation")

        def _handle_vision_update(self, result: UpdateResult) -> None:
            decision = self._failsafe.evaluate(self._filter.health, latest_update=result)
            self._last_failsafe_code = decision.code
            if decision.changed:
                self._request_mode(decision.mode)
                self.get_logger().warn(f"Failsafe mode switched to {decision.mode}: {decision.reason} [{decision.code.value}]")

            if self._filter.timestamp_s is not None:
                self._publish_pose(self._filter.timestamp_s)
            self._publish_status(latest_update=result)

        def _request_mode(self, target_mode: str) -> None:
            if self._set_mode_client is None:
                return
            if not self._set_mode_client.wait_for_service(timeout_sec=0.1):
                self.get_logger().warn("/mavros/set_mode service not available")
                return

            custom_mode = "AUTO.MISSION" if target_mode == "AUTO" else "AUTO.LOITER"
            request = SetMode.Request()
            request.base_mode = 0
            request.custom_mode = custom_mode
            self._set_mode_client.call_async(request)

        def _publish_pose(self, timestamp_s: float) -> None:
            pose = PoseStamped()
            pose.header.stamp = _seconds_to_stamp(timestamp_s)
            pose.header.frame_id = "map"

            position = self._filter.position_enu_m
            quaternion = self._filter.quaternion_wxyz

            pose.pose.position.x = float(position[0])
            pose.pose.position.y = float(position[1])
            pose.pose.position.z = float(position[2])

            pose.pose.orientation.w = float(quaternion[0])
            pose.pose.orientation.x = float(quaternion[1])
            pose.pose.orientation.y = float(quaternion[2])
            pose.pose.orientation.z = float(quaternion[3])

            self._vision_pose_pub.publish(pose)

        def _publish_status(self, latest_update: Optional[UpdateResult]) -> None:
            health = self._filter.health
            update_status = latest_update.reason if latest_update is not None else "predict_only"
            data = (
                f"mode={self._failsafe.mode};"
                f"failsafe_code={self._last_failsafe_code.value};"
                f"rej_all={health.consecutive_rejections};"
                f"rej_lightglue={health.consecutive_lightglue_rejections};"
                f"vision_loss={health.consecutive_vision_loss};"
                f"trace_xy={health.position_trace_xy_m2:.3f};"
                f"trace_v={health.velocity_trace_m2ps2:.3f};"
                f"pzz={health.pzz_m2:.3f};"
                f"last_source={health.last_update_source};"
                f"update={update_status}"
            )
            msg = String()
            msg.data = data
            self._status_pub.publish(msg)

        def _get_latest_frame(self) -> tuple[Optional[NDArray[np.uint8]], Optional[float]]:
            with self._frame_lock:
                if self._latest_frame is None or self._latest_frame_timestamp_s is None:
                    return None, None
                return self._latest_frame.copy(), self._latest_frame_timestamp_s

        def _image_to_grayscale(self, msg: Any) -> NDArray[np.uint8]:
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            encoding = str(msg.encoding).lower()

            if encoding in {"mono8", "8uc1"}:
                if msg.step < msg.width:
                    raise ValueError("invalid mono8 image step")
                row_major = raw.reshape((msg.height, msg.step))[:, : msg.width]
                return row_major.copy()

            if encoding in {"rgb8", "bgr8"}:
                channels = 3
                width_from_step = msg.step // channels
                if width_from_step < msg.width:
                    raise ValueError("invalid rgb image step")
                image = raw.reshape((msg.height, width_from_step, channels))[:, : msg.width, :]
                if encoding == "rgb8":
                    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            raise ValueError(f"unsupported camera encoding: {msg.encoding}")

        @staticmethod
        def _center_crop(image: NDArray[np.uint8], output_size_px: int) -> NDArray[np.uint8]:
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

            x0 = (width - output_size_px) // 2
            y0 = (height - output_size_px) // 2
            return image[y0 : y0 + output_size_px, x0 : x0 + output_size_px]


    def main(args: Optional[list[str]] = None) -> None:
        rclpy.init(args=args)
        node = HierarchicalLocalizationNode()
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
            rclpy.shutdown()


else:

    class HierarchicalLocalizationNode:
        def __init__(self) -> None:
            raise ImportError("ROS 2 Python dependencies are unavailable. Install rclpy and message packages.")


    def main(args: Optional[list[str]] = None) -> None:
        raise ImportError("ROS 2 Python dependencies are unavailable. Install rclpy and message packages.")


if __name__ == "__main__":
    main()
