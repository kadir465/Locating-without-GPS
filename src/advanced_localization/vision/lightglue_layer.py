from __future__ import annotations

import warnings
from time import perf_counter
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from ..config import SystemConfig
from ..types import VisionMeasurement
from .superpoint_layer import SuperPointLayer


class LightGlueLayer:
    def compute_dynamic_covariance(
        self,
        inlier_count: int,
        confidence: float,
        R_base: float = 1.0,
        alpha: float = 10.0,
        beta: float = 0.05,
    ) -> np.ndarray:
        dynamic_term = alpha * np.exp(-beta * inlier_count * confidence)
        R_k = R_base + dynamic_term
        return self._covariance_eye2 * R_k

    def __init__(self, config: SystemConfig) -> None:
        self._config = config
        self._covariance_eye2 = np.eye(2, dtype=np.float64)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._superpoint = SuperPointLayer(config)
        self._startup_error: Optional[str] = self._superpoint.startup_error
        self._last_run_debug: dict[str, float | int | str | bool] = {
            "reason": "init",
            "yaw_deg": 0.0,
            "match_count": 0,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "confidence": 0.0,
            "reproj_rmse_px": float("nan"),
        }

        self._torch = self._superpoint.torch_module
        self._matcher = None
        self._device = self._superpoint.device
        self._real_lightglue_ready = False
        self._cached_camera_image_key: Optional[tuple[int, float, tuple[int, ...], float, float, int, bool]] = None
        self._cached_camera_image: Optional[NDArray[np.uint8]] = None
        self._cached_camera_key: Optional[tuple[int, float, tuple[int, ...], float, float, int, bool]] = None
        self._cached_camera_features = None
        self._initialize_real_lightglue()

    @property
    def is_ready(self) -> bool:
        return self._real_lightglue_ready

    @property
    def startup_error(self) -> Optional[str]:
        return self._startup_error

    @property
    def device(self) -> str:
        return self._device

    @property
    def last_run_debug(self) -> dict[str, float | int | str | bool]:
        return dict(self._last_run_debug)

    def synchronize_device(self) -> None:
        if self._torch is None or not self._device.startswith("cuda"):
            return
        try:
            self._torch.cuda.synchronize()
        except Exception:
            return

    def run(
        self,
        map_patch: NDArray[np.uint8],
        camera_patch: NDArray[np.uint8],
        roi_center_xy_enu_m: NDArray[np.float64],
        gsd_m_per_px: float,
        prior_yaw_rad: float,
        capture_timestamp_s: float,
        camera_gsd_m_per_px: Optional[float] = None,
        gsd_xy_m_per_px: Optional[tuple[float, float]] = None,
    ) -> Optional[VisionMeasurement]:
        if not self._config.lightglue.enabled:
            self._set_last_run_debug(reason="disabled", yaw_rad=prior_yaw_rad)
            return None

        if not self._real_lightglue_ready:
            self._set_last_run_debug(reason="not_ready", yaw_rad=prior_yaw_rad)
            raise RuntimeError(self._startup_error or "LightGlue is not ready.")

        start_time = perf_counter()

        gsd_x_m_per_px = float(gsd_m_per_px)
        gsd_y_m_per_px = float(gsd_m_per_px)
        if gsd_xy_m_per_px is not None:
            gsd_x_m_per_px = float(max(gsd_xy_m_per_px[0], 1e-6))
            gsd_y_m_per_px = float(max(gsd_xy_m_per_px[1], 1e-6))
        gsd_for_camera_scale = float(max(0.5 * (gsd_x_m_per_px + gsd_y_m_per_px), 1e-6))
        target_size = max(0, int(self._config.lightglue.image_size_px))

        map_img = self._to_grayscale(map_patch)
        map_img = self._rotate_with_reflect_padding(map_img, yaw_rad=prior_yaw_rad)
        if target_size > 0:
            map_img = self._center_crop_or_reflect_pad(map_img, output_size_px=target_size)
        if self._config.lightglue.use_clahe:
            map_img = self._clahe.apply(map_img)

        camera_key = (
            id(camera_patch),
            float(capture_timestamp_s),
            tuple(int(v) for v in camera_patch.shape),
            float(camera_gsd_m_per_px or 0.0),
            float(gsd_for_camera_scale),
            int(target_size),
            bool(self._config.lightglue.use_clahe),
        )
        if self._cached_camera_image_key == camera_key and self._cached_camera_image is not None:
            cam_img = self._cached_camera_image
        else:
            cam_img = self._to_grayscale(camera_patch)
            cam_img = self._match_camera_to_map_gsd(
                camera_gray=cam_img,
                camera_gsd_m_per_px=camera_gsd_m_per_px,
                map_patch_gsd_m_per_px=gsd_for_camera_scale,
            )
            if target_size > 0:
                cam_img = self._center_crop_or_reflect_pad(cam_img, output_size_px=target_size)
            elif map_img.shape != cam_img.shape:
                cam_img = cv2.resize(cam_img, (map_img.shape[1], map_img.shape[0]), interpolation=cv2.INTER_LINEAR)
            if self._config.lightglue.use_clahe:
                cam_img = self._clahe.apply(cam_img)
            self._cached_camera_image_key = camera_key
            self._cached_camera_image = cam_img

        if target_size <= 0 and map_img.shape != cam_img.shape:
            cam_img = cv2.resize(cam_img, (map_img.shape[1], map_img.shape[0]), interpolation=cv2.INTER_LINEAR)

        map_features = self._superpoint.extract(map_img)

        if self._cached_camera_key == camera_key and self._cached_camera_features is not None:
            cam_features = self._cached_camera_features
        else:
            cam_features = self._superpoint.extract(cam_img)
            self._cached_camera_key = camera_key
            self._cached_camera_features = cam_features

        if cam_features is None or map_features is None:
            self._set_last_run_debug(reason="no_features", yaw_rad=prior_yaw_rad)
            return None

        matched = self._extract_matched_points(cam_features=cam_features, map_features=map_features)
        if matched is None:
            self._set_last_run_debug(reason="no_matches", yaw_rad=prior_yaw_rad)
            return None

        keypoints_cam_np, keypoints_map_np, score_np = matched
        match_count = int(keypoints_cam_np.shape[0])
        min_matches = max(3, int(self._config.lightglue.min_matches))
        if match_count < 3:
            self._set_last_run_debug(reason="too_few_matches", yaw_rad=prior_yaw_rad, match_count=match_count)
            return None
        low_match_mode = bool(match_count < min_matches)

        inlier_mask, inlier_count, transform_matrix, homography_matrix, using_homography = self._estimate_geometric_model(
            keypoints_cam_np=keypoints_cam_np,
            keypoints_map_np=keypoints_map_np,
        )
        robust_shift = self._robust_shift_from_matches(
            keypoints_cam_np=keypoints_cam_np,
            keypoints_map_np=keypoints_map_np,
        )

        minimum_inliers = max(3, int(self._config.lightglue.min_inliers))
        using_robust_shift = False
        center_shift_px: Optional[NDArray[np.float64]] = None
        reproj_rmse_px = float("nan")
        failsafe_mode = False

        if inlier_mask is None or inlier_count < minimum_inliers:
            if robust_shift is None:
                inlier_ratio = float(inlier_count / max(1, match_count))
                self._set_last_run_debug(
                    reason="low_inliers_failsafe",
                    yaw_rad=prior_yaw_rad,
                    match_count=match_count,
                    inlier_count=inlier_count,
                    inlier_ratio=inlier_ratio,
                    min_inliers_required=minimum_inliers,
                )
                return None
            inlier_mask = robust_shift["inlier_mask"]
            inlier_count = int(np.count_nonzero(inlier_mask))
            center_shift_px = np.asarray(robust_shift["shift_px"], dtype=np.float64)
            reproj_rmse_px = float(robust_shift["rmse_px"])
            using_robust_shift = True
            failsafe_mode = True

        if inlier_mask is None or inlier_count <= 0:
            self._set_last_run_debug(reason="no_inliers", yaw_rad=prior_yaw_rad, match_count=match_count)
            return None

        good_cam = keypoints_cam_np[inlier_mask]
        good_map = keypoints_map_np[inlier_mask]
        if using_homography and homography_matrix is not None:
            projected_cam = cv2.perspectiveTransform(
                good_cam.reshape(-1, 1, 2),
                np.asarray(homography_matrix, dtype=np.float32),
            ).reshape(-1, 2)
        elif (not using_robust_shift) and transform_matrix is not None:
            projected_cam = cv2.transform(
                good_cam.reshape(-1, 1, 2),
                np.asarray(transform_matrix, dtype=np.float32),
            ).reshape(-1, 2)
        else:
            fallback_shift = np.asarray(np.median(good_map - good_cam, axis=0), dtype=np.float32)
            projected_cam = good_cam + fallback_shift

        reproj_error_px = projected_cam - good_map
        reproj_rmse_px = float(np.sqrt(np.mean(np.sum(reproj_error_px * reproj_error_px, axis=1))))

        inlier_ratio = float(inlier_count / max(1, match_count))
        min_soft_inlier = max(8, minimum_inliers)
        soft_failsafe_threshold = min_soft_inlier + 4
        if inlier_count < soft_failsafe_threshold:
            failsafe_mode = True
            confidence_weight = float(np.clip(inlier_count / max(float(soft_failsafe_threshold), 1.0), 0.08, 0.9))
        else:
            confidence_weight = 1.0

        if inlier_ratio < float(self._config.lightglue.min_inlier_ratio):
            failsafe_mode = True
            confidence_weight = min(confidence_weight, 0.15)

        score_inliers = score_np[inlier_mask] if score_np.shape[0] == inlier_mask.shape[0] else score_np
        if score_inliers.size == 0:
            score_inliers = score_np
        match_confidence = float(np.mean(score_inliers)) if score_inliers.size > 0 else 0.0
        if match_confidence < float(self._config.lightglue.min_confidence):
            failsafe_mode = True
            confidence_weight = min(
                confidence_weight,
                float(np.clip(match_confidence / max(float(self._config.lightglue.min_confidence), 1e-6), 0.08, 0.35)),
            )

        if center_shift_px is None:
            if using_homography and homography_matrix is not None:
                center_shift_px = self._center_shift_from_homography(
                    homography_matrix=homography_matrix,
                    source_width_px=float(cam_img.shape[1]),
                    source_height_px=float(cam_img.shape[0]),
                    target_width_px=float(map_img.shape[1]),
                    target_height_px=float(map_img.shape[0]),
                )
            elif transform_matrix is not None:
                center_shift_px = self._center_shift_from_affine(
                    transform_matrix=transform_matrix,
                    source_width_px=float(cam_img.shape[1]),
                    source_height_px=float(cam_img.shape[0]),
                    target_width_px=float(map_img.shape[1]),
                    target_height_px=float(map_img.shape[0]),
                )
            if center_shift_px is None and robust_shift is not None:
                center_shift_px = np.asarray(robust_shift["shift_px"], dtype=np.float64)
                using_robust_shift = True
                failsafe_mode = True
                confidence_weight = min(confidence_weight, 0.20)

        if center_shift_px is None:
            self._set_last_run_debug(
                reason="center_shift_fail",
                yaw_rad=prior_yaw_rad,
                match_count=match_count,
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
                confidence=match_confidence,
                reproj_rmse_px=reproj_rmse_px,
            )
            return None

        median_shift_px = np.median(good_map - good_cam, axis=0).astype(np.float64)
        shift_disagreement_px = float(np.hypot(*(center_shift_px - median_shift_px)))
        shift_disagreement_limit_px = float(max(6.0, 0.06 * min(map_img.shape[0], map_img.shape[1])))
        low_support_center_model = bool(
            using_homography
            or low_match_mode
            or inlier_count < max(8, minimum_inliers + 2)
            or inlier_ratio < 0.25
        )
        if shift_disagreement_px > shift_disagreement_limit_px or (
            low_support_center_model and shift_disagreement_px > 3.0
        ):
            dx_px = float(median_shift_px[0])
            dy_px = float(median_shift_px[1])
            center_shift_source = "median"
        else:
            dx_px = float(center_shift_px[0])
            dy_px = float(center_shift_px[1])
            center_shift_source = "model_center"

        if low_match_mode:
            failsafe_mode = True
            confidence_weight = min(confidence_weight, float(np.clip(match_count / max(float(min_matches), 1.0), 0.15, 1.0)))

        delta_xy_enu_m = self._rotated_patch_delta_to_enu_m(
            dx_px=dx_px,
            dy_px=dy_px,
            gsd_x_m_per_px=gsd_x_m_per_px,
            gsd_y_m_per_px=gsd_y_m_per_px,
            yaw_rad=prior_yaw_rad,
        )
        estimate_xy_enu_m = np.asarray(roi_center_xy_enu_m, dtype=np.float64) + delta_xy_enu_m

        processing_latency_s = perf_counter() - start_time
        latency_scale = 1.0 if processing_latency_s * 1000.0 <= self._config.timing.lightglue_budget_ms else 2.5
        confidence_scale = float(np.clip(1.0 / max(match_confidence, 0.2), 1.0, 2.5))
        inlier_scale = float(np.clip(1.0 / max(inlier_ratio, 0.2), 1.0, 2.5))
        covariance_scale = latency_scale * confidence_scale * inlier_scale

        self._set_last_run_debug(
            reason="ok_robust_shift" if using_robust_shift else "ok",
            yaw_rad=prior_yaw_rad,
            match_count=match_count,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            confidence=match_confidence,
            reproj_rmse_px=reproj_rmse_px,
            failsafe_mode=failsafe_mode,
            confidence_weight=confidence_weight,
            low_match_mode=low_match_mode,
            robust_shift_used=using_robust_shift,
            center_shift_source=center_shift_source,
            center_shift_disagreement_px=shift_disagreement_px,
            center_shift_dx_px=dx_px,
            center_shift_dy_px=dy_px,
            delta_enu_x_m=float(delta_xy_enu_m[0]),
            delta_enu_y_m=float(delta_xy_enu_m[1]),
        )

        dynamic_cov = self.compute_dynamic_covariance(inlier_count, match_confidence)
        dynamic_cov = dynamic_cov / max(confidence_weight, 1e-2)
        dynamic_cov = dynamic_cov * covariance_scale

        model = (
            "LIGHTGLUE_SUPERPOINT_ROBUST_SHIFT"
            if using_robust_shift
            else (
                "LIGHTGLUE_SUPERPOINT_HOMOGRAPHY_CENTER_SHIFT"
                if using_homography
                else "LIGHTGLUE_SUPERPOINT_AFFINE_CENTER_SHIFT"
            )
        )

        return VisionMeasurement(
            timestamp_s=capture_timestamp_s,
            position_xy_enu_m=estimate_xy_enu_m,
            covariance=dynamic_cov,
            yaw_rad=prior_yaw_rad,
            source="lightglue",
            processing_latency_s=processing_latency_s,
            vision_dof=2,
            yaw_valid=False,
            model=model,
            inlier_ratio=inlier_ratio,
            match_confidence=match_confidence,
            match_count=match_count,
            inlier_count=inlier_count,
            reprojection_rmse_px=reproj_rmse_px,
            failsafe_mode=failsafe_mode,
            confidence_weight=confidence_weight,
        )

    def _set_last_run_debug(
        self,
        reason: str,
        yaw_rad: float,
        match_count: int = 0,
        inlier_count: int = 0,
        inlier_ratio: float = 0.0,
        confidence: float = 0.0,
        reproj_rmse_px: float = float("nan"),
        failsafe_mode: bool = False,
        **kwargs,
    ) -> None:
        self._last_run_debug = {
            "reason": str(reason),
            "yaw_deg": float(np.degrees(float(yaw_rad))),
            "match_count": int(match_count),
            "inlier_count": int(inlier_count),
            "inlier_ratio": float(inlier_ratio),
            "confidence": float(confidence),
            "reproj_rmse_px": float(reproj_rmse_px),
            "failsafe_mode": failsafe_mode,
        }
        self._last_run_debug.update(kwargs)

    def _estimate_geometric_model(
        self,
        keypoints_cam_np: NDArray[np.float32],
        keypoints_map_np: NDArray[np.float32],
    ) -> tuple[Optional[NDArray[np.bool_]], int, Optional[NDArray[np.float64]], Optional[NDArray[np.float64]], bool]:
        transform_matrix, inlier_mask_raw = cv2.estimateAffinePartial2D(
            keypoints_cam_np,
            keypoints_map_np,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(self._config.lightglue.ransac_reproj_threshold_px),
        )
        inlier_mask: Optional[NDArray[np.bool_]] = None
        inlier_count = 0
        using_homography = False
        homography_matrix: Optional[NDArray[np.float64]] = None

        if transform_matrix is not None and inlier_mask_raw is not None:
            inlier_mask = inlier_mask_raw.reshape(-1).astype(bool)
            inlier_count = int(np.count_nonzero(inlier_mask))

        min_inliers = max(3, int(self._config.lightglue.min_inliers))
        match_count = int(keypoints_cam_np.shape[0])
        if inlier_count < min_inliers and match_count >= 4:
            homography_candidate, homography_inlier_mask_raw = cv2.findHomography(
                keypoints_cam_np,
                keypoints_map_np,
                method=cv2.RANSAC,
                ransacReprojThreshold=float(self._config.lightglue.ransac_reproj_threshold_px),
            )
            if homography_candidate is not None and homography_inlier_mask_raw is not None:
                homo_mask = homography_inlier_mask_raw.reshape(-1).astype(bool)
                homo_count = int(np.count_nonzero(homo_mask))
                if homo_count >= min_inliers:
                    inlier_mask = homo_mask
                    inlier_count = homo_count
                    homography_matrix = np.asarray(homography_candidate, dtype=np.float64)
                    using_homography = True

        return inlier_mask, inlier_count, transform_matrix, homography_matrix, using_homography

    @staticmethod
    def _robust_shift_from_matches(
        keypoints_cam_np: NDArray[np.float32],
        keypoints_map_np: NDArray[np.float32],
    ) -> Optional[dict[str, NDArray[np.float64] | NDArray[np.bool_] | float]]:
        if keypoints_cam_np.shape[0] < 3 or keypoints_map_np.shape[0] < 3:
            return None

        deltas = keypoints_map_np.astype(np.float64) - keypoints_cam_np.astype(np.float64)
        median_shift = np.median(deltas, axis=0)
        residuals = np.linalg.norm(deltas - median_shift, axis=1)
        med_res = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - med_res)))
        robust_sigma = max(1e-6, 1.4826 * mad)
        threshold_px = float(max(2.5, med_res + 2.5 * robust_sigma))

        inlier_mask = residuals <= threshold_px
        inlier_count = int(np.count_nonzero(inlier_mask))
        if inlier_count < 3:
            order = np.argsort(residuals)
            keep_count = min(keypoints_cam_np.shape[0], max(3, int(round(0.35 * keypoints_cam_np.shape[0]))))
            inlier_mask = np.zeros_like(residuals, dtype=bool)
            inlier_mask[order[:keep_count]] = True
            inlier_count = int(np.count_nonzero(inlier_mask))
            if inlier_count < 3:
                return None

        robust_shift = np.median(deltas[inlier_mask], axis=0)
        reproj_rmse_px = float(np.sqrt(np.mean(np.sum((deltas[inlier_mask] - robust_shift) ** 2, axis=1))))
        return {
            "shift_px": np.asarray(robust_shift, dtype=np.float64),
            "inlier_mask": inlier_mask.astype(bool),
            "rmse_px": reproj_rmse_px,
        }

    def _initialize_real_lightglue(self) -> None:
        if not self._superpoint.is_ready:
            if self._startup_error is None:
                self._startup_error = "SuperPoint layer is not ready."
            return

        try:
            import torch
            from lightglue import LightGlue
        except Exception:
            self._startup_error = "LightGlue backend could not be imported."
            return

        if bool(self._config.lightglue.require_cuda) and not bool(torch.cuda.is_available()):
            message = "CUDA is required for LightGlue, but no CUDA device is available."
            warnings.warn(message)
            self._startup_error = message
            return

        self._torch = torch

        if bool(self._config.lightglue.require_cuda) and not self._device.startswith("cuda"):
            message = "LightGlue initialization blocked: model execution must use CUDA."
            warnings.warn(message)
            self._startup_error = message
            return

        if self._device.startswith("cuda"):
            try:
                self._torch.backends.cuda.matmul.allow_tf32 = True
                self._torch.backends.cudnn.allow_tf32 = True
                self._torch.backends.cudnn.benchmark = True
            except Exception:
                pass

        candidate_kwargs = [
            {
                "features": "superpoint",
                "filter_threshold": float(self._config.lightglue.filter_threshold),
                "depth_confidence": float(self._config.lightglue.depth_confidence),
                "width_confidence": float(self._config.lightglue.width_confidence),
            },
            {
                "features": "superpoint",
                "filter_threshold": float(self._config.lightglue.filter_threshold),
            },
            {
                "features": "superpoint",
            },
        ]

        matcher = None
        for kwargs in candidate_kwargs:
            try:
                matcher = LightGlue(**kwargs)
                break
            except TypeError:
                continue
            except Exception:
                self._startup_error = "LightGlue matcher initialization failed."
                return

        if matcher is None:
            self._startup_error = "LightGlue matcher initialization failed."
            return

        try:
            matcher = matcher.eval().to(self._device)
        except Exception:
            self._startup_error = "LightGlue matcher could not be moved to the target device."
            return

        self._matcher = matcher
        self._warmup_models()
        self._real_lightglue_ready = True
        self._startup_error = None

    def _warmup_models(self) -> None:
        warmup_runs = max(0, int(self._config.lightglue.warmup_runs))
        if warmup_runs <= 0 or self._torch is None or self._matcher is None:
            return

        target_size = max(64, int(self._config.lightglue.image_size_px))
        dummy = self._make_warmup_image(target_size)

        cam_features = self._superpoint.extract(dummy)
        map_features = self._superpoint.extract(dummy)
        if cam_features is None or map_features is None:
            return

        use_amp = bool(self._config.lightglue.use_amp) and self._device.startswith("cuda")
        try:
            with self._torch.inference_mode():
                for _ in range(warmup_runs):
                    if use_amp:
                        with self._torch.autocast(device_type="cuda", dtype=self._torch.float16):
                            _ = self._matcher({"image0": cam_features, "image1": map_features})
                    else:
                        _ = self._matcher({"image0": cam_features, "image1": map_features})

            if self._device.startswith("cuda"):
                self._torch.cuda.synchronize()
        except Exception:
            return

    @staticmethod
    def _make_warmup_image(size_px: int) -> NDArray[np.uint8]:
        size = max(64, int(size_px))
        yy, xx = np.indices((size, size), dtype=np.int32)
        image = ((xx * 7 + yy * 11) % 256).astype(np.uint8)

        step = max(16, size // 8)
        radius = max(2, size // 64)
        for y in range(step // 2, size, step):
            for x in range(step // 2, size, step):
                value = 255 if ((x // step) + (y // step)) % 2 == 0 else 32
                cv2.circle(image, (int(x), int(y)), radius, int(value), -1, lineType=cv2.LINE_AA)

        return image

    def _extract_matched_points(
        self,
        cam_features,
        map_features,
    ) -> Optional[tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]]:
        if self._matcher is None or self._torch is None:
            return None

        use_amp = bool(self._config.lightglue.use_amp) and self._device.startswith("cuda")
        try:
            with self._torch.inference_mode():
                if use_amp:
                    with self._torch.autocast(device_type="cuda", dtype=self._torch.float16):
                        matches = self._matcher({"image0": cam_features, "image1": map_features})
                else:
                    matches = self._matcher({"image0": cam_features, "image1": map_features})
        except Exception:
            return None

        keypoints_cam = cam_features.get("keypoints")
        keypoints_map = map_features.get("keypoints")
        if keypoints_cam is None or keypoints_map is None:
            return None

        match_pairs = matches.get("matches")
        if isinstance(match_pairs, (list, tuple)):
            if not match_pairs:
                return None
            match_pairs = match_pairs[0]
        if match_pairs is None:
            match_pairs = self._pairs_from_matches0(matches)
            if match_pairs is None:
                return None

        if keypoints_cam.ndim == 3:
            keypoints_cam = keypoints_cam[0]
        if keypoints_map.ndim == 3:
            keypoints_map = keypoints_map[0]
        if match_pairs.ndim == 3:
            match_pairs = match_pairs[0]
        if match_pairs.ndim != 2 or match_pairs.shape[1] != 2:
            return None

        valid = (match_pairs[:, 0] >= 0) & (match_pairs[:, 1] >= 0)
        if int(valid.sum().item()) == 0:
            return None

        match_pairs = match_pairs[valid].long()
        keypoints_cam = keypoints_cam[match_pairs[:, 0]].detach().cpu().numpy().astype(np.float32)
        keypoints_map = keypoints_map[match_pairs[:, 1]].detach().cpu().numpy().astype(np.float32)

        scores = matches.get("scores")
        if isinstance(scores, (list, tuple)):
            scores = scores[0] if scores else None
        if scores is not None and scores.ndim == 2:
            scores = scores[0]
        if scores is None or int(scores.shape[0]) != int(valid.shape[0]):
            score_np = np.ones((keypoints_cam.shape[0],), dtype=np.float32)
        else:
            score_np = scores[valid].detach().cpu().numpy().astype(np.float32)

        return keypoints_cam, keypoints_map, score_np

    @staticmethod
    def _center_shift_from_affine(
        transform_matrix: NDArray[np.float64] | NDArray[np.float32],
        source_width_px: float,
        source_height_px: float,
        target_width_px: float,
        target_height_px: float,
    ) -> Optional[NDArray[np.float64]]:
        matrix = np.asarray(transform_matrix, dtype=np.float32)
        if matrix.shape != (2, 3):
            return None

        source_center_x = float(source_width_px) / 2.0
        source_center_y = float(source_height_px) / 2.0
        target_center_x = float(target_width_px) / 2.0
        target_center_y = float(target_height_px) / 2.0

        source_center_pt = np.array([[[source_center_x, source_center_y]]], dtype=np.float32)
        transformed_center_pt = cv2.transform(source_center_pt, matrix)

        return np.array(
            [
                float(transformed_center_pt[0, 0, 0] - target_center_x),
                float(transformed_center_pt[0, 0, 1] - target_center_y),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _center_shift_from_homography(
        homography_matrix: NDArray[np.float64] | NDArray[np.float32],
        source_width_px: float,
        source_height_px: float,
        target_width_px: float,
        target_height_px: float,
    ) -> Optional[NDArray[np.float64]]:
        matrix = np.asarray(homography_matrix, dtype=np.float32)
        if matrix.shape != (3, 3):
            return None

        source_center_x = float(source_width_px) / 2.0
        source_center_y = float(source_height_px) / 2.0
        target_center_x = float(target_width_px) / 2.0
        target_center_y = float(target_height_px) / 2.0

        source_center_pt = np.array([[[source_center_x, source_center_y]]], dtype=np.float32)
        transformed_center_pt = cv2.perspectiveTransform(source_center_pt, matrix)

        return np.array(
            [
                float(transformed_center_pt[0, 0, 0] - target_center_x),
                float(transformed_center_pt[0, 0, 1] - target_center_y),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rotated_patch_delta_to_enu_m(
        dx_px: float,
        dy_px: float,
        gsd_x_m_per_px: float,
        gsd_y_m_per_px: float,
        yaw_rad: float,
    ) -> NDArray[np.float64]:
        cos_yaw = float(np.cos(float(yaw_rad)))
        sin_yaw = float(np.sin(float(yaw_rad)))
        world_dx_px = cos_yaw * float(dx_px) + sin_yaw * float(dy_px)
        world_dy_px = -sin_yaw * float(dx_px) + cos_yaw * float(dy_px)
        return np.array(
            [
                world_dx_px * float(gsd_x_m_per_px),
                -world_dy_px * float(gsd_y_m_per_px),
            ],
            dtype=np.float64,
        )

    def _pairs_from_matches0(self, matches):
        if self._torch is None:
            return None
        matches0 = matches.get("matches0")
        if matches0 is None:
            return None
        if matches0.ndim == 2:
            matches0 = matches0[0]
        if matches0.ndim != 1:
            return None

        idx0 = self._torch.arange(matches0.shape[0], device=matches0.device)
        valid = matches0 >= 0
        if int(valid.sum().item()) == 0:
            return None
        idx0 = idx0[valid]
        idx1 = matches0[valid]
        return self._torch.stack((idx0, idx1), dim=1)

    @staticmethod
    def _to_grayscale(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
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

    @staticmethod
    def _match_camera_to_map_gsd(
        camera_gray: NDArray[np.uint8],
        camera_gsd_m_per_px: Optional[float],
        map_patch_gsd_m_per_px: float,
    ) -> NDArray[np.uint8]:
        if camera_gsd_m_per_px is None:
            return camera_gray
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

    @staticmethod
    def _center_crop_or_reflect_pad(gray_image: NDArray[np.uint8], output_size_px: int) -> NDArray[np.uint8]:
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
