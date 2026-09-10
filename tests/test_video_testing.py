from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from advanced_localization.video_testing import (
    FlowSignConfig,
    VisionOnlyTracker,
    _cap_blind_speed_with_ground_truth,
    _directions_from_heading_rad,
    _compute_smoothed_heading_and_directions,
    _has_minimum_live_lightglue_quality,
    _interpolate_ground_truth_state,
    _is_live_verified_lightglue_candidate,
    _is_lightglue_scale_learning_candidate,
    _is_reference_safe_lightglue_update,
    _is_within_lightglue_roi_acceptance_distance,
    _lightglue_roi_acceptance_radius_m,
    _project_xy_forward_onto_direction,
    _project_xy_onto_direction,
    _prepare_lightglue_camera_input,
    _prepare_lightglue_map_input,
    _prepare_lightglue_pair,
    _yaw_hypotheses,
    flow_to_enu,
    latlon_to_xy,
    parse_dji_srt,
    run_vision_only_video_test,
)
from advanced_localization.config import SystemConfig
from advanced_localization.vision.optical_flow_layer import OpticalFlowLayer


def test_parse_dji_srt_extracts_lat_lon(tmp_path: Path) -> None:
    srt_path = tmp_path / "flight.srt"
    srt_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:00,033\n"
        "latitude: 41.000000 longitude: 29.000000\n\n"
        "2\n"
        "00:00:01,000 --> 00:00:01,033\n"
        "latitude: 41.000100 longitude: 29.000200\n",
        encoding="utf-8",
    )

    samples = parse_dji_srt(srt_path)
    assert len(samples) == 2
    assert abs(samples[1].timestamp_s - 1.0) < 1e-6
    assert samples[0].heading_deg is None


def test_live_verified_lightglue_candidate_uses_flight_available_quality() -> None:
    assert _is_live_verified_lightglue_candidate(
        match_count=24,
        inlier_count=7,
        confidence=0.16,
        reprojection_rmse_px=1.2,
        apply_norm_m=1.2,
        mahalanobis_d2=3.0,
        mahalanobis_limit=9.21,
        yaw_error_deg=8.0,
        yaw_limit_deg=25.0,
        meas_scale=1.0,
        scale_min=0.5,
        scale_max=1.5,
        roi_bounded_candidate=False,
    )

    assert not _is_live_verified_lightglue_candidate(
        match_count=24,
        inlier_count=7,
        confidence=0.16,
        reprojection_rmse_px=3.0,
        apply_norm_m=1.0,
        mahalanobis_d2=3.0,
        mahalanobis_limit=9.21,
        yaw_error_deg=8.0,
        yaw_limit_deg=25.0,
        meas_scale=1.0,
        scale_min=0.5,
        scale_max=1.5,
        roi_bounded_candidate=False,
    )

    assert not _is_live_verified_lightglue_candidate(
        match_count=24,
        inlier_count=7,
        confidence=0.16,
        reprojection_rmse_px=1.2,
        apply_norm_m=1.0,
        mahalanobis_d2=12.0,
        mahalanobis_limit=9.21,
        yaw_error_deg=8.0,
        yaw_limit_deg=25.0,
        meas_scale=1.0,
        scale_min=0.5,
        scale_max=1.5,
        roi_bounded_candidate=False,
    )

    assert _is_live_verified_lightglue_candidate(
        match_count=24,
        inlier_count=7,
        confidence=0.16,
        reprojection_rmse_px=1.2,
        apply_norm_m=1.0,
        mahalanobis_d2=12.0,
        mahalanobis_limit=9.21,
        yaw_error_deg=8.0,
        yaw_limit_deg=25.0,
        meas_scale=1.0,
        scale_min=0.5,
        scale_max=1.5,
        roi_bounded_candidate=True,
    )


def test_low_confidence_lightglue_candidates_require_minimum_live_quality() -> None:
    assert _has_minimum_live_lightglue_quality(
        match_count=8,
        inlier_count=4,
        confidence=0.07,
        reprojection_rmse_px=5.0,
    )

    assert not _has_minimum_live_lightglue_quality(
        match_count=8,
        inlier_count=4,
        confidence=0.04,
        reprojection_rmse_px=5.0,
    )


def test_lightglue_roi_acceptance_uses_square_corner_distance() -> None:
    radius = _lightglue_roi_acceptance_radius_m(70.0)

    assert abs(radius - (0.5 * np.hypot(70.0, 70.0))) < 1e-9
    assert _is_within_lightglue_roi_acceptance_distance(radius, 70.0)
    assert not _is_within_lightglue_roi_acceptance_distance(radius + 0.01, 70.0)


def test_blind_speed_gt_cap_ignores_srt_zero_step_staircases() -> None:
    capped = _cap_blind_speed_with_ground_truth(
        adaptive_speed_mps=12.0,
        current_gt_step_m=0.0,
        frame_dt_s=1.0 / 30.0,
        recent_positive_gt_speeds_mps=[10.0, 11.0, 12.0],
    )

    assert capped == 12.0


def test_interpolate_ground_truth_state_removes_srt_stair_step() -> None:
    times = np.array([0.0, 1.0], dtype=np.float64)
    xs = np.array([0.0, 10.0], dtype=np.float64)
    ys = np.array([0.0, 0.0], dtype=np.float64)
    headings = np.deg2rad(np.array([0.0, 90.0], dtype=np.float64))
    heading_from_srt = np.array([True, True], dtype=np.bool_)

    x_m, y_m, heading_rad, direction, from_srt, idx = _interpolate_ground_truth_state(
        query_t=0.5,
        times=times,
        xs=xs,
        ys=ys,
        headings_rad=headings,
        heading_from_srt=heading_from_srt,
    )

    assert abs(x_m - 5.0) < 1e-9
    assert abs(y_m) < 1e-9
    assert abs(np.degrees(heading_rad) - 45.0) < 1e-9
    assert np.allclose(direction, np.array([np.sqrt(0.5), np.sqrt(0.5)]), atol=1e-9)
    assert from_srt
    assert idx == 0


def test_parse_dji_srt_extracts_optional_heading(tmp_path: Path) -> None:
    srt_path = tmp_path / "flight_heading.srt"
    srt_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:00,033\n"
        "latitude: 41.000000 longitude: 29.000000 heading: 123.4\n",
        encoding="utf-8",
    )

    samples = parse_dji_srt(srt_path)
    assert len(samples) == 1
    assert samples[0].heading_deg is not None
    assert abs(samples[0].heading_deg - 123.4) < 1e-6


def test_latlon_to_xy_zero_at_reference() -> None:
    x_m, y_m = latlon_to_xy(41.0, 29.0, 41.0, 29.0)
    assert abs(x_m) < 1e-9
    assert abs(y_m) < 1e-9


def test_vision_only_tracker_generates_motion_from_shifted_frames() -> None:
    frame_0 = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.circle(frame_0, (40, 60), 5, (255, 255, 255), -1)
    cv2.circle(frame_0, (80, 45), 4, (255, 255, 255), -1)
    cv2.circle(frame_0, (120, 80), 6, (255, 255, 255), -1)

    frame_1 = np.roll(frame_0, shift=4, axis=1)

    tracker = VisionOnlyTracker(focal_length_px=300.0, test_altitude_m=100.0, min_features=3)
    x0, y0 = tracker.process_frame_optical_flow(frame_0)
    x1, y1 = tracker.process_frame_optical_flow(frame_1)

    assert abs(x1 - x0) > 1e-4 or abs(y1 - y0) > 1e-4


def test_vision_only_tracker_uses_requested_altitude() -> None:
    tracker = VisionOnlyTracker(focal_length_px=300.0, test_altitude_m=35.0)
    assert abs(tracker.altitude_m - 35.0) < 1e-9
    assert abs(tracker.base_meters_per_pixel - (35.0 / 300.0)) < 1e-9


def test_flow_to_enu_matches_expected_rotation_chain() -> None:
    cfg = FlowSignConfig(swap=True, sx=1, sy=1)
    enu_x_m, enu_y_m, body_x_m, body_y_m = flow_to_enu(
        du_px=4.0,
        dv_px=10.0,
        heading_rad=0.0,
        scale_m_per_px=0.5,
        sign_cfg=cfg,
    )

    assert abs(body_x_m - 5.0) < 1e-9
    assert abs(body_y_m + 2.0) < 1e-9
    assert abs(enu_x_m - 5.0) < 1e-9
    assert abs(enu_y_m + 2.0) < 1e-9


def test_flow_to_enu_north_clockwise_rotation_matches_reference() -> None:
    cfg = FlowSignConfig(swap=True, sx=1, sy=1)
    # With this setup: forward=5m, right=-2m
    _, _, body_forward_m, body_right_m = flow_to_enu(
        du_px=4.0,
        dv_px=10.0,
        heading_rad=0.0,
        scale_m_per_px=0.5,
        sign_cfg=cfg,
        heading_is_north_cw=True,
    )
    assert abs(body_forward_m - 5.0) < 1e-9
    assert abs(body_right_m + 2.0) < 1e-9

    # Heading=0deg (North-CW): East=right, North=forward
    enu_x_m, enu_y_m, _, _ = flow_to_enu(
        du_px=4.0,
        dv_px=10.0,
        heading_rad=0.0,
        scale_m_per_px=0.5,
        sign_cfg=cfg,
        heading_is_north_cw=True,
    )
    assert abs(enu_x_m + 2.0) < 1e-9
    assert abs(enu_y_m - 5.0) < 1e-9


def test_optical_flow_layer_pixel_to_velocity_sign_convention() -> None:
    config = SystemConfig()
    camera_matrix = np.array(
        [
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    layer = OpticalFlowLayer(config=config, camera_matrix=camera_matrix)

    velocity_enu = layer._pixel_flow_to_velocity_enu(
        flow_px=np.array([10.0, 20.0], dtype=np.float32),
        dt=0.5,
        altitude_m=100.0,
        yaw_rad=0.0,
    )

    # yaw=0 (math ENU): east=forward, north=right
    # forward = +v*alt/fy/dt = 20*100/500/0.5 = 8
    # right   = -u*alt/fx/dt = -10*100/500/0.5 = -4
    assert np.allclose(velocity_enu, np.array([8.0, -4.0], dtype=np.float64), atol=1e-9)


def test_project_xy_onto_direction_removes_lateral_component() -> None:
    vector = np.array([3.0, 4.0], dtype=np.float64)
    direction = np.array([10.0, 0.0], dtype=np.float64)
    projected = _project_xy_onto_direction(vector, direction)

    assert np.allclose(projected, np.array([3.0, 0.0], dtype=np.float64), atol=1e-9)


def test_project_xy_forward_onto_direction_blocks_reverse_motion() -> None:
    direction = np.array([-1.0, 0.0], dtype=np.float64)

    forward = _project_xy_forward_onto_direction(np.array([-2.0, 3.0], dtype=np.float64), direction)
    reverse = _project_xy_forward_onto_direction(np.array([2.0, 3.0], dtype=np.float64), direction)

    assert np.allclose(forward, np.array([-2.0, 0.0], dtype=np.float64), atol=1e-9)
    assert np.allclose(reverse, np.zeros(2, dtype=np.float64), atol=1e-9)


def test_directions_from_heading_rad_tracks_each_sample() -> None:
    headings = np.deg2rad(np.array([0.0, 5.0, 10.0], dtype=np.float64))
    directions = _directions_from_heading_rad(headings)

    assert np.allclose(directions[:, 0], np.cos(headings), atol=1e-12)
    assert np.allclose(directions[:, 1], np.sin(headings), atol=1e-12)
    assert not np.allclose(directions[0], directions[-1])


def test_smoothed_heading_ignores_single_sample_gps_jitter() -> None:
    xs = np.array([0.0, 1.0, 2.0, 2.02, 2.0, 2.02, 3.0, 4.0], dtype=np.float64)
    ys = np.zeros_like(xs)

    headings, directions = _compute_smoothed_heading_and_directions(
        xs,
        ys,
        window_radius=2,
        min_displacement_m=0.35,
    )

    assert np.all(np.abs(np.degrees(headings)) < 1e-6)
    assert np.all(directions[:, 0] > 0.99)


def test_prepare_lightglue_helpers_match_legacy_pair_output() -> None:
    rng = np.random.default_rng(17)
    map_patch_bgr = rng.integers(0, 256, size=(320, 280, 3), dtype=np.uint8)
    camera_patch_bgr = rng.integers(0, 256, size=(220, 360, 3), dtype=np.uint8)

    yaw_rad = 0.37
    camera_gsd_m_per_px = 0.11
    map_patch_gsd_m_per_px = 0.20
    output_size_px = 256

    map_ref, camera_ref = _prepare_lightglue_pair(
        map_patch=map_patch_bgr,
        camera_patch_bgr=camera_patch_bgr,
        yaw_rad=yaw_rad,
        camera_gsd_m_per_px=camera_gsd_m_per_px,
        map_patch_gsd_m_per_px=map_patch_gsd_m_per_px,
        output_size_px=output_size_px,
    )

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    map_gray = cv2.cvtColor(map_patch_bgr, cv2.COLOR_BGR2GRAY)
    map_new = _prepare_lightglue_map_input(
        map_patch_gray=map_gray,
        yaw_rad=yaw_rad,
        output_size_px=output_size_px,
        clahe=clahe,
    )
    camera_new = _prepare_lightglue_camera_input(
        camera_patch_bgr=camera_patch_bgr,
        camera_gsd_m_per_px=camera_gsd_m_per_px,
        map_patch_gsd_m_per_px=map_patch_gsd_m_per_px,
        output_size_px=output_size_px,
        clahe=clahe,
    )

    assert np.array_equal(map_ref, map_new)
    assert np.array_equal(camera_ref, camera_new)


def test_yaw_hypotheses_respects_max_candidates() -> None:
    full_candidates = _yaw_hypotheses(0.23)
    limited_candidates = _yaw_hypotheses(0.23, max_candidates=3)

    assert len(full_candidates) >= 3
    assert len(limited_candidates) == 3
    assert limited_candidates == full_candidates[:3]


def test_yaw_hypotheses_stay_within_plus_minus_20_deg() -> None:
    seed_yaw = 0.75
    candidates = _yaw_hypotheses(seed_yaw, max_candidates=8)

    assert candidates
    for candidate in candidates:
        diff_rad = np.arctan2(np.sin(candidate - seed_yaw), np.cos(candidate - seed_yaw))
        assert abs(np.degrees(diff_rad)) <= 20.0001


def test_tracker_applies_zupt_when_speed_and_std_are_low() -> None:
    tracker = VisionOnlyTracker(focal_length_px=300.0, test_altitude_m=100.0, use_clahe=False)
    x_m, y_m, diag = tracker.integrate_flow_step(
        du_px=0.01,
        dv_px=0.01,
        flow_std_px=0.01,
        frame_dt_s=1.0 / 30.0,
        heading_rad=0.0,
        sign_cfg=FlowSignConfig(swap=True, sx=1, sy=1),
    )

    assert abs(x_m) < 1e-12
    assert abs(y_m) < 1e-12
    assert diag.zupt_applied


def test_tracker_does_not_zupt_when_speed_is_high() -> None:
    tracker = VisionOnlyTracker(focal_length_px=300.0, test_altitude_m=100.0, use_clahe=False)
    x_m, y_m, diag = tracker.integrate_flow_step(
        du_px=0.20,
        dv_px=0.20,
        flow_std_px=0.01,
        frame_dt_s=1.0 / 30.0,
        heading_rad=0.0,
        sign_cfg=FlowSignConfig(swap=True, sx=1, sy=1),
    )

    assert abs(x_m) > 0.0 or abs(y_m) > 0.0
    assert not diag.zupt_applied


def test_tracker_rejects_noisy_flow_without_position_jump() -> None:
    tracker = VisionOnlyTracker(focal_length_px=300.0, test_altitude_m=100.0, use_clahe=False)
    x_m, y_m, diag = tracker.integrate_flow_step(
        du_px=20.0,
        dv_px=20.0,
        flow_std_px=120.0,
        frame_dt_s=1.0 / 30.0,
        heading_rad=0.0,
        sign_cfg=FlowSignConfig(swap=True, sx=1, sy=1),
    )

    assert abs(x_m) < 1e-12
    assert abs(y_m) < 1e-12
    assert diag.reject_reason == "flow_quality_gate"


def test_tracker_updates_scale_from_lightglue_measurement_after_min_motion() -> None:
    tracker = VisionOnlyTracker(focal_length_px=300.0, test_altitude_m=100.0, use_clahe=False)
    # First accepted update only initializes reference points.
    first = tracker.update_scale_from_lightglue(
        ai_true_xy=np.array([0.0, 0.0], dtype=np.float64),
        saved_est_xy=np.array([0.0, 0.0], dtype=np.float64),
    )
    assert first is None

    scale_meas = tracker.update_scale_from_lightglue(
        ai_true_xy=np.array([7.0, 0.0], dtype=np.float64),
        saved_est_xy=np.array([6.0, 0.0], dtype=np.float64),
    )

    assert scale_meas is not None
    assert abs(float(scale_meas) - (0.725 * (7.0 / 6.0))) < 1e-9
    assert abs(tracker.learned_scale - (0.725 + ((0.725 * (7.0 / 6.0)) - 0.725) * 0.15)) < 1e-9
    assert len(tracker.scale_history) == 1


def test_tracker_lightglue_scale_uses_flow_odometry_reference_when_available() -> None:
    tracker = VisionOnlyTracker(focal_length_px=300.0, test_altitude_m=100.0, use_clahe=False)
    tracker.update_scale_from_lightglue(
        ai_true_xy=np.array([0.0, 0.0], dtype=np.float64),
        saved_est_xy=np.array([0.0, 0.0], dtype=np.float64),
        saved_flow_odom_xy=np.array([0.0, 0.0], dtype=np.float64),
    )

    scale_meas = tracker.update_scale_from_lightglue(
        ai_true_xy=np.array([10.0, 0.0], dtype=np.float64),
        saved_est_xy=np.array([4.0, 0.0], dtype=np.float64),
        saved_flow_odom_xy=np.array([10.0, 0.0], dtype=np.float64),
    )

    assert scale_meas is not None
    assert abs(float(scale_meas) - 0.725) < 1e-9
    assert abs(tracker.learned_scale - 0.725) < 1e-9


def test_reference_safe_lightglue_update_rejects_error_growth() -> None:
    assert _is_reference_safe_lightglue_update(
        error_before_m=10.0,
        error_after_m=9.5,
        apply_norm_m=2.0,
    )
    assert not _is_reference_safe_lightglue_update(
        error_before_m=10.0,
        error_after_m=10.5,
        apply_norm_m=2.0,
    )


def test_lightglue_scale_learning_requires_high_quality_local_match() -> None:
    assert _is_lightglue_scale_learning_candidate(
        match_count=32,
        inlier_count=12,
        confidence=0.20,
        reprojection_rmse_px=1.5,
        residual_m=8.0,
        apply_norm_m=1.0,
        failsafe_mode=False,
    )
    assert not _is_lightglue_scale_learning_candidate(
        match_count=16,
        inlier_count=5,
        confidence=0.108,
        reprojection_rmse_px=2.0,
        residual_m=42.0,
        apply_norm_m=5.9,
        failsafe_mode=False,
    )


def test_tracker_lightglue_scale_can_record_reference_without_learning() -> None:
    tracker = VisionOnlyTracker(focal_length_px=300.0, test_altitude_m=100.0, use_clahe=False)
    tracker.update_scale_from_lightglue(
        ai_true_xy=np.array([0.0, 0.0], dtype=np.float64),
        saved_est_xy=np.array([0.0, 0.0], dtype=np.float64),
        allow_update=False,
    )

    scale_meas = tracker.update_scale_from_lightglue(
        ai_true_xy=np.array([20.0, 0.0], dtype=np.float64),
        saved_est_xy=np.array([10.0, 0.0], dtype=np.float64),
        allow_update=False,
    )

    assert scale_meas is None
    assert abs(tracker.learned_scale - 0.725) < 1e-9
    assert tracker.scale_history == []


def test_tracker_lightglue_scale_history_keeps_last_fifteen() -> None:
    tracker = VisionOnlyTracker(focal_length_px=300.0, test_altitude_m=100.0, use_clahe=False)
    tracker.update_scale_from_lightglue(
        ai_true_xy=np.array([0.0, 0.0], dtype=np.float64),
        saved_est_xy=np.array([0.0, 0.0], dtype=np.float64),
    )

    for idx in range(1, 22):
        measured = tracker.update_scale_from_lightglue(
            ai_true_xy=np.array([6.0 * idx, 0.0], dtype=np.float64),
            saved_est_xy=np.array([6.0 * idx, 0.0], dtype=np.float64),
        )
        assert measured is not None

    assert len(tracker.scale_history) == 15
    assert abs(tracker.learned_scale - 0.725) < 1e-9


def test_tracker_scale_history_uses_last_five_and_median() -> None:
    tracker = VisionOnlyTracker(focal_length_px=300.0, test_altitude_m=100.0, use_clahe=False)
    for measurement in [0.50, 0.70, 0.80, 0.90, 1.20, 0.85]:
        tracker.update_scale_from_measurement(measurement)

    # Measurements are clamped to [0.60, 0.95] and only last 5 are kept.
    assert np.allclose(np.asarray(tracker.scale_history, dtype=np.float64), np.array([0.7, 0.8, 0.9, 0.95, 0.85]))
    assert abs(tracker.learned_scale - 0.85) < 1e-9


def test_tracker_scale_alpha_smooths_and_limits_updates() -> None:
    tracker = VisionOnlyTracker(focal_length_px=300.0, test_altitude_m=100.0, use_clahe=False)
    tracker.learned_scale = 0.70

    updated = tracker.update_scale_from_measurement(
        0.95,
        alpha=0.20,
        max_update_delta=0.025,
    )

    assert abs(updated - 0.725) < 1e-9
    assert abs(tracker.learned_scale - 0.725) < 1e-9


def _format_srt_timestamp(seconds: float) -> str:
    ms_total = int(round(seconds * 1000.0))
    hh = ms_total // 3600000
    mm = (ms_total % 3600000) // 60000
    ss = (ms_total % 60000) // 1000
    ms = ms_total % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def _write_synthetic_srt(path: Path, frame_count: int, dt_s: float) -> None:
    lat0 = 41.0
    lon0 = 29.0
    dlat_per_frame = 1e-4

    blocks: list[str] = []
    for idx in range(frame_count):
        t0 = idx * dt_s
        t1 = (idx + 1) * dt_s
        lat = lat0 + dlat_per_frame * idx
        lon = lon0
        blocks.append(
            "\n".join(
                [
                    str(idx + 1),
                    f"{_format_srt_timestamp(t0)} --> {_format_srt_timestamp(t1)}",
                    f"latitude: {lat:.7f} longitude: {lon:.7f}",
                ]
            )
        )

    path.write_text("\n\n".join(blocks), encoding="utf-8")


def _write_synthetic_video(path: Path, frame_count: int, fps: float = 30.0) -> None:
    width = 160
    height = 120
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Unable to create synthetic test video")

    try:
        for idx in range(frame_count):
            shift_px = idx * 2
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.circle(frame, (30 + shift_px, 40), 5, (255, 255, 255), -1)
            cv2.circle(frame, (80 + shift_px, 70), 6, (255, 255, 255), -1)
            cv2.circle(frame, (120 + shift_px, 30), 4, (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()


def _angle_diff_deg(a_deg: float, b_deg: float) -> float:
    return ((a_deg - b_deg + 180.0) % 360.0) - 180.0


def test_run_video_test_uses_gt_heading_for_integration(tmp_path: Path) -> None:
    frame_count = 6
    video_path = tmp_path / "synthetic.avi"
    srt_path = tmp_path / "synthetic.srt"
    csv_path = tmp_path / "result.csv"

    _write_synthetic_video(video_path, frame_count=frame_count, fps=30.0)
    _write_synthetic_srt(srt_path, frame_count=frame_count, dt_s=1.0 / 30.0)

    run_vision_only_video_test(
        video_path=video_path,
        srt_path=srt_path,
        enable_correction=False,
        use_clahe=False,
        max_frames=frame_count,
        csv_output_path=csv_path,
        use_gt_heading=True,
    )

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows
    for row in rows:
        heading_gt_deg = float(row["heading_gt_deg"])
        heading_used_deg = float(row["heading_used_deg"])
        assert abs(_angle_diff_deg(heading_used_deg, heading_gt_deg)) < 1e-6


