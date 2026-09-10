from __future__ import annotations

import numpy as np

from advanced_localization.vision.lightglue_layer import LightGlueLayer


def test_center_shift_from_affine_matches_pure_translation() -> None:
    transform_matrix = np.array(
        [
            [1.0, 0.0, 14.0],
            [0.0, 1.0, -6.5],
        ],
        dtype=np.float32,
    )

    center_shift = LightGlueLayer._center_shift_from_affine(
        transform_matrix=transform_matrix,
        source_width_px=256.0,
        source_height_px=256.0,
        target_width_px=256.0,
        target_height_px=256.0,
    )

    assert center_shift is not None
    assert np.allclose(center_shift, np.array([14.0, -6.5], dtype=np.float64), atol=1e-9)


def test_center_shift_from_affine_detects_origin_shift_with_rotation() -> None:
    angle_rad = np.deg2rad(90.0)
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    transform_matrix = np.array(
        [
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0],
        ],
        dtype=np.float32,
    )

    center_shift = LightGlueLayer._center_shift_from_affine(
        transform_matrix=transform_matrix,
        source_width_px=256.0,
        source_height_px=256.0,
        target_width_px=256.0,
        target_height_px=256.0,
    )

    assert center_shift is not None
    assert np.allclose(center_shift, np.array([-256.0, 0.0], dtype=np.float64), atol=1e-6)


def test_rotated_patch_delta_to_enu_is_identity_at_zero_yaw() -> None:
    delta_enu_m = LightGlueLayer._rotated_patch_delta_to_enu_m(
        dx_px=10.0,
        dy_px=-4.0,
        gsd_x_m_per_px=0.2,
        gsd_y_m_per_px=0.2,
        yaw_rad=0.0,
    )

    assert np.allclose(delta_enu_m, np.array([2.0, 0.8], dtype=np.float64), atol=1e-9)


def test_rotated_patch_delta_to_enu_applies_inverse_yaw_rotation() -> None:
    yaw_rad = np.deg2rad(30.0)
    gsd_m_per_px = 0.5
    expected_enu_m = np.array([12.5, -7.0], dtype=np.float64)

    # Build a synthetic rotated-frame pixel delta from a known world ENU delta.
    dx_px_world = float(expected_enu_m[0] / gsd_m_per_px)
    dy_px_world = float(-expected_enu_m[1] / gsd_m_per_px)
    cos_neg = float(np.cos(-yaw_rad))
    sin_neg = float(np.sin(-yaw_rad))
    dx_px_rot = cos_neg * dx_px_world + sin_neg * dy_px_world
    dy_px_rot = -sin_neg * dx_px_world + cos_neg * dy_px_world

    recovered_enu_m = LightGlueLayer._rotated_patch_delta_to_enu_m(
        dx_px=dx_px_rot,
        dy_px=dy_px_rot,
        gsd_x_m_per_px=gsd_m_per_px,
        gsd_y_m_per_px=gsd_m_per_px,
        yaw_rad=yaw_rad,
    )

    assert np.allclose(recovered_enu_m, expected_enu_m, atol=1e-6)


def test_rotated_patch_delta_to_enu_handles_anisotropic_gsd() -> None:
    yaw_rad = np.deg2rad(-20.0)
    gsd_x_m_per_px = 0.2
    gsd_y_m_per_px = 0.35
    expected_enu_m = np.array([6.0, 3.0], dtype=np.float64)

    dx_px_world = float(expected_enu_m[0] / gsd_x_m_per_px)
    dy_px_world = float(-expected_enu_m[1] / gsd_y_m_per_px)
    cos_neg = float(np.cos(-yaw_rad))
    sin_neg = float(np.sin(-yaw_rad))
    dx_px_rot = cos_neg * dx_px_world + sin_neg * dy_px_world
    dy_px_rot = -sin_neg * dx_px_world + cos_neg * dy_px_world

    recovered_enu_m = LightGlueLayer._rotated_patch_delta_to_enu_m(
        dx_px=dx_px_rot,
        dy_px=dy_px_rot,
        gsd_x_m_per_px=gsd_x_m_per_px,
        gsd_y_m_per_px=gsd_y_m_per_px,
        yaw_rad=yaw_rad,
    )

    assert np.allclose(recovered_enu_m, expected_enu_m, atol=1e-6)
