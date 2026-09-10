from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def skew_symmetric(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y, z = vector
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def normalize_angle(angle_rad: float) -> float:
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


def clip_vector_norm(vector: NDArray[np.float64], max_norm: float) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm < 1e-12:
        return vector
    return vector * (max_norm / norm)


def quaternion_normalize(quaternion_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(quaternion_wxyz))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quaternion_wxyz / norm


def quaternion_multiply(lhs_wxyz: NDArray[np.float64], rhs_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    lw, lx, ly, lz = lhs_wxyz
    rw, rx, ry, rz = rhs_wxyz
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def quaternion_from_small_angle(delta_theta_rad: NDArray[np.float64]) -> NDArray[np.float64]:
    theta = float(np.linalg.norm(delta_theta_rad))
    if theta < 1e-12:
        return quaternion_normalize(
            np.array([1.0, 0.5 * delta_theta_rad[0], 0.5 * delta_theta_rad[1], 0.5 * delta_theta_rad[2]], dtype=np.float64)
        )

    half_theta = 0.5 * theta
    axis = delta_theta_rad / theta
    sin_half = np.sin(half_theta)
    return quaternion_normalize(
        np.array(
            [
                np.cos(half_theta),
                axis[0] * sin_half,
                axis[1] * sin_half,
                axis[2] * sin_half,
            ],
            dtype=np.float64,
        )
    )


def rotation_matrix_from_quaternion(quaternion_wxyz: NDArray[np.float64]) -> NDArray[np.float64]:
    w, x, y, z = quaternion_normalize(quaternion_wxyz)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def yaw_from_quaternion(quaternion_wxyz: NDArray[np.float64]) -> float:
    w, x, y, z = quaternion_normalize(quaternion_wxyz)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return normalize_angle(float(yaw))


def quaternion_from_yaw(yaw_rad: float) -> NDArray[np.float64]:
    half = 0.5 * yaw_rad
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def force_symmetric(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (matrix + matrix.T)
