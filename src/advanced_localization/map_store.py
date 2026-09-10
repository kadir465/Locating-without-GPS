from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


class StaticGeoMap:
    def __init__(self, grayscale_map: NDArray[np.uint8], gsd_m_per_px: float) -> None:
        if grayscale_map.ndim != 2:
            raise ValueError("grayscale_map must be single-channel")
        if gsd_m_per_px <= 0.0:
            raise ValueError("gsd_m_per_px must be positive")

        self._map = grayscale_map
        self.gsd_m_per_px = gsd_m_per_px
        self._height, self._width = grayscale_map.shape
        self._center_px = np.array([self._width / 2.0, self._height / 2.0], dtype=np.float64)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    @classmethod
    def from_path(cls, image_path: str | Path, gsd_m_per_px: float, apply_clahe: bool = True) -> "StaticGeoMap":
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"map file not found: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"failed to load map image from {path}")

        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)

        return cls(image, gsd_m_per_px)

    def crop_from_enu(
        self,
        center_xy_enu_m: NDArray[np.float64],
        window_size_m: float,
        output_size_px: int,
        window_size_x_m: float | None = None,
        window_size_y_m: float | None = None,
    ) -> NDArray[np.uint8]:
        if output_size_px < 8:
            raise ValueError("output_size_px must be >= 8")

        center_xy_enu_m = np.asarray(center_xy_enu_m, dtype=np.float64)
        center_px = self._enu_to_px(center_xy_enu_m)

        window_size_x = float(window_size_m if window_size_x_m is None else window_size_x_m)
        window_size_y = float(window_size_m if window_size_y_m is None else window_size_y_m)
        window_size_x_px = max(8, int(round(window_size_x / self.gsd_m_per_px)))
        window_size_y_px = max(8, int(round(window_size_y / self.gsd_m_per_px)))

        crop = cv2.getRectSubPix(
            self._map,
            (window_size_x_px, window_size_y_px),
            (float(center_px[0]), float(center_px[1])),
        )
        if crop.shape[0] != output_size_px or crop.shape[1] != output_size_px:
            crop = cv2.resize(crop, (output_size_px, output_size_px), interpolation=cv2.INTER_LINEAR)

        return self._clahe.apply(crop)

    def _enu_to_px(self, xy_enu_m: NDArray[np.float64]) -> NDArray[np.float64]:
        east_m = xy_enu_m[0]
        north_m = xy_enu_m[1]

        px = self._center_px[0] + east_m / self.gsd_m_per_px
        py = self._center_px[1] - north_m / self.gsd_m_per_px
        return np.array([px, py], dtype=np.float64)
