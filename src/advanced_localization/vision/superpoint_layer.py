from __future__ import annotations

import warnings
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from ..config import SystemConfig


class SuperPointLayer:
    def __init__(self, config: SystemConfig) -> None:
        self._config = config
        self._torch = None
        self._extractor = None
        self._device = "cpu"
        self._ready = False
        self._startup_error: Optional[str] = None
        self._initialize()

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def torch_module(self):
        return self._torch

    @property
    def device(self) -> str:
        return self._device

    @property
    def startup_error(self) -> Optional[str]:
        return self._startup_error

    def extract(self, image: NDArray[np.uint8]):
        if not self._ready or self._extractor is None or self._torch is None:
            return None

        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_np = np.ascontiguousarray(image)
        image_cpu_tensor = self._torch.from_numpy(image_np)
        if self._device.startswith("cuda"):
            try:
                image_cpu_tensor = image_cpu_tensor.pin_memory()
                image_tensor = image_cpu_tensor.to(
                    device=self._device,
                    dtype=self._torch.float32,
                    non_blocking=True,
                )
            except Exception:
                image_tensor = image_cpu_tensor.to(
                    device=self._device,
                    dtype=self._torch.float32,
                )
        else:
            image_tensor = image_cpu_tensor.to(
                device=self._device,
                dtype=self._torch.float32,
            )
        image_tensor = (image_tensor / 255.0).unsqueeze(0).unsqueeze(0)

        try:
            with self._torch.inference_mode():
                features = self._extractor.extract(image_tensor)
        except Exception:
            return None

        return features

    def _initialize(self) -> None:
        try:
            import torch
            from lightglue import SuperPoint
        except Exception:
            self._startup_error = "SuperPoint backend could not be imported."
            return

        if bool(self._config.lightglue.require_cuda) and not bool(torch.cuda.is_available()):
            message = "CUDA is required for SuperPoint + LightGlue, but no CUDA device is available."
            warnings.warn(message)
            self._startup_error = message
            return

        self._torch = torch
        self._device = self._select_device(torch)

        if bool(self._config.lightglue.require_cuda) and not self._device.startswith("cuda"):
            message = "SuperPoint initialization blocked: model execution must use CUDA."
            warnings.warn(message)
            self._startup_error = message
            return

        extractor = self._build_extractor(SuperPoint)
        if extractor is None:
            self._startup_error = "SuperPoint extractor initialization failed."
            return

        try:
            extractor = extractor.eval().to(self._device)
        except Exception:
            self._startup_error = "SuperPoint extractor could not be moved to the target device."
            return

        self._extractor = extractor
        self._ready = True

    def _build_extractor(self, superpoint_cls):
        cfg = self._config.superpoint
        candidate_kwargs = [
            {
                "max_num_keypoints": int(cfg.max_keypoints),
                "keypoint_threshold": float(cfg.keypoint_threshold),
                "nms_radius": int(cfg.nms_radius),
            },
            {
                "max_num_keypoints": int(cfg.max_keypoints),
                "detection_threshold": float(cfg.keypoint_threshold),
                "nms_radius": int(cfg.nms_radius),
            },
            {
                "max_num_keypoints": int(cfg.max_keypoints),
            },
            {},
        ]

        for kwargs in candidate_kwargs:
            try:
                return superpoint_cls(**kwargs)
            except TypeError:
                continue
            except Exception:
                return None
        return None

    def _select_device(self, torch_module) -> str:
        if bool(self._config.lightglue.require_cuda):
            return "cuda" if torch_module.cuda.is_available() else "cpu"

        requested = str(self._config.lightglue.device).strip().lower()
        if requested == "cpu":
            return "cpu"
        if requested in {"cuda", "gpu"}:
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if requested.startswith("cuda:"):
            return requested if torch_module.cuda.is_available() else "cpu"
        return "cuda" if torch_module.cuda.is_available() else "cpu"
