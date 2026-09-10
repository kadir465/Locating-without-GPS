from __future__ import annotations

from dataclasses import dataclass

from .eskf import FilterHealth
from .types import FailsafeCode, UpdateResult


@dataclass(frozen=True)
class FailsafeDecision:
    mode: str
    reason: str
    code: FailsafeCode
    changed: bool


class FailsafeManager:
    def __init__(self, max_rejections: int, max_position_trace_m2: float) -> None:
        self._max_rejections = max_rejections
        self._max_position_trace_m2 = max_position_trace_m2
        self._mode = "AUTO"

    @property
    def mode(self) -> str:
        return self._mode

    def evaluate(self, health: FilterHealth, latest_update: UpdateResult | None) -> FailsafeDecision:
        requested_mode = self._mode
        reason = "hold"
        code = FailsafeCode.NONE

        if self._mode == "AUTO":
            if health.consecutive_lightglue_rejections >= self._max_rejections:
                requested_mode = "LOITER"
                reason = "lightglue_rejection_limit"
                code = FailsafeCode.INCONSISTENCY
            elif health.consecutive_vision_loss >= self._max_rejections:
                requested_mode = "LOITER"
                reason = "vision_loss_limit"
                code = FailsafeCode.VISION_LOSS
            elif health.position_trace_xy_m2 > self._max_position_trace_m2:
                requested_mode = "LOITER"
                reason = "covariance_limit"
                code = FailsafeCode.HIGH_COVARIANCE

        elif self._mode == "LOITER":
            if latest_update is not None and latest_update.accepted and health.position_trace_xy_m2 <= self._max_position_trace_m2:
                requested_mode = "AUTO"
                reason = "vision_recovered"
                code = FailsafeCode.NONE

        changed = requested_mode != self._mode
        if changed:
            self._mode = requested_mode

        return FailsafeDecision(mode=self._mode, reason=reason, code=code, changed=changed)
