from __future__ import annotations

from advanced_localization.eskf import FilterHealth
from advanced_localization.failsafe import FailsafeManager
from advanced_localization.types import FailsafeCode, UpdateResult


def test_failsafe_enters_loiter_on_rejection_limit() -> None:
    manager = FailsafeManager(max_rejections=5, max_position_trace_m2=100.0)
    health = FilterHealth(
        consecutive_rejections=5,
        consecutive_lightglue_rejections=5,
        consecutive_vision_loss=0,
        position_trace_xy_m2=10.0,
        velocity_trace_m2ps2=2.0,
        pzz_m2=4.0,
        last_update_source="rejected:LightGlue",
    )

    decision = manager.evaluate(health=health, latest_update=None)
    assert decision.mode == "LOITER"
    assert decision.code == FailsafeCode.INCONSISTENCY
    assert decision.changed


def test_failsafe_returns_to_auto_after_recovery() -> None:
    manager = FailsafeManager(max_rejections=5, max_position_trace_m2=100.0)

    manager.evaluate(
        health=FilterHealth(
            consecutive_rejections=5,
            consecutive_lightglue_rejections=5,
            consecutive_vision_loss=0,
            position_trace_xy_m2=120.0,
            velocity_trace_m2ps2=5.0,
            pzz_m2=10.0,
            last_update_source="rejected:LightGlue",
        ),
        latest_update=None,
    )

    accepted_update = UpdateResult(accepted=True, reason="accepted", source="LightGlue")
    decision = manager.evaluate(
        health=FilterHealth(
            consecutive_rejections=0,
            consecutive_lightglue_rejections=0,
            consecutive_vision_loss=0,
            position_trace_xy_m2=5.0,
            velocity_trace_m2ps2=1.0,
            pzz_m2=2.0,
            last_update_source="LightGlue",
        ),
        latest_update=accepted_update,
    )

    assert decision.mode == "AUTO"
    assert decision.code == FailsafeCode.NONE
    assert decision.changed

