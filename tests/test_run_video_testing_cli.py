from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_video_testing import ROOT as SCRIPT_ROOT
from run_video_testing import _default_map_output_for_video, _resolve_output_path


def test_default_downloaded_map_output_uses_video_stem() -> None:
    output_path = _default_map_output_for_video(Path("video/Konum1/1.mp4"))

    assert output_path == SCRIPT_ROOT / "outputs" / "1.tif"


def test_explicit_downloaded_map_output_is_preserved() -> None:
    output_path = _resolve_output_path("outputs/custom_map.tif")

    assert output_path == SCRIPT_ROOT / "outputs" / "custom_map.tif"
