from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


LIGHTGLUE_EVENT_RE = re.compile(
    r"LightGlue (?P<event>MISS|REJECT|HOLD|UPDATE|STALE)\s+@\s+(?P<time_s>[0-9.]+)s:",
    re.IGNORECASE,
)
HYP_RE = re.compile(r"\bhyp=(?P<hyp>[0-9]+)", re.IGNORECASE)
WALL_RE = re.compile(r"\bwall=(?P<wall_s>[0-9.]+)s", re.IGNORECASE)
MATCH_RE = re.compile(
    r"matches=(?P<matches>[0-9]+).*?inliers=(?P<inliers>[0-9]+).*?"
    r"(?:inlier=(?P<inlier>[0-9.]+))?",
    re.IGNORECASE,
)
CONF_RE = re.compile(r"(?:conf=|confidence=)(?P<conf>[0-9.]+)", re.IGNORECASE)
GPU_ROW_RE = re.compile(r"^\s*(?P<util>[0-9]+)\s*,\s*(?P<mem>[0-9]+)\s*,\s*(?P<power>[0-9.]+|N/A)")


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "p50": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
        "p99": _percentile(values, 99.0),
        "max": max(values) if values else None,
    }


def _resolve_output(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _monitor_gpu(
    rows: list[dict[str, float]],
    stop_event: threading.Event,
    sample_ms: int,
) -> None:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits",
        f"--loop-ms={int(sample_ms)}",
    ]
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
    except OSError:
        return

    assert proc.stdout is not None
    start = time.perf_counter()
    try:
        while not stop_event.is_set():
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            match = GPU_ROW_RE.search(line)
            if match is None:
                continue
            power_raw = match.group("power")
            rows.append(
                {
                    "elapsed_s": time.perf_counter() - start,
                    "gpu_util_pct": float(match.group("util")),
                    "memory_used_mib": float(match.group("mem")),
                    "power_w": float(power_raw) if power_raw != "N/A" else float("nan"),
                }
            )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()


def _parse_lightglue_log(lines: list[str]) -> list[dict[str, float | int | str | None]]:
    events: list[dict[str, float | int | str | None]] = []
    for line in lines:
        event_match = LIGHTGLUE_EVENT_RE.search(line)
        if event_match is None:
            continue

        entry: dict[str, float | int | str | None] = {
            "event": event_match.group("event").upper(),
            "time_s": float(event_match.group("time_s")),
            "hypotheses": None,
            "wall_s": None,
            "matches": None,
            "inliers": None,
            "inlier_ratio": None,
            "confidence": None,
        }
        hyp_info = HYP_RE.search(line)
        if hyp_info is not None:
            entry["hypotheses"] = int(hyp_info.group("hyp"))
        wall_info = WALL_RE.search(line)
        if wall_info is not None:
            entry["wall_s"] = float(wall_info.group("wall_s"))
        match_info = MATCH_RE.search(line)
        if match_info is not None:
            entry["matches"] = int(match_info.group("matches"))
            entry["inliers"] = int(match_info.group("inliers"))
            if match_info.group("inlier"):
                entry["inlier_ratio"] = float(match_info.group("inlier"))
        conf_info = CONF_RE.search(line)
        if conf_info is not None:
            entry["confidence"] = float(conf_info.group("conf"))
        events.append(entry)
    return events


def _write_gpu_csv(path: Path | None, rows: list[dict[str, float]]) -> None:
    if path is None:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["elapsed_s", "gpu_util_pct", "memory_used_mib", "power_w"])
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(
    returncode: int,
    elapsed_s: float,
    events: list[dict[str, float | int | str | None]],
    gpu_rows: list[dict[str, float]],
    spike_util_pct: float,
) -> dict[str, object]:
    wall_values = [float(e["wall_s"]) for e in events if e.get("wall_s") is not None]
    match_values = [float(e["matches"]) for e in events if e.get("matches") is not None]
    inlier_values = [float(e["inliers"]) for e in events if e.get("inliers") is not None]
    confidence_values = [float(e["confidence"]) for e in events if e.get("confidence") is not None]
    gpu_util_values = [float(row["gpu_util_pct"]) for row in gpu_rows]
    gpu_mem_values = [float(row["memory_used_mib"]) for row in gpu_rows]

    event_counts: dict[str, int] = {}
    for event in events:
        name = str(event["event"])
        event_counts[name] = event_counts.get(name, 0) + 1

    return {
        "returncode": returncode,
        "elapsed_s": elapsed_s,
        "lightglue_events": event_counts,
        "lightglue_wall_s": _stats(wall_values),
        "lightglue_matches": _stats(match_values),
        "lightglue_inliers": _stats(inlier_values),
        "lightglue_confidence": _stats(confidence_values),
        "gpu_samples": len(gpu_rows),
        "gpu_util_pct": _stats(gpu_util_values),
        "gpu_memory_used_mib": _stats(gpu_mem_values),
        "gpu_spike_samples": sum(1 for value in gpu_util_values if value >= spike_util_pct),
        "gpu_spike_threshold_pct": spike_util_pct,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command while sampling nvidia-smi and summarizing LightGlue verbose logs."
    )
    parser.add_argument("--sample-ms", type=int, default=100, help="GPU sampling period for nvidia-smi.")
    parser.add_argument("--spike-util-pct", type=float, default=85.0, help="GPU util threshold counted as spike.")
    parser.add_argument("--log-output", default="outputs/lightglue_profile.log", help="Captured stdout/stderr log path.")
    parser.add_argument("--gpu-output", default="outputs/lightglue_gpu_samples.csv", help="GPU sample CSV path.")
    parser.add_argument("--summary-output", default="outputs/lightglue_profile_summary.json", help="Summary JSON path.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after --")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("Provide the profiled command after --")
    return args


def main() -> int:
    args = parse_args()
    log_path = _resolve_output(args.log_output)
    gpu_path = _resolve_output(args.gpu_output)
    summary_path = _resolve_output(args.summary_output)

    gpu_rows: list[dict[str, float]] = []
    stop_event = threading.Event()
    gpu_thread = threading.Thread(
        target=_monitor_gpu,
        args=(gpu_rows, stop_event, max(20, int(args.sample_ms))),
        daemon=True,
    )
    gpu_thread.start()

    command_start = time.perf_counter()
    log_lines: list[str] = []
    with log_path.open("w", encoding="utf-8", newline="") if log_path else subprocess.DEVNULL as log_file:
        proc = subprocess.Popen(
            args.command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
            log_lines.append(line)
        returncode = proc.wait()

    elapsed_s = time.perf_counter() - command_start
    stop_event.set()
    gpu_thread.join(timeout=3.0)

    events = _parse_lightglue_log(log_lines)
    _write_gpu_csv(gpu_path, gpu_rows)
    summary = _build_summary(
        returncode=returncode,
        elapsed_s=elapsed_s,
        events=events,
        gpu_rows=gpu_rows,
        spike_util_pct=float(args.spike_util_pct),
    )

    if summary_path is not None:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== LightGlue GPU Profile Summary ===")
    print(json.dumps(summary, indent=2))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
