# Advanced Localization Prototype

Hierarchical visual-inertial localization stack for GPS-denied flight.

## Current Scope

- Layer 1: 16-state delayed ESKF core (100 Hz)
- Layer 1.5: KLT optical-flow pseudo-velocity observer (30 Hz)
- Layer 2: Coarse visual matching interface (4 Hz)
- Layer 3: Fine visual matching with multi-model RANSAC (Homography vs Fundamental), scale checks, and historical update support (1 Hz)
- MAVROS integration points for `/mavros/imu/data_raw` and `/mavros/vision_pose/pose`
- Enum failsafe codes for `VISION_LOSS`, `INCONSISTENCY`, and `HIGH_COVARIANCE`

## Pre-Flight Map Pipeline (Layer 0)

Generate grayscale + CLAHE GeoTIFF maps from Mapbox Satellite:

```bash
python scripts/preflight_map_pipeline.py --mapbox-token <TOKEN> --center-lat 41.015 --center-lon 28.979 --radius-m 2000 --gsd 0.2 --output deployment/map_preflight_clahe.tif
```

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

Optional for deterministic real LightGlue startup (without runtime weight download), set a local checkpoint path in [config/system_config.yaml](config/system_config.yaml):

```yaml
LightGlue:
	enabled: true
	checkpoint_path: "weights/LightGlue_outdoor.ckpt"
	allow_orb_fallback: true
```

For video testing with stronger LightGlue mismatch mitigation (scale alignment, yaw-aligned map patch, reflect padding, fixed 256x256 context), use the tuning profile in [config/system_config_video_LightGlue.yaml](config/system_config_video_LightGlue.yaml):

```bash
python scripts/run_video_testing.py --video video/your_flight.mp4 --srt path/to/flight.srt --map-path outputs/video_test_map.jpg --LightGlue-config-yaml config/system_config_video_LightGlue.yaml --LightGlue-window-size-m 100 --LightGlue-map-roi-size-px 500 --LightGlue-mahalanobis-gate 9.21 --LightGlue-max-speed-mps 15 --LightGlue-speed-margin-m 5 --csv-output outputs/video_test_LightGlue.csv
```

2. Run tests:

```bash
pytest
```

3. Run synthetic flight consistency check:

```bash
python -m simulation.synthetic_flight
```

4. Run vision-only video testing (no IMU):

```bash
python scripts/run_video_testing.py --video video/your_flight.mp4 --srt path/to/flight.srt --csv-output outputs/video_test.csv
```

5. Run vision-only testing with live map point&trail window:

```bash
python scripts/run_video_testing.py --video video/your_flight.mp4 --srt path/to/flight.srt --map-path deployment/map_preflight_clahe.tif --map-gsd 0.2 --show-map-window --csv-output outputs/video_test_with_map.csv
```

6. Download map from Mapbox and run live map point&trail in one command:

```bash
python scripts/run_video_testing.py --video video/your_flight.mp4 --srt path/to/flight.srt --download-map --mapbox-token <TOKEN> --map-gsd 0.2 --map-output outputs/video_test_map.tif --show-map-window --csv-output outputs/video_test_with_map.csv
```

7. For ROS 2 usage, source your ROS 2 workspace and run the node module from `src/advanced_localization/ros2/navigation_node.py`.

