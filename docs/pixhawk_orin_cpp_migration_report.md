# Pixhawk + NVIDIA Orin Nano C++ / ROS 2 uXRCE-DDS Migration Report

## Target Architecture

The first aircraft integration runs the localization stack on the NVIDIA Orin Nano as a ROS 2 C++ node and connects to Pixhawk through the PX4 uXRCE-DDS bridge. PX4 remains responsible for vehicle modes, mission execution, and failsafe authority in this first version. The companion computer publishes a state estimate only.

Runtime layout:

- Pixhawk: PX4 v1.14 or newer with the uXRCE-DDS client enabled.
- Orin Nano: ROS 2, `px4_msgs`, camera driver, Micro XRCE-DDS Agent v2.x, and `advanced_localization_cpp`.
- Link: serial or UDP between Pixhawk uXRCE-DDS client and the Micro XRCE-DDS Agent.
- Localization node: subscribes to PX4 IMU and camera frames, runs optical-flow/vision-aided ESKF, and publishes `px4_msgs/msg/VehicleVisualOdometry` to `/fmu/in/vehicle_visual_odometry`.

The C++ node intentionally does not publish `VehicleCommand`, `OffboardControlMode`, or `TrajectorySetpoint`. Advisory health is published on `/advanced_localization/status`.

## Current Python Modules and C++ Replacements

| Python module | C++ replacement |
| --- | --- |
| `config.py` | `SystemConfig` with yaml-cpp loader |
| `math_utils.py` | Eigen quaternion/vector helpers |
| `types.py` | strongly typed C++ measurement/result structs |
| `ring_buffer.py` | templated time-indexed ring buffer |
| `eskf.py` | 16-state delayed ESKF with historical replay |
| `failsafe.py` | advisory failsafe manager |
| `map_store.py` | OpenCV static map cropper |
| `vision/roi_selector.py` | shifted ROI selector |
| `vision/optical_flow_layer.py` | OpenCV KLT optical-flow layer |
| `vision/lightglue_layer.py` | `VisualLocalizer` interface with ONNX/TensorRT-ready geometric matcher path |
| `ros2/navigation_node.py` | ROS 2 C++ PX4 uXRCE-DDS node |

## Topic and Frame Contract

PX4 uXRCE-DDS topics use PX4 frames and timestamps. The estimator keeps the existing internal ENU convention to preserve Python parity, and all frame conversion is isolated in the ROS/PX4 adapter.

Subscriptions:

- `/fmu/out/sensor_combined`: IMU sample source from PX4. Accelerometer and gyro vectors are converted from PX4 FRD/body convention into the estimator convention before prediction.
- `/camera/image_mono`: camera frames as `sensor_msgs/msg/Image`; `mono8`, `8UC1`, `rgb8`, and `bgr8` encodings are supported.

Publications:

- `/fmu/in/vehicle_visual_odometry`: PX4 NED pose and velocity estimate. Position, velocity, quaternion, and covariance fields are converted from internal ENU to NED.
- `/advanced_localization/status`: compact advisory status string with mode, rejection counters, covariance traces, source, and update reason.

The node uses ROS 2 sensor-data QoS for PX4 subscriptions and does not rely on MAVROS.

## Build and Deployment

Build in a ROS 2 workspace containing matching `px4_msgs`:

```bash
mkdir -p ~/gps_denied_ws/src
cd ~/gps_denied_ws/src
git clone <this-repo>
git clone https://github.com/PX4/px4_msgs.git
cd ..
source /opt/ros/humble/setup.bash
colcon build --packages-select advanced_localization_cpp
```

Run Micro XRCE-DDS Agent with the PX4-compatible v2.x agent. Serial example:

```bash
MicroXRCEAgent serial --dev /dev/ttyTHS1 -b 921600
```

UDP example:

```bash
MicroXRCEAgent udp4 -p 8888
```

Launch the node:

```bash
ros2 launch advanced_localization_cpp localization.launch.py \
  config_yaml:=/path/to/config/system_config.yaml \
  map_path:=/path/to/map_preflight_clahe.tif
```

## Flight-Test Gates

1. Build and unit-test the C++ package on the Orin Nano.
2. Run SITL with PX4 uXRCE-DDS and verify topic connectivity.
3. Bench-test Pixhawk + Orin with props removed; verify timestamp monotonicity, topic rates, and odometry frame signs.
4. Ground/tether test with the PX4 primary estimator still authoritative.
5. Low-risk flight with state-estimate publication enabled but no external control authority.
6. Only after repeated healthy logs should estimator-fusion parameters or offboard behaviors be considered.

## Current Limitations

The C++ vision backend is structured for ONNX/TensorRT deployment through the `VisualLocalizer` interface and currently uses a C++ geometric matching path to produce measurements once ONNX model paths are configured. Production LightGlue/SuperPoint TensorRT output parsing should still be validated on the Orin before relying on visual map matching in flight.
