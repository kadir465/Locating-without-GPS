# C++ Video Testing Runner

The C++ runner ports the Python `video_testing.py` frame loop for speed and parity testing. It keeps Python-compatible CSV fields and uses the current `VisualLocalizer` interface as the LightGlue/SuperPoint backend boundary. The backend is still replaceable with the Orin TensorRT/ONNX implementation.

Build on the ROS 2/Orin workspace:

```bash
colcon build --packages-select advanced_localization_cpp --cmake-target video_testing_runner --cmake-args -DCMAKE_BUILD_TYPE=Release
```

KLT/SRT-only test:

```bash
./build/advanced_localization_cpp/video_testing_runner \
  --video /path/to/input.mp4 \
  --srt /path/to/input.srt \
  --disable-correction \
  --csv-output outputs/cpp_video_test.csv \
  --profile-output outputs/cpp_video_profile.csv \
  --realtime-profile-output outputs/cpp_video_realtime_profile.csv \
  --flow-max-side-px 1280 \
  --vio-rate-hz 10 \
  --profile
```

`--sync-decode` disables the default background video decode queue and is useful only for baseline profiling. `--flow-max-side-px 1280` runs KLT on a resized grayscale image and rescales optical-flow pixels back to the original 4K geometry; use `--benchmark-preset parity` or `--flow-max-side-px 0` for exact full-resolution parity.

Decode-only benchmark:

```bash
./build/advanced_localization_cpp/video_testing_runner \
  --video /path/to/input.mp4 \
  --decode-only \
  --frame-source opencv \
  --profile \
  --realtime-profile-output outputs/decode_profile.csv
```

KLT-only benchmark without per-frame CSV or SRT metrics:

```bash
./build/advanced_localization_cpp/video_testing_runner \
  --video /path/to/input.mp4 \
  --klt-only \
  --flow-max-side-px 1280 \
  --profile
```

Orin NVDEC file benchmark, optimized for DJI H.265/HEVC MP4 input:

```bash
./build/advanced_localization_cpp/video_testing_runner \
  --video /path/to/input.mp4 \
  --srt /path/to/input.srt \
  --disable-correction \
  --benchmark-preset orin-file \
  --profile \
  --realtime-profile-output outputs/orin_file_profile.csv
```

`orin-file` now selects `--frame-source gst-nvdec-gray`, which uses Jetson `nvv4l2decoder enable-max-performance=true` plus `nvvidconv` to output `GRAY8` directly into the estimator. The runner only needs camera luminance for KLT and LightGlue-compatible correction, so this avoids the BGR conversion path. The compatibility path `--frame-source gst-nvdec` still outputs BGR. The default gray pipeline is tuned for H.265/HEVC MP4; use `--gst-pipeline "..."` to override it for H.264 or camera-specific pipelines. Custom native-fallback pipelines must name the final sink `appsink`.

Aircraft runtime should use the ROS 2 localization node subscribed to live camera frames, not the MP4 file runner. The launch parameter `flow_max_side_px` defaults to `1280` so the optical-flow layer processes a bounded grayscale size while preserving original-pixel flow scaling.

Predecoded image-sequence benchmark:

```bash
./build/advanced_localization_cpp/video_testing_runner \
  --video /path/to/frame_directory \
  --klt-only \
  --frame-source image-sequence \
  --image-sequence-fps 29.97002997 \
  --flow-max-side-px 1280 \
  --profile
```

Fastest WSL/offline repeat benchmark:

```bash
ffmpeg -y -i /path/to/input.mp4 -vf "scale=1280:-2,format=gray" -f rawvideo outputs/input_1280x720_gray.raw

./build/advanced_localization_cpp/video_testing_runner \
  --video outputs/input_1280x720_gray.raw \
  --frame-source raw-gray \
  --raw-width 1280 \
  --raw-height 720 \
  --raw-fps 29.97002997 \
  --srt /path/to/input.srt \
  --map-path /path/to/map.tif \
  --map-gsd 0.2 \
  --profile \
  --realtime-profile-output outputs/raw_gray_profile.csv
```

This bypasses H.265 decode during repeated SuperGlue/LightGlue and map-window tests. Use full-resolution raw gray only if disk space is acceptable; a 3840x2160 4006-frame raw file is roughly 33 GB.

KLT plus visual correction:

```bash
./build/advanced_localization_cpp/video_testing_runner \
  --video /path/to/input.mp4 \
  --srt /path/to/input.srt \
  --map-path /path/to/map.tif \
  --map-gsd 0.2 \
  --lightglue-config-yaml advanced_localization_cpp/config/system_config.yaml \
  --csv-output outputs/cpp_video_test.csv \
  --profile-output outputs/cpp_video_profile.csv
```

Map downloading remains in the existing Python pipeline. Generate or provide the map first, then pass it with `--map-path`.

For fair speed comparisons on WSL, keep the video, SRT, map, CSV, and profile output under the Linux filesystem, for example `~/gps_denied_ws/data` and `~/gps_denied_ws/outputs`, not `/mnt/c`.
