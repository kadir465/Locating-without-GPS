from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import rasterio
from rasterio.transform import from_origin

EARTH_RADIUS_M = 6378137.0
MAX_ZOOM = 22
MIN_ZOOM = 0


@dataclass(frozen=True)
class MercatorBounds:
    min_x_m: float
    min_y_m: float
    max_x_m: float
    max_y_m: float


def latlon_to_mercator(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    lat_deg = float(np.clip(lat_deg, -85.05112878, 85.05112878))
    lon_deg = float(np.clip(lon_deg, -180.0, 180.0))

    x_m = EARTH_RADIUS_M * math.radians(lon_deg)
    y_m = EARTH_RADIUS_M * math.log(math.tan(math.pi * 0.25 + math.radians(lat_deg) * 0.5))
    return x_m, y_m


def mercator_to_latlon(x_m: float, y_m: float) -> tuple[float, float]:
    lon_deg = math.degrees(x_m / EARTH_RADIUS_M)
    lat_rad = 2.0 * math.atan(math.exp(y_m / EARTH_RADIUS_M)) - math.pi * 0.5
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def meters_per_pixel(latitude_deg: float, zoom: int) -> float:
    latitude_rad = math.radians(latitude_deg)
    return math.cos(latitude_rad) * 2.0 * math.pi * EARTH_RADIUS_M / (256.0 * (2**zoom))


def choose_zoom_for_gsd(latitude_deg: float, gsd_m_per_px: float) -> int:
    best_zoom = MIN_ZOOM
    best_error = float("inf")

    for zoom in range(MIN_ZOOM, MAX_ZOOM + 1):
        mpp = meters_per_pixel(latitude_deg, zoom)
        error = abs(mpp - gsd_m_per_px)
        if error < best_error:
            best_error = error
            best_zoom = zoom

    return best_zoom


def build_bounds(center_x_m: float, center_y_m: float, radius_m: float) -> MercatorBounds:
    return MercatorBounds(
        min_x_m=center_x_m - radius_m,
        min_y_m=center_y_m - radius_m,
        max_x_m=center_x_m + radius_m,
        max_y_m=center_y_m + radius_m,
    )


def create_retry_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(connect=5, read=5, backoff_factor=0.3, status_forcelist=(500, 502, 504))
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def download_google_tile(x: int, y: int, z: int, session: requests.Session) -> np.ndarray:
    url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = session.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    image_array = np.frombuffer(response.content, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode tile image from Google Maps XYZ response")
    
    return image


def build_mosaic(
    center_lat_deg: float,
    center_lon_deg: float,
    radius_m: float,
    gsd_m_per_px: float,
) -> tuple[np.ndarray, MercatorBounds, int, float]:
    zoom = choose_zoom_for_gsd(center_lat_deg, gsd_m_per_px)

    center_x_m, center_y_m = latlon_to_mercator(center_lat_deg, center_lon_deg)
    
    lat_rad = math.radians(center_lat_deg)
    scale_factor = math.cos(lat_rad)
    map_radius_m = radius_m / scale_factor
    
    bounds = build_bounds(center_x_m, center_y_m, map_radius_m)

    min_lat, min_lon = mercator_to_latlon(bounds.min_x_m, bounds.min_y_m)
    max_lat, max_lon = mercator_to_latlon(bounds.max_x_m, bounds.max_y_m)

    def lat_lon_to_tile(lat, lon, z):
        lat_rad = math.radians(lat)
        n = 2.0 ** z
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    x_min, y_max = lat_lon_to_tile(min_lat, min_lon, zoom)
    x_max, y_min = lat_lon_to_tile(max_lat, max_lon, zoom)

    tx_start = min(x_min, x_max)
    tx_end = max(x_min, x_max)
    ty_start = min(y_min, y_max)
    ty_end = max(y_min, y_max)

    num_tiles_x = tx_end - tx_start + 1
    num_tiles_y = ty_end - ty_start + 1

    stitched = np.zeros((num_tiles_y * 256, num_tiles_x * 256, 3), dtype=np.uint8)

    total_tiles = num_tiles_x * num_tiles_y
    print(f"Downloading {total_tiles} tiles at zoom {zoom}...")
    
    session = create_retry_session()

    def fetch_and_place(tx: int, ty: int) -> None:
        try:
            tile = download_google_tile(tx, ty, zoom, session)
            stitched[
                (ty - ty_start) * 256 : (ty - ty_start + 1) * 256,
                (tx - tx_start) * 256 : (tx - tx_start + 1) * 256,
                :
            ] = tile
        except Exception as e:
            print(f"Warning: Failed to fetch tile {tx},{ty}: {e}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for ty in range(ty_start, ty_end + 1):
            for tx in range(tx_start, tx_end + 1):
                futures.append(executor.submit(fetch_and_place, tx, ty))
        
        for _ in tqdm(as_completed(futures), total=total_tiles, desc="Downloading map"):
            pass

    def tile_to_mercator(tx, ty, z):
        n = 2.0 ** z
        lon_deg = tx / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ty / n)))
        lat_deg = math.degrees(lat_rad)
        return latlon_to_mercator(lat_deg, lon_deg)

    stitch_topleft_x_m, stitch_topleft_y_m = tile_to_mercator(tx_start, ty_start, zoom)

    world_width_m = 2.0 * math.pi * EARTH_RADIUS_M
    pixel_size_map_units = world_width_m / (256.0 * (2 ** zoom))

    crop_x_px = int(round((bounds.min_x_m - stitch_topleft_x_m) / pixel_size_map_units))
    crop_y_px = int(round((stitch_topleft_y_m - bounds.max_y_m) / pixel_size_map_units))

    width_px = int(round((bounds.max_x_m - bounds.min_x_m) / pixel_size_map_units))
    height_px = int(round((bounds.max_y_m - bounds.min_y_m) / pixel_size_map_units))

    crop_y_start = max(0, crop_y_px)
    crop_y_end = min(stitched.shape[0], crop_y_px + height_px)
    crop_x_start = max(0, crop_x_px)
    crop_x_end = min(stitched.shape[1], crop_x_px + width_px)

    final_mosaic = stitched[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    return final_mosaic, bounds, zoom, pixel_size_map_units


def preprocess_grayscale_clahe(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def write_geotiff(
    output_path: Path,
    image_gray: np.ndarray,
    bounds: MercatorBounds,
    mpp: float,
) -> None:
    transform = from_origin(bounds.min_x_m, bounds.max_y_m, mpp, mpp)

    corners = {
        "corner_ul": mercator_to_latlon(bounds.min_x_m, bounds.max_y_m),
        "corner_ur": mercator_to_latlon(bounds.max_x_m, bounds.max_y_m),
        "corner_lr": mercator_to_latlon(bounds.max_x_m, bounds.min_y_m),
        "corner_ll": mercator_to_latlon(bounds.min_x_m, bounds.min_y_m),
    }

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=image_gray.shape[0],
        width=image_gray.shape[1],
        count=1,
        dtype=image_gray.dtype,
        crs="EPSG:3857",
        transform=transform,
        compress="lzw",
    ) as dataset:
        dataset.write(image_gray, 1)
        dataset.update_tags(
            bbox_min_x_m=bounds.min_x_m,
            bbox_min_y_m=bounds.min_y_m,
            bbox_max_x_m=bounds.max_x_m,
            bbox_max_y_m=bounds.max_y_m,
            corner_ul_lat=corners["corner_ul"][0],
            corner_ul_lon=corners["corner_ul"][1],
            corner_ur_lat=corners["corner_ur"][0],
            corner_ur_lon=corners["corner_ur"][1],
            corner_lr_lat=corners["corner_lr"][0],
            corner_lr_lon=corners["corner_lr"][1],
            corner_ll_lat=corners["corner_ll"][0],
            corner_ll_lon=corners["corner_ll"][1],
        )


def run_pipeline(
    center_lat_deg: float,
    center_lon_deg: float,
    radius_m: float,
    gsd_m_per_px: float,
    output_path: Path,
) -> tuple[Path, float]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mosaic, bounds, zoom, mpp = build_mosaic(
        center_lat_deg=center_lat_deg,
        center_lon_deg=center_lon_deg,
        radius_m=radius_m,
        gsd_m_per_px=gsd_m_per_px,
    )
    gray_clahe = preprocess_grayscale_clahe(mosaic)
    write_geotiff(output_path=output_path, image_gray=gray_clahe, bounds=bounds, mpp=mpp)
    
    jpg_path = output_path.with_suffix('.jpg')
    cv2.imwrite(str(jpg_path), mosaic)

    print("Map generation complete")
    print(f"Output TIF: {output_path}")
    print(f"Output JPG: {jpg_path}")
    print(f"Selected zoom: {zoom}")
    print(f"Achieved resolution: {mpp:.4f} m/px (Web Mercator units)")
    
    return jpg_path, mpp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-flight Google Maps XYZ to GeoTIFF pipeline")
    parser.add_argument("--center-lat", type=float, required=True, help="Map center latitude")
    parser.add_argument("--center-lon", type=float, required=True, help="Map center longitude")
    parser.add_argument("--radius-m", type=float, default=2000.0, help="Coverage radius in meters")
    parser.add_argument("--gsd", type=float, default=0.2, help="Target GSD in meters per pixel")
    parser.add_argument("--output", type=Path, default=Path("deployment/map_preflight_clahe.tif"), help="Output GeoTIFF path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        center_lat_deg=args.center_lat,
        center_lon_deg=args.center_lon,
        radius_m=args.radius_m,
        gsd_m_per_px=args.gsd,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
