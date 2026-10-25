import math

import geopandas as gpd
import numpy as np
import rasterio.transform
from rasterio.features import shapes as rio_shapes
from shapely.geometry import Polygon, shape


def make_disaster_swath(center_lat, center_lon, length_km=2.0, width_km=0.4, bearing_deg=30, n_segments=3, seed=0):
    rng = np.random.default_rng(seed)
    polys, levels = [], []
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(center_lat))
    dx = math.sin(math.radians(bearing_deg))
    dy = math.cos(math.radians(bearing_deg))
    seg_len = length_km / n_segments
    for i in range(n_segments):
        t0, t1 = i * seg_len, (i + 1) * seg_len
        lat0 = center_lat + (dy * t0) / km_per_deg_lat
        lon0 = center_lon + (dx * t0) / km_per_deg_lon
        lat1 = center_lat + (dy * t1) / km_per_deg_lat
        lon1 = center_lon + (dx * t1) / km_per_deg_lon
        half_w_lat = (width_km / 2) / km_per_deg_lat
        half_w_lon = (width_km / 2) / km_per_deg_lon
        perp_dx, perp_dy = -dy, dx
        poly = Polygon([
            (lon0 + perp_dx * half_w_lon, lat0 + perp_dy * half_w_lat),
            (lon1 + perp_dx * half_w_lon, lat1 + perp_dy * half_w_lat),
            (lon1 - perp_dx * half_w_lon, lat1 - perp_dy * half_w_lat),
            (lon0 - perp_dx * half_w_lon, lat0 - perp_dy * half_w_lat),
        ])
        level = int(np.clip(3 - rng.integers(0, 2) - i // max(1, n_segments - 2), 0, 3))
        polys.append(poly)
        levels.append(level)
    return gpd.GeoDataFrame({"damage_level": levels}, geometry=polys, crs="EPSG:4326")


def mask_to_geodataframe(mask, geo_bounds, crs="EPSG:4326"):
    h, w = mask.shape
    minx, miny, maxx, maxy = geo_bounds
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, w, h)
    polys, levels = [], []
    for geom, val in rio_shapes(mask, mask=mask > 0, transform=transform):
        if val == 0:
            continue
        polys.append(shape(geom))
        levels.append(int(val))
    if not polys:
        return gpd.GeoDataFrame({"damage_level": []}, geometry=[], crs=crs)
    return gpd.GeoDataFrame({"damage_level": levels}, geometry=polys, crs=crs)
