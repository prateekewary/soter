import math

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon


def _swath(center_lat, center_lon, length_km, width_km, bearing_deg, levels):
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(center_lat))
    dx = math.sin(math.radians(bearing_deg))
    dy = math.cos(math.radians(bearing_deg))
    n_segments = len(levels)
    seg_len = length_km / n_segments
    polys = []
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
        polys.append(poly)
    return gpd.GeoDataFrame({"damage_level": levels}, geometry=polys, crs="EPSG:4326")


def flood_corridor(center_lat, center_lon):
    return _swath(center_lat, center_lon, length_km=3.0, width_km=0.5, bearing_deg=70, levels=[2, 3, 3, 2])


def earthquake_fault(center_lat, center_lon):
    return _swath(center_lat, center_lon, length_km=2.2, width_km=0.25, bearing_deg=10, levels=[3, 3, 3])


def hospital_access_hit(center_lat, center_lon):
    return _swath(center_lat, center_lon, length_km=0.8, width_km=0.6, bearing_deg=0, levels=[3])


PRESETS = {
    "Flood corridor": flood_corridor,
    "Earthquake fault line": earthquake_fault,
    "Direct hit on hospital access": hospital_access_hit,
}
