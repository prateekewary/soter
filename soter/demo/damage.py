import geopandas as gpd
import numpy as np
import rasterio.transform
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape

import config


def mask_to_geodataframe(mask, bounds, crs="EPSG:4326"):
    h, w = mask.shape
    minx, miny, maxx, maxy = bounds
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


def classify_brush_layer(layer_rgba):
    arr = np.array(layer_rgba)
    if arr.shape[-1] < 4:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    alpha = arr[..., 3]
    rgb = arr[..., :3].astype(int)
    painted = alpha > 10
    mask = np.zeros(alpha.shape, dtype=np.uint8)
    for level, color in config.BRUSH_LEVEL_RGB.items():
        dist = np.abs(rgb - np.array(color)).sum(axis=-1)
        close = painted & (dist < 90)
        mask[close] = level
    return mask
