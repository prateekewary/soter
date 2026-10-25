import glob
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from shapely import wkt as shapely_wkt

from .. import config


class XBDSegDataset(torch.utils.data.Dataset):
    def __init__(self, root, processor, split="train", image_paths=None, max_items=None):
        self.processor = processor
        if image_paths is not None:
            self.image_paths = image_paths
        else:
            pattern = os.path.join(root, split, "images", "*post_disaster.png")
            self.image_paths = sorted(glob.glob(pattern))
            if max_items:
                self.image_paths = self.image_paths[:max_items]
        self.label_dir = os.path.join(root, split, "labels")

    def __len__(self):
        return len(self.image_paths)

    def _mask_from_json(self, img_path, size):
        label_path = img_path.replace("images", "labels").replace(".png", ".json")
        mask = np.zeros(size[::-1], dtype=np.uint8)
        if not os.path.exists(label_path):
            return mask
        with open(label_path) as f:
            lbl = json.load(f)
        for feat in lbl.get("features", {}).get("xy", []):
            subtype = feat["properties"].get("subtype", "no-damage")
            level = config.XBD_SUBTYPE_TO_LEVEL.get(subtype, 0)
            poly = shapely_wkt.loads(feat["wkt"])
            pts = np.array(poly.exterior.coords, dtype=np.int32)
            cv2.fillPoly(mask, [pts], level)
        return mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        mask = self._mask_from_json(img_path, image.size)
        enc = self.processor(image, Image.fromarray(mask), return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}


def tile_contains_classes(label_path, target_subtypes):
    if not os.path.exists(label_path):
        return False
    with open(label_path) as f:
        lbl = json.load(f)
    for feat in lbl.get("features", {}).get("xy", []):
        if feat["properties"].get("subtype", "no-damage") in target_subtypes:
            return True
    return False


def build_oversampled_paths(root, split="train"):
    all_image_paths = sorted(glob.glob(os.path.join(root, split, "images", "*post_disaster.png")))
    destroyed_tiles, major_tiles, other_tiles = [], [], []
    for img_path in all_image_paths:
        label_path = img_path.replace("images", "labels").replace(".png", ".json")
        if tile_contains_classes(label_path, {"destroyed"}):
            destroyed_tiles.append(img_path)
        elif tile_contains_classes(label_path, {"major-damage"}):
            major_tiles.append(img_path)
        else:
            other_tiles.append(img_path)

    paths = (
        destroyed_tiles * config.DESTROYED_OVERSAMPLE
        + major_tiles * config.MAJOR_OVERSAMPLE
        + other_tiles
    )
    import random

    random.shuffle(paths)
    return paths
