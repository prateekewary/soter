import os

WORK_DIR = os.environ.get("SOTER_WORK_DIR", "/content/drive/MyDrive/Soler")
GRAPH_CACHE = f"{WORK_DIR}/road_graph.graphml"
OSMNX_CACHE = f"{WORK_DIR}/osmnx_cache"

XBD_ROOT = f"{WORK_DIR}/xbd"
XBD_RAW_DIR = f"{WORK_DIR}/xbd_raw"

SEGFORMER_CKPT = "nvidia/segformer-b0-finetuned-ade-512-512"
SEGFORMER_OUT_DIR = f"{WORK_DIR}/segformer-xbd-v2"
SEGFORMER_FINAL_DIR = f"{WORK_DIR}/segformer-xbd-v2-final"
UNET_PATH = f"{WORK_DIR}/unet_baseline.pt"

PLACE_NAME = "Kathmandu, Nepal"

NUM_LABELS = 4
CLASS_NAMES = ["no-damage", "minor", "major", "destroyed"]

XBD_SUBTYPE_TO_LEVEL = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
    "un-classified": 0,
}

DAMAGE_WEIGHT_MULTIPLIER = {
    0: 1.0,
    1: 1.5,
    2: 6.0,
    3: float("inf"),
}

DAMAGE_COLORS = {
    0: "#2ecc71",
    1: "#f1c40f",
    2: "#e67e22",
    3: "#e74c3c",
}

CLASS_WEIGHTS = [0.4, 3.0, 3.0, 6.0]
DESTROYED_OVERSAMPLE = 3
MAJOR_OVERSAMPLE = 1
