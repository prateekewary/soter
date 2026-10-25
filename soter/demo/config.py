import os

WORK_DIR = "/content/drive/MyDrive/Soler"
GRAPH_CACHE = f"{WORK_DIR}/road_graph.graphml"
SEGFORMER_DIR = f"{WORK_DIR}/segformer-xbd-v2-final"
UNET_PATH = f"{WORK_DIR}/unet_baseline.pt"

PLACE_NAME = "Kathmandu, Nepal"

SAMPLE_CHIP_DIR = os.path.join(os.path.dirname(__file__), "assets", "sample_chips")

SEGFORMER_RECALL = 0.838
SEGFORMER_PRECISION = 0.461
UNET_RECALL = None
UNET_PRECISION = None

CANVAS_WIDTH = 900
CANVAS_HEIGHT = 900

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

BRUSH_LEVEL_RGB = {
    1: (241, 196, 15),
    2: (230, 126, 34),
    3: (231, 76, 60),
}

OSMNX_CACHE = os.path.join(WORK_DIR, "osmnx_cache")