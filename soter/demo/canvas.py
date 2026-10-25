import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import osmnx as ox
from PIL import Image

import config


def render_base_image(G):
    w, h = config.CANVAS_WIDTH, config.CANVAS_HEIGHT
    fig, ax = ox.plot_graph(
        G,
        show=False,
        close=False,
        bgcolor="#f4f1ea",
        node_size=0,
        edge_color="#555555",
        edge_linewidth=1.0,
        figsize=(w / 100, h / 100),
        dpi=100,
    )
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), buf, "raw", "RGBA", 0, 1)
    img = img.convert("RGB").resize((w, h))
    plt.close(fig)
    bounds = (xlim[0], ylim[0], xlim[1], ylim[1])
    return img, bounds


def pixel_to_lonlat(px, py, bounds):
    minx, miny, maxx, maxy = bounds
    lon = minx + (px / config.CANVAS_WIDTH) * (maxx - minx)
    lat = maxy - (py / config.CANVAS_HEIGHT) * (maxy - miny)
    return lat, lon


def lonlat_to_pixel(lat, lon, bounds, width, height):
    """Inverse of pixel_to_lonlat, parameterized by image size so it lines up
    with whatever the damage mask/geometry was actually built from (e.g. a
    chip's native resolution), not just the fixed CANVAS_WIDTH/HEIGHT."""
    minx, miny, maxx, maxy = bounds
    px = (lon - minx) / (maxx - minx) * width
    py = (maxy - lat) / (maxy - miny) * height
    return px, py