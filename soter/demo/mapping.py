import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from PIL import Image, ImageDraw

import config
from canvas import lonlat_to_pixel

_DAMAGE_RGBA = {
    0: (46, 204, 113, 110),
    1: (241, 196, 15, 140),
    2: (230, 126, 34, 150),
    3: (231, 76, 60, 160),
}


def render_mask_on_chip(chip_image, damage_gdf, bounds):
    """Vision mode only: overlay the predicted damage mask on the chip's own
    image. No route, no depot/hospital, no road graph -- just the chip and
    what the model found on it."""
    w, h = chip_image.size
    base = chip_image.convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for _, row in damage_gdf.iterrows():
        color = _DAMAGE_RGBA.get(int(row["damage_level"]), (153, 153, 153, 120))
        xs, ys = row["geometry"].exterior.xy
        pts = [lonlat_to_pixel(y, x, bounds, w, h) for x, y in zip(xs, ys)]
        draw.polygon(pts, fill=color, outline=(0, 0, 0, 200))

    composited = Image.alpha_composite(base, overlay).convert("RGB")
    cw, ch = config.CANVAS_WIDTH, config.CANVAS_HEIGHT
    return composited.resize((cw, ch))


def render_result_on_chip(chip_image, damage_gdf, plan, depot, hospital, G, bounds):
    """Draw the predicted damage + route on top of the chip's own pixels.

    Vision mode's bounds are only ever the affine box used to rasterize the
    model's mask (via damage.mask_to_geodataframe) -- they are not a real
    geocoded location. Plotting them against the default road-network graph
    (e.g. Kathmandu) made the result look like it ran on the wrong city.
    Drawing directly on the chip sidesteps that: the polygons line up with
    this exact image because they came from an affine transform of it.
    """
    w, h = chip_image.size
    base = chip_image.convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for _, row in damage_gdf.iterrows():
        color = _DAMAGE_RGBA.get(int(row["damage_level"]), (153, 153, 153, 120))
        xs, ys = row["geometry"].exterior.xy
        pts = [lonlat_to_pixel(y, x, bounds, w, h) for x, y in zip(xs, ys)]
        draw.polygon(pts, fill=color, outline=(0, 0, 0, 200))

    if plan.get("truck_route"):
        pts = [
            lonlat_to_pixel(G.nodes[n]["y"], G.nodes[n]["x"], bounds, w, h)
            for n in plan["truck_route"]
        ]
        if len(pts) > 1:
            draw.line(pts, fill=(41, 128, 185, 255), width=4)

    if plan.get("drone_leg"):
        launch = plan["drone_leg"]["launch_latlon"]
        hosp = plan["drone_leg"]["hospital_latlon"]
        lp = lonlat_to_pixel(launch[0], launch[1], bounds, w, h)
        hp = lonlat_to_pixel(hosp[0], hosp[1], bounds, w, h)
        draw.line([lp, hp], fill=(192, 57, 43, 255), width=3)
        r = 6
        draw.ellipse([lp[0] - r, lp[1] - r, lp[0] + r, lp[1] + r], fill=(230, 126, 34, 255))

    for node, color in [(depot, (41, 128, 185, 255)), (hospital, (192, 57, 43, 255))]:
        if node in G.nodes:
            px, py = lonlat_to_pixel(G.nodes[node]["y"], G.nodes[node]["x"], bounds, w, h)
            r = 7
            draw.rectangle([px - r, py - r, px + r, py + r], fill=color)

    composited = Image.alpha_composite(base, overlay).convert("RGB")
    cw, ch = config.CANVAS_WIDTH, config.CANVAS_HEIGHT
    return composited.resize((cw, ch))


def render_result_image(G, damage_gdf, plan, depot, hospital, bounds):
    w, h = config.CANVAS_WIDTH, config.CANVAS_HEIGHT
    fig, ax = ox.plot_graph(
        G,
        show=False,
        close=False,
        bgcolor="#f4f1ea",
        node_size=0,
        edge_color="#999999",
        edge_linewidth=0.8,
        figsize=(w / 100, h / 100),
        dpi=100,
    )
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    for _, row in damage_gdf.iterrows():
        color = config.DAMAGE_COLORS.get(row["damage_level"], "#999999")
        xs, ys = row["geometry"].exterior.xy
        ax.fill(xs, ys, color=color, alpha=0.45, zorder=3)

    if plan["truck_route"]:
        xs = [G.nodes[n]["x"] for n in plan["truck_route"]]
        ys = [G.nodes[n]["y"] for n in plan["truck_route"]]
        ax.plot(xs, ys, color="#2980b9", linewidth=4, zorder=5)

    if plan["drone_leg"]:
        launch = plan["drone_leg"]["launch_latlon"]
        hosp = plan["drone_leg"]["hospital_latlon"]
        ax.plot([launch[1], hosp[1]], [launch[0], hosp[0]], color="#c0392b", linewidth=3, linestyle="--", zorder=6)
        ax.scatter([launch[1]], [launch[0]], color="#e67e22", s=80, zorder=7, marker="^")

    ax.scatter([G.nodes[depot]["x"]], [G.nodes[depot]["y"]], color="#2980b9", s=90, zorder=8, marker="s")
    ax.scatter([G.nodes[hospital]["x"]], [G.nodes[hospital]["y"]], color="#c0392b", s=110, zorder=8, marker="P")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((w, h))
    return img