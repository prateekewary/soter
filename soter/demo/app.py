import time

import gradio as gr
import numpy as np
import osmnx as ox
from PIL import Image

import config
import models
import scenarios
import stats
from canvas import pixel_to_lonlat, render_base_image
from damage import classify_brush_layer, mask_to_geodataframe
from graph_utils import build_edges_gdf, heal_graph_with_damage, load_graph, pick_depot_hospital
from mapping import render_mask_on_chip, render_result_image
from routing import compute_truck_and_drone_plan

G = load_graph()
EDGES_GDF = build_edges_gdf(G)
default_depot, default_hospital = pick_depot_hospital(G)
base_image, bounds = render_base_image(G)
center_lat = G.nodes[default_hospital]["y"]
center_lon = G.nodes[default_hospital]["x"]

empty_editor = {"background": base_image, "layers": [], "composite": base_image}


def run_pipeline(damage_gdf, depot, hospital, severity):
    t0 = time.perf_counter()
    healed, removed, penalized = heal_graph_with_damage(G, damage_gdf, edges_gdf=EDGES_GDF, severity=severity)
    heal_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    plan = compute_truck_and_drone_plan(healed, depot, hospital)
    route_ms = (time.perf_counter() - t0) * 1000

    healed_img = render_result_image(healed, damage_gdf, plan, depot, hospital, bounds)
    baseline_plan = compute_truck_and_drone_plan(G, depot, hospital)
    baseline_img = render_result_image(G, damage_gdf.iloc[0:0], baseline_plan, depot, hospital, bounds)

    status = stats.build_stats(G, healed, plan, removed, penalized, heal_ms, route_ms)
    return baseline_img, healed_img, status


def heal_and_reroute(editor_value, depot, hospital, severity):
    if editor_value is None or not editor_value.get("layers"):
        return base_image, base_image, "Paint a disaster zone to begin."

    level_mask = classify_brush_layer(editor_value["layers"][0])
    if not np.any(level_mask):
        return base_image, base_image, "Paint a disaster zone to begin."

    damage_gdf = mask_to_geodataframe(level_mask, bounds)
    if len(damage_gdf) == 0:
        return base_image, base_image, "Paint a disaster zone to begin."

    return run_pipeline(damage_gdf, depot, hospital, severity)


def apply_preset(preset_name, depot, hospital, severity):
    fn = scenarios.PRESETS.get(preset_name)
    if fn is None:
        return empty_editor, base_image, base_image, "Pick a scenario preset."
    damage_gdf = fn(center_lat, center_lon)
    baseline_img, healed_img, status = run_pipeline(damage_gdf, depot, hospital, severity)
    return empty_editor, baseline_img, healed_img, status


def reset_all(depot, hospital):
    return empty_editor, base_image, base_image, "Paint a disaster zone or pick a preset to begin."


def set_pin(pin_mode, depot, hospital, evt: gr.SelectData):
    px, py = evt.index
    lat, lon = pixel_to_lonlat(px, py, bounds)
    node = ox.distance.nearest_nodes(G, X=lon, Y=lat)
    if pin_mode == "Depot":
        depot = node
    else:
        hospital = node
    return depot, hospital, f"{pin_mode} set to node {node}."


def run_vision_mode(chip_name, model_name):
    import os
    import traceback

    if not chip_name:
        return None, None, "No sample chip selected."

    chip_path = os.path.join(config.SAMPLE_CHIP_DIR, chip_name)

    try:
        image = Image.open(chip_path).convert("RGB")
        original_view = image.resize((config.CANVAS_WIDTH, config.CANVAS_HEIGHT))

        t0 = time.perf_counter()
        mask = models.predict_damage_mask(image, model_name)
        infer_ms = (time.perf_counter() - t0) * 1000

        damage_gdf = mask_to_geodataframe(mask, bounds)
        if len(damage_gdf) == 0:
            return original_view, original_view, (
                f"No damage detected by {model_name} on this chip.\n\n"
                f"**Inference time:** {infer_ms:.1f} ms"
            )

        mask_img = render_mask_on_chip(image, damage_gdf, bounds)

        counts = damage_gdf["damage_level"].value_counts().to_dict()
        status = (
            f"**Model:** {model_name}\n\n"
            f"**Inference time:** {infer_ms:.1f} ms\n\n"
            f"**Regions detected:** minor {counts.get(1, 0)}, "
            f"major {counts.get(2, 0)}, destroyed {counts.get(3, 0)}"
        )
        return original_view, mask_img, status
    except Exception:
        return None, None, f"Error running {model_name}:\n\n{traceback.format_exc()}"


with gr.Blocks(title="Soter") as demo:
    gr.Markdown("# Soter: Paint a Disaster, Watch the Route Heal")
    gr.Markdown(
        "Yellow = minor damage, orange = major damage, red = destroyed. "
        "Paint a zone, pick a preset, or run a real satellite chip through the vision model."
    )

    depot_state = gr.State(default_depot)
    hospital_state = gr.State(default_hospital)

    with gr.Row():
        with gr.Column(scale=1):
            editor = gr.ImageEditor(
                value=empty_editor,
                type="pil",
                brush=gr.Brush(colors=["#f1c40f", "#e67e22", "#e74c3c"], default_size=18),
                height=config.CANVAS_HEIGHT,
                width=config.CANVAS_WIDTH,
                label="Disaster Brush",
            )

            with gr.Row():
                preset_dd = gr.Dropdown(choices=list(scenarios.PRESETS.keys()), label="Scenario preset")
                preset_btn = gr.Button("Run preset")
            with gr.Row():
                reset_btn = gr.Button("Reset")

            severity_slider = gr.Slider(0.25, 3.0, value=1.0, step=0.25, label="Disaster severity multiplier")

            gr.Markdown("### Place depot and hospital")
            pin_radio = gr.Radio(["Depot", "Hospital"], value="Depot", label="Pin mode")
            pin_picker = gr.Image(value=base_image, height=300, width=300, label="Click map to place pin")
            pin_status = gr.Markdown("")

            gr.Markdown("### Vision model mode")
            _ckpt = models.checkpoint_status()
            gr.Markdown(
                f"SegFormer checkpoint: {'found' if _ckpt['segformer'] else 'NOT FOUND at ' + config.SEGFORMER_DIR}\n\n"
                f"U-Net checkpoint: {'found' if _ckpt['unet'] else 'NOT FOUND at ' + config.UNET_PATH}"
            )
            chip_dd = gr.Dropdown(choices=models.list_sample_chips(), label="Sample satellite chip")
            model_radio = gr.Radio(["SegFormer v2", "U-Net Baseline"], value="SegFormer v2", label="Model")
            vision_btn = gr.Button("Run model on chip")

            with gr.Row():
                vision_baseline_image = gr.Image(height=300, width=300, label="Chip: original")
                vision_output_image = gr.Image(height=300, width=300, label="Chip: predicted damage mask")
            vision_status_box = gr.Markdown("Run the model to see the chip and its predicted damage mask here.")

            gr.Markdown(
                "### Damage taxonomy\n\n"
                "| Level | Meaning | Routing effect |\n"
                "|---|---|---|\n"
                "| 0 | No damage | No change |\n"
                "| 1 | Minor | Small time penalty |\n"
                "| 2 | Major | Heavy time penalty |\n"
                "| 3 | Destroyed | Road removed |\n"
            )

        with gr.Column(scale=1):
            with gr.Row():
                baseline_image = gr.Image(value=base_image, height=420, width=420, label="Baseline (no disaster)")
                output_image = gr.Image(value=base_image, height=420, width=420, label="Healed route")
            status_box = gr.Markdown("Paint a disaster zone or pick a preset to begin.")

    editor.change(
        fn=heal_and_reroute,
        inputs=[editor, depot_state, hospital_state, severity_slider],
        outputs=[baseline_image, output_image, status_box],
    )

    preset_btn.click(
        fn=apply_preset,
        inputs=[preset_dd, depot_state, hospital_state, severity_slider],
        outputs=[editor, baseline_image, output_image, status_box],
    )

    reset_btn.click(
        fn=reset_all,
        inputs=[depot_state, hospital_state],
        outputs=[editor, baseline_image, output_image, status_box],
    )

    pin_picker.select(
        fn=set_pin,
        inputs=[pin_radio, depot_state, hospital_state],
        outputs=[depot_state, hospital_state, pin_status],
    )

    vision_btn.click(
        fn=run_vision_mode,
        inputs=[chip_dd, model_radio],
        outputs=[vision_baseline_image, vision_output_image, vision_status_box],
    )

if __name__ == "__main__":
    demo.launch(share=True)
