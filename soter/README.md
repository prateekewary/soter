# Soter

Satellite-updated graph healing for post-disaster vaccine logistics. Fuses a fine-tuned SegFormer damage classifier with A* graph routing to reroute a cold-chain truck around disaster damage, handing off to a drone when the road network is severed.

## Structure

- `soter/` — core library (graph healing, routing, damage polygons, visualization, benchmarking)
- `soter/vision/` — SegFormer/U-Net training, inference, evaluation on xBD
- `demo/` — Gradio interactive demo (disaster brush -> live reroute)

## Setup

```bash
git clone https://github.com/prateekewary/soter
cd soter
pip install -r requirements.txt
```

Set `SOTER_WORK_DIR` to a persistent folder (e.g. a Drive path in Colab) for cached road graphs and model checkpoints:

```bash
export SOTER_WORK_DIR=/path/to/persistent/storage
```

### Demo checkpoints

The demo's vision-model mode needs the fine-tuned SegFormer and U-Net baseline checkpoints on disk — download them here:

https://drive.google.com/drive/folders/1KQr5QEZGz87MQ0O2-0vS5Ktr7bjxa4wA?usp=sharing

`demo/config.py` currently points at:

```python
WORK_DIR = "/content/drive/MyDrive/Soler"
SEGFORMER_DIR = f"{WORK_DIR}/segformer-xbd-v2-final"
UNET_PATH = f"{WORK_DIR}/unet_baseline.pt"
```

Either place the downloaded checkpoints at those exact paths, or edit `SEGFORMER_DIR` / `UNET_PATH` (and `WORK_DIR`, `GRAPH_CACHE`, `OSMNX_CACHE`) in `demo/config.py` to match wherever you saved them. The demo loads the road network for **Kathmandu, Nepal** by default (`PLACE_NAME` in `demo/config.py`) — change that too if you want to switch cities.

Without checkpoints present, the demo still runs fully on synthetic/preset disaster scenarios (disaster brush + one-click presets) — only the vision-model panel (real inference on real satellite chips) is disabled.

## Training

```bash
python -m soter.vision.train_segformer
```

A ResNet34 U-Net baseline can be trained with `python -m soter.vision.train_unet`.

## Demo

```bash
cd demo
pip install -r requirements.txt
python app.py
```

`demo/requirements.txt`: gradio, osmnx, networkx==3.3, geopandas, shapely, rasterio, matplotlib, numpy, pillow, torch, torchvision, transformers, segmentation-models-pytorch.
