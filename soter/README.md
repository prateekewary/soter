# Soter

Satellite-updated graph healing for post-disaster vaccine logistics. Fuses a fine-tuned SegFormer damage classifier with A* graph routing to reroute a cold-chain truck around disaster damage, handing off to a drone when the road network is severed.

## Structure

- `soter/` — core library (graph healing, routing, damage polygons, visualization, benchmarking)
- `soter/vision/` — SegFormer/U-Net training, inference, evaluation on xBD
- `demo/` — Gradio interactive demo (disaster brush -> live reroute)

## Setup

```
pip install -r requirements.txt
```

Set `SOTER_WORK_DIR` to a persistent folder (Drive path in Colab) for cached graphs and model checkpoints.

## Training

```
python -m soter.vision.train_segformer
```

## Demo

```
cd demo
pip install -r requirements.txt
python app.py
```
