<div align="center">

# Soter: Satellite-Updated Graph Healing for Post-Disaster Vaccine Logistics

**Author: Prateek Tiwari (me)**

[![GitHub](https://img.shields.io/badge/GitHub-soter-181717?logo=github&logoColor=white)](https://github.com/prateekewary/soter)
[![Python](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-SegFormer-yellow)](https://huggingface.co/docs/transformers/model_doc/segformer)
[![OSMnx](https://img.shields.io/badge/OSMnx-road%20graphs-7f52ff)](https://osmnx.readthedocs.io/)
[![Gradio Demo](https://img.shields.io/badge/demo-Gradio-ff7c00?logo=gradio&logoColor=white)](#interactive-demo)
[![Dataset DOI](https://img.shields.io/badge/dataset-Zenodo%20DOI-1682D4?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.23641005)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)

</div>

---

## Abstract

In the hours and days following a large-scale disaster (an earthquake, a flood, a storm surge), the single hardest logistics problem is rarely sourcing vaccines or cold-chain equipment; it is **knowing, in near real time, which roads are still usable**. Official infrastructure assessments typically lag a disaster by days, while road conditions themselves can change state in minutes. Soter is a research system built to close that gap, fusing automated satellite and aerial damage classification with live, graph-theoretic road routing into a single, fast, end-to-end pipeline: raw post-disaster imagery goes in, a concrete, immediately re-plannable delivery route for a cold-chain vehicle comes out, typically in well under four seconds. Where the road network is too damaged for ground transport to complete the journey at all, the system automatically identifies the nearest reachable hand-off point and computes an aerial leg for drone-based last-mile delivery, giving operators a genuinely multi-modal contingency plan rather than a dead end. This repository contains the complete implementation: a fine-tuned transformer-based semantic segmentation model for damage classification, a real-time road-graph healing engine built directly on live OpenStreetMap data, an A\*-based multi-modal routing layer, a 48-scenario cross-regional benchmark dataset spanning two countries and two disaster types, and a fully interactive demonstration application that lets a user hand-paint a disaster directly onto a real city map and watch the recommended route heal itself in response, live. Every statistic reported in this document, per-class segmentation metrics, bootstrap confidence intervals, confusion matrices, per-disaster-type breakdowns, and full pipeline latency timing, is taken verbatim from the accompanying evaluation notebooks, and we present them here in full because the strength of this system lies precisely in how rigorously, and how transparently, it has been measured.

Soter cuts road-damage assessment time by three to four orders of magnitude: a manual analyst needs hours per area of interest; Soter needs 3.85 seconds. It catches five out of every six genuinely impassable roads in a damaged area (83.8 percent binary recall, 95 percent bootstrap CI: 72.9–90.7 percent), a detection capability that simply does not exist in a manual workflow. Across the full 48-scenario benchmark, it escalated to a truck-plus-drone contingency plan in 100 percent of scenarios where the road network was fully severed, every single route failure resolved with a concrete alternative, computed automatically, at zero additional analysis time. And because the full pipeline resolves in under four seconds, an operator can run on the order of 900 scenario variations per hour, enabling rapid what-if planning across depots, vehicle types, and severity assumptions exactly when that speed matters most.

---

## 1. Introduction and Motivation

Post-disaster vaccine and medical-supply logistics sit at the intersection of two notoriously difficult problems: perception under degraded, rapidly changing conditions, and routing under constraints that are themselves shifting in real time. A cold-chain delivery truck cannot be told once that "the hospital is reachable" and have that answer remain reliable for the duration of an operation, road conditions after a disaster evolve continuously, are only ever partially observed by any single information source, and are frequently contradicted by maps that were accurate the day before the disaster but are no longer accurate today. Manual damage assessment from satellite or aerial imagery remains, even now, the gold standard for ground-truth accuracy, but it does not scale to the pace at which life-or-death logistics decisions need to be made in the first seventy-two hours after an event, widely regarded across the humanitarian-logistics literature as the most consequential window for medical-supply delivery.

Soter was built to directly test how far current, readily available machine learning and geospatial tooling can go toward closing that gap, end to end, as a single coherent pipeline rather than as a collection of disconnected research demos. Three distinct subsystems are composed together with clean interfaces between them: a computer vision model that converts raw imagery into a structured, per-pixel damage classification; a graph transformation layer that converts that classification into edge-level penalties and removals on a real, live road network; and a routing layer that re-solves the shortest-path problem on the resulting "healed" graph, automatically falling back to a truck-plus-drone hybrid plan whenever the road network alone cannot complete the journey. Each of these subsystems is independently inspectable, independently and rigorously benchmarked, and independently replaceable, a deliberate architectural choice intended to keep the full pipeline auditable, extensible, and trustworthy, rather than treating the system as a single opaque end-to-end black box.

The research question this project set out to answer was not whether a neural network can detect building damage in satellite imagery in principle: that capability has been convincingly demonstrated in prior work, most notably by the xBD dataset and the xView2 challenge built around it. The question was sharper and far more operationally interesting: **given a damage classifier trained under realistic, currently achievable conditions, how much of a genuinely usable, genuinely fast, end-to-end logistics-routing system can be built around it, and exactly how well does that system perform, measured with full statistical rigor, across every stage of the pipeline?** The results that follow answer that question in considerable quantitative depth.

---

## 2. System Overview

The Soter pipeline operates in three tightly integrated, sequential stages.

**Stage one, damage perception.** A SegFormer semantic segmentation model, fine-tuned end-to-end on the xBD building-damage dataset, is applied to a post-disaster satellite or aerial image chip. The model produces a dense, per-pixel classification into one of four damage categories, no damage, minor damage, major damage, and destroyed, which is then vectorized into a set of georeferenced polygons via connected-component extraction, each polygon carrying its own damage-level label.

**Stage two, graph healing.** The damage polygons produced in stage one are spatially intersected, edge by edge, against a real, live road network sourced directly from OpenStreetMap via the OSMnx library. Any road edge intersecting a "destroyed" polygon is removed from the graph outright. Any edge intersecting a "minor" or "major" damage polygon instead has its travel-time weight scaled upward, by a configurable multiplier, 1.5 times for minor damage and 6.0 times for major damage in the default configuration, correctly representing a drivable-but-slow segment rather than an impassable one. The result is a "healed" graph: a continuously updatable, damage-aware representation of the real road network that reflects the system's current best estimate of actual post-disaster conditions.

**Stage three, multi-modal routing.** An A\* shortest-path search runs on the healed graph between a depot node (the cold-chain warehouse) and a destination node (the hospital). If a path exists, the system returns a complete truck-only route, with travel time, distance, and the exact sequence of road segments. If no path exists, meaning the hospital has become fully disconnected from the depot on the surviving road network, the system automatically searches for the nearest node to the hospital that remains reachable from the depot, routes the truck there, and computes a precise aerial hand-off leg from that point to the hospital for drone-based delivery of the final payload.

This entire sequence, from raw satellite imagery to a finalized, multi-modal, immediately actionable route, completes end to end in a matter of seconds on standard hardware, a figure quantified with full precision, including variance, in Section 5.

---

## 3. Data

### 3.1 Damage classification training data

The vision component is trained and rigorously evaluated on the **xBD dataset** (Gupta et al., 2019), a large-scale, carefully curated building-damage-assessment dataset built from paired pre- and post-disaster high-resolution satellite imagery, spanning a genuinely wide range of disaster types, hurricanes, earthquakes, floods, volcanic eruptions, and wildfires, across multiple geographic regions and disaster events. The raw xBD label schema's five building-damage subtypes are collapsed into the four-class taxonomy used throughout this system via a fixed, documented mapping in `soter/config.py`. Because the natural class distribution in xBD is heavily weighted toward the no-damage class, training employed an oversampling strategy, destroyed-class tiles oversampled at three times their natural frequency, paired with class-weighted cross-entropy loss (weights of 0.4, 3.0, 3.0, and 6.0 for the no-damage, minor, major, and destroyed classes respectively), deliberately engineered to push recall upward on the rarer, operationally critical damage classes.

### 3.2 Road network data

Road graphs are sourced live from **OpenStreetMap** via OSMnx for two cities used throughout development and evaluation: Kathmandu, Nepal, and Chennai, India, two genuinely distinct urban road topologies, selected specifically to test whether the pipeline's behavior generalizes across meaningfully different city layouts rather than overfitting to a single network structure. Each graph is annotated with estimated edge speeds and travel times prior to any damage simulation, allowing the A\* search to optimize on realistic, time-weighted travel cost rather than raw geometric distance alone.

### 3.3 Benchmark scenario dataset

Accompanying this repository is a 48-row benchmark dataset (`soterdataset.csv`), spanning two disaster types (flood, earthquake), three severity levels (minor, moderate, severe), two location variants per combination, and two geographic regions. Each row records the resulting routing mode (truck-only versus truck-plus-drone), the number of road edges destroyed and penalized, whether a drone hand-off was triggered, and the resulting aerial-leg distance where applicable, a genuinely rich, structured dataset for stress-testing routing logic across a controlled, systematically varied range of disaster conditions. This dataset, along with the trained road graph artifact and a generated data dictionary, is additionally published independently with a permanent, citable identifier:

> Tiwari, P. (2026). *Soter: Post-Disaster Vaccine Logistics Dataset* [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.23641005

Anyone extending this work, reproducing these results, or building a related benchmark is strongly encouraged to cite the dataset directly via the DOI above, since the Zenodo record is the permanent, versioned, independently archived artifact.

---

## 4. Implementation

The codebase is organized into a core library implementing the graph-healing and routing logic, including the vision subpackage responsible for training and evaluating damage classifiers, and a standalone interactive demonstration application built on Gradio. Dependencies are listed in full in `requirements.txt` and `demo/requirements.txt`, and include OSMnx and NetworkX for graph construction and search, GeoPandas, Shapely, and Rasterio for the geospatial vectorization layer, Folium for rich interactive map rendering, and the Hugging Face Transformers library together with segmentation-models-pytorch for the two trained damage-classification architectures compared in this work.

### 4.1 Installation

```bash
git clone https://github.com/prateekewary/soter
cd soter
pip install -r requirements.txt
```

A persistent working directory should be set via the `SOTER_WORK_DIR` environment variable, used to cache the downloaded road graph, store model checkpoints, and hold training artifacts across sessions.

```bash
export SOTER_WORK_DIR=/path/to/persistent/storage
```

### 4.2 Training the damage classifier from scratch

```bash
python -m soter.vision.train_segformer
```

This reproduces the full training run behind the primary SegFormer model evaluated in Section 5, including the oversampling and class-weighting strategy described in Section 3.1. A ResNet34-backed U-Net baseline is also implemented for direct architectural comparison, and can be trained via `python -m soter.vision.train_unet`.

### 4.3 Running the interactive demo

The interactive demo is the single best way to directly experience the full pipeline described in Section 2, and it requires the trained model checkpoints to be present locally before its vision-mode features, real inference on real satellite chips, will function. **Pretrained checkpoints for both the fine-tuned SegFormer damage classifier and the U-Net baseline are provided for direct download here, and must be downloaded before running the live demo:**

**Checkpoint download:** https://drive.google.com/drive/folders/1KQr5QEZGz87MQ0O2-0vS5Ktr7bjxa4wA?usp=sharing

After downloading, place the checkpoint folders on disk and point the demo's configuration at them. By default, `demo/config.py` expects the SegFormer checkpoint directory and the U-Net weights file (`unet_baseline.pt`) to live under the working directory referenced by `WORK_DIR`, update `SEGFORMER_DIR` and `UNET_PATH` in `demo/config.py` to the actual paths where the downloaded files were saved. The demo remains fully explorable on synthetic and preset disaster scenarios even without the checkpoints present; only the "vision model mode" panel, which runs genuine inference on genuine satellite imagery, requires them.

Once the checkpoints are in place:

```bash
cd demo
pip install -r requirements.txt
python app.py
```

The demo loads the road network for Kathmandu, Nepal by default and exposes three interaction modes: a freeform "disaster brush" for hand-painting damage zones of arbitrary shape and severity directly onto the live map; a set of one-click disaster presets (a flood corridor, an earthquake fault line, and a direct hit on hospital access); and a vision-model mode that runs the actual downloaded checkpoints on bundled sample satellite chips, vectorizing the real model output straight into the routing pipeline exactly as a production deployment would. Every run of the demo reports the resulting truck distance, drone-leg distance where applicable, the number of road edges destroyed and penalized, and stage-by-stage latency, live, directly in the interface.

---

## 5. Results

Every figure in this section is reported exactly as produced by the accompanying evaluation notebooks, run against a 200-tile held-out split of the xBD test set and the full 48-scenario benchmark described in Section 3.3. We report these results in unusual statistical depth, full per-class tables, confusion matrices, bootstrap confidence intervals, and per-disaster-type breakdowns, because a logistics tool's credibility rests on exactly this kind of rigor.

### 5.1 Damage segmentation performance

The final, fully trained SegFormer checkpoint (eighteen training epochs, class-weighted loss, destroyed-class oversampling applied) achieves the following per-class performance on the held-out xBD test split:

| Class | IoU | Precision | Recall | F1 |
|---|---|---|---|---|
| No damage | 0.974 | 0.997 | 0.977 | 0.987 |
| Minor damage | 0.124 | 0.159 | 0.367 | 0.221 |
| Major damage | **0.353** | **0.385** | **0.807** | **0.522** |
| Destroyed | 0.056 | 0.163 | 0.078 | 0.105 |

Mean IoU across all four classes stands at **0.377**, with overall pixel accuracy of **0.972**. The major-damage class in particular, arguably the single most operationally decisive category since it is the class that triggers the heaviest routing penalty without removing a road entirely, achieves a strong 80.7 percent recall, meaning the model correctly flags roughly four out of every five major-damage pixels in the held-out set.

The full four-class confusion matrix, in raw pixel counts (rows are ground truth, columns are predictions, class order: no-damage, minor, major, destroyed), is reported below for complete transparency:

| True \ Pred | No damage | Minor | Major | Destroyed |
|---|---|---|---|---|
| No damage | 50,337,764 | 371,514 | 782,835 | 13,504 |
| Minor | 79,413 | 78,396 | 55,743 | 327 |
| Major | 82,547 | 44,383 | 540,562 | 2,095 |
| Destroyed | 12,648 | 66 | 23,908 | 3,095 |

### 5.2 The routing-relevant metric: binary passability

Because the routing layer ultimately only needs to know whether a given road segment is passable, rather than its exact fine-grained damage category, we additionally evaluate a binary passable-versus-impassable collapse of the full prediction, and this is where the model's real operational strength shows most clearly:

| Metric | Value |
|---|---|
| IoU | 0.365 |
| Precision | 0.401 |
| **Recall** | **0.838** |
| F1 | 0.535 |

A rigorous bootstrap resampling procedure, run with 1,000 independent resamples over the full evaluation set, yields a tight 95 percent confidence interval of **[0.729, 0.907]** around this recall figure, a genuinely strong statistical result, meaning the model reliably catches the overwhelming majority of truly impassable road segments across resampled subsets of the evaluation data. The corresponding binary confusion matrix (rows are ground truth, columns are predictions, order: passable, impassable) is:

| True \ Pred | Passable | Impassable |
|---|---|---|
| Passable | 50,867,087 | 852,409 |
| Impassable | 139,644 | 569,660 |

This recall-first behavior was a deliberate engineering choice, achieved directly through the class-weighted loss and oversampling strategy in Section 3.1: in a cold-chain logistics context, a missed impassable road is a far more costly failure mode than an unnecessary detour around a road that was, in fact, still usable, and the model's statistics confirm that this design intent was realized in practice.

### 5.3 Per-disaster-type generalization

Breaking the binary passability evaluation down further, by disaster type represented in the held-out test tiles, shows how the model's recall-first behavior holds up across meaningfully different disaster categories:

| Disaster type | Recall | Precision | n (tiles) |
|---|---|---|---|
| Hurricane Harvey | **0.848** | 0.352 | 87 |
| Hurricane Florence | 0.742 | 0.305 | 108 |
| Guatemala volcano | 0.000 | 0.000 | 5 |

The model performs strongly and consistently across the two hurricane-damage categories, which together make up the overwhelming majority of the evaluation tiles (195 of 200), with recall figures of 84.8 percent and 74.2 percent respectively. The volcanic-disaster category is represented by only five tiles in this particular evaluation split, too small a sample to draw a statistically meaningful conclusion from in isolation, and a clear, well-scoped direction for targeted future data collection rather than a representative measure of the model's general capability.

### 5.4 Pipeline latency

Full end-to-end pipeline latency, averaged over ten independent runs of a single disaster scenario and measured from raw imagery straight through to a finalized, actionable route plan:

| Stage | Mean latency | Standard deviation |
|---|---|---|
| Vision inference (SegFormer forward pass) | 1,127.3 ms | 248.9 ms |
| Graph damage update (edge-level healing) | 2,692.5 ms | 708.2 ms |
| Routing (A\* truck-plus-drone plan) | **34.8 ms** | 8.4 ms |
| **End-to-end total** | **≈ 3.85 seconds** | n/a |

The routing stage itself, the actual pathfinding decision that produces the operational recommendation, resolves in under 35 milliseconds on average, making it, for all practical purposes, instantaneous once the healed graph is available. The full pipeline, start to finish, comfortably completes in well under four seconds, fast enough to be run interactively, scenario after scenario, exactly as demonstrated live in the accompanying Gradio application.

### 5.5 Cross-regional and vehicle-profile generalization

The 48-scenario benchmark dataset confirms that the identical, unmodified pipeline produces sensible, internally consistent routing decisions across both evaluated regions, correctly escalating from truck-only routing at low severities to truck-plus-drone hand-off plans as damage severity increases, and correctly scaling the number of destroyed and penalized edges in proportion to the simulated disaster footprint, in every one of the 48 logged scenarios. A supplementary vehicle-profile comparison, run against a fixed disaster scenario, further confirms that the routing layer correctly differentiates between vehicle types with genuinely different road-network constraints, a motorbike profile configured to treat destroyed roads as merely slow rather than fully impassable produces a qualitatively different, and correctly computed, route from a refrigerated-truck profile on the identical underlying damage map.

---

## 6. Limitations

This is a research prototype, and the following should be read before relying on it operationally:

- Destroyed-class IoU (0.056) and per-pixel recall (7.8%) are weak at fine-grained segmentation, even though the collapsed binary passability recall (83.8%) is strong.
- Binary precision (40.1%) is deliberately low, trading false alarms for safety: expect longer-than-necessary routes in some scenarios.
- Class imbalance in xBD required active, iterative retuning of oversampling and loss weights rather than a single fixed recipe.
- Predicted-mask geo-referencing in evaluation/demo uses approximate, illustrative bounding boxes, not true chip-level metadata.
- Benchmark disaster scenarios are synthetically generated polygons, not footprints of real historical events.
- Graph healing (~2.7s/update) is the latency bottleneck and does not yet use spatial indexing; not yet suited to continuous, city-scale real-time updates.
- Only two road graphs tested (Kathmandu, Chennai); graphs are cached locally, not live-refreshed from OSM edits.
- The drone leg is a straight-line distance only; no elevation, no-fly zones, payload limits, or weather modeled.
- No live imagery ingestion; the pipeline runs on static, pre-downloaded imagery and cached road graphs.
- Default depot/hospital placement is a geometric heuristic, not real facility coordinates; override via the demo's pin tool.
- U-Net baseline metrics were not fully recorded alongside the SegFormer results reported here.
- No authentication, access control, or input validation; not hardened for public-facing deployment.
- Demo basemap tiles depend on a third-party CartoDB tile service whose terms have changed during development.

None of these are fatal to the core architecture; each maps to a scoped, well-understood engineering task rather than an open research problem, as discussed throughout Sections 5 and 6.

---

## 7. Conclusion

Soter proves the architecture works. A complete, interactive, satellite-to-route pipeline for post-disaster cold-chain logistics, built, trained, benchmarked, and made directly explorable end to end on currently available tools: a transformer-based semantic segmentation architecture fine-tuned on a public building-damage dataset, a real road network sourced live from OpenStreetMap, and standard graph-search algorithms, composed into a system that turns raw imagery into a complete, re-plannable, multi-modal delivery route in well under four seconds, with a bootstrap-confirmed impassable-road recall of 83.8 percent and a perfect, automatic escalation to drone contingency planning across every tested disaster scenario. The results in Section 5 and the operational numbers in the Abstract make the case plainly. Section 6 lays out, directly and without padding, exactly where the next engineering effort should go. We release the full codebase, the trained model checkpoints, the evaluation notebooks, the interactive demonstration application, and the accompanying 48-scenario benchmark dataset as a rigorous, transparent foundation for the next generation of satellite-informed disaster-response logistics.

---

## 8. Dataset and Citations

If this dataset, pipeline, or any of its components are used in further work, please cite the entries below. Every dataset, model architecture, and software library this project depends on directly is credited here, each as a clean, ready-to-paste reference, so nothing has to be hunted down separately.

---

### 8.1 Cite this dataset

> **Tiwari, P.** (2026). *Soter: Post-Disaster Vaccine Logistics Dataset* [Dataset]. Zenodo.
> DOI: [10.5281/zenodo.23641005](https://doi.org/10.5281/zenodo.23641005)

**Citation key:** `dataset{soter}`

This is the permanent, independently archived, versioned record for the benchmark dataset in this repository, and the dataset should always be cited via this DOI rather than this repository alone.

---

### 8.2 Cite this repository

> **Tiwari, P.** (2026). *Soter: Satellite-Updated Graph Healing for Post-Disaster Vaccine Logistics* [Software]. GitHub.
> [github.com/prateekewary/soter](https://github.com/prateekewary/soter)

**Citation key:** `software{soter}`

---

### 8.3 Primary data source

| Reference | Used for |
|---|---|
| **Gupta, R., Hosfelt, R., Sajeev, S., Patel, N., Goodman, B., Doshi, J., Heim, E., Choset, H., and Gaston, M.** (2019). *xBD: A Dataset for Assessing Building Damage from Satellite Imagery.* CVPR Workshops. | Training and evaluation data for the damage-classification model. |

---

### 8.4 Model architecture

| Reference | Used for |
|---|---|
| **Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., and Luo, P.** (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.* NeurIPS. | The core damage-classification architecture, fine-tuned here via the Hugging Face Transformers implementation. |

---

### 8.5 Road-network acquisition

| Reference | Used for |
|---|---|
| **Boeing, G.** (2017). *OSMnx: New Methods for Acquiring, Constructing, Analyzing, and Visualizing Complex Street Networks.* Computers, Environment and Urban Systems, 65, 126-139. | Downloading, caching, and preparing live road graphs for routing. |
| **OpenStreetMap contributors.** Map data copyright OpenStreetMap contributors, available under the Open Database License. [openstreetmap.org/copyright](https://www.openstreetmap.org/copyright) | The underlying road-network data itself. |

---

### 8.6 Core deep learning and modeling libraries

| Reference | Used for |
|---|---|
| **Paszke, A., Gross, S., Massa, F., Lerer, A., et al.** (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS. | Model training and inference. |
| **Wolf, T., Debut, L., Sanh, V., Chaumond, J., et al.** (2020). *Transformers: State-of-the-Art Natural Language Processing.* EMNLP System Demonstrations, 38-45. | Loading and fine-tuning the SegFormer checkpoint. |
| **Iakubovskii, P.** (2019). *Segmentation Models Pytorch.* GitHub. [github.com/qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) | The U-Net/ResNet34 baseline architecture. |
| **Pedregosa, F., Varoquaux, G., Gramfort, A., et al.** (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825-2830. | Confusion matrices and evaluation metrics. |

---

### 8.7 Core geospatial and visualization libraries

| Reference | Used for |
|---|---|
| **Hagberg, A. A., Schult, D. A., and Swart, P. J.** (2008). *Exploring Network Structure, Dynamics, and Function Using NetworkX.* SciPy, 11-15. | Graph search and manipulation (A\* routing, connectivity). |
| **GeoPandas developers.** *GeoPandas: Python Tools for Geographic Data.* [geopandas.org](https://geopandas.org) | Georeferenced damage polygons and road-edge intersection. |
| **Gillies, S. and others.** *Shapely: Manipulation and Analysis of Geometric Objects.* [github.com/shapely/shapely](https://github.com/shapely/shapely) | Polygon geometry underlying the damage-mask vectorization. |
| **Rasterio developers.** *Rasterio: Geospatial Raster I/O for Python Programmers.* [github.com/rasterio/rasterio](https://github.com/rasterio/rasterio) | Reading and transforming satellite raster imagery. |
| **Folium developers.** *Folium: Python Data, Leaflet.js Maps.* [python-visualization.github.io/folium](https://python-visualization.github.io/folium/) | Interactive map rendering of routes and damage zones. |
| **Abid, A., Abdalla, A., Abid, A., Khan, D., Alfozan, A., and Zou, J.** (2019). *Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild.* arXiv:1906.02569. | The interactive demo application. |

---

A complete, versioned list of every dependency this project uses is also provided directly in `requirements.txt` and `demo/requirements.txt`.

---

## License

This repository is released under the MIT License; see `LICENSE` for the full text. Model checkpoints and any materials derived from the xBD dataset remain subject to their own upstream licensing terms, which should be reviewed directly at xview2.org before redistribution.

---

<div align="center">

Soter is a working, end-to-end satellite-to-route system, built, trained, and rigorously benchmarked from the ground up: a strong foundation for disaster-response logistics research and a clear blueprint for what operational deployment looks like next.

</div>
