import time

import numpy as np

from .damage import make_disaster_swath
from .graph import heal_graph_with_damage
from .routing import compute_truck_and_drone_plan


def run_latency_benchmark(G, depot, hospital, center_lat, center_lon, n_runs=10):
    graph_times, routing_times = [], []
    for i in range(n_runs):
        damage_gdf = make_disaster_swath(center_lat, center_lon, length_km=2.0, width_km=0.4, seed=i)

        t0 = time.perf_counter()
        healed, _, _ = heal_graph_with_damage(G, damage_gdf)
        graph_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        compute_truck_and_drone_plan(healed, depot, hospital)
        routing_times.append(time.perf_counter() - t0)

    return {
        "graph_update_ms": float(np.mean(graph_times) * 1000),
        "routing_ms": float(np.mean(routing_times) * 1000),
        "total_ms": float((np.mean(graph_times) + np.mean(routing_times)) * 1000),
    }
