import math
from math import atan2, cos, radians, sin, sqrt

import networkx as nx

from .graph import astar_route


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def compute_truck_and_drone_plan(graph, depot, hospital):
    route = astar_route(graph, depot, hospital)
    if route is not None:
        return {"mode": "truck_only", "truck_route": route, "drone_leg": None}

    reachable = nx.node_connected_component(graph.to_undirected(), depot)
    hosp_lat, hosp_lon = graph.nodes[hospital]["y"], graph.nodes[hospital]["x"]

    best_node, best_dist = None, math.inf
    for n in reachable:
        d = haversine_km(graph.nodes[n]["y"], graph.nodes[n]["x"], hosp_lat, hosp_lon)
        if d < best_dist:
            best_dist, best_node = d, n

    truck_route = astar_route(graph, depot, best_node)
    drone_leg = {
        "launch_node": best_node,
        "launch_latlon": (graph.nodes[best_node]["y"], graph.nodes[best_node]["x"]),
        "hospital_latlon": (hosp_lat, hosp_lon),
        "distance_km": best_dist,
    }
    return {"mode": "truck_plus_drone", "truck_route": truck_route, "drone_leg": drone_leg}
