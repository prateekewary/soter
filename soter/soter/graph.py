import math
import os

import networkx as nx
import osmnx as ox
from shapely.ops import unary_union

from . import config

ox.settings.use_cache = True
ox.settings.cache_folder = config.OSMNX_CACHE


def load_graph(place_name=None, refresh=False):
    place_name = place_name or config.PLACE_NAME
    if os.path.exists(config.GRAPH_CACHE) and not refresh:
        return ox.load_graphml(config.GRAPH_CACHE)
    G = ox.graph_from_place(place_name, network_type="drive")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    os.makedirs(os.path.dirname(config.GRAPH_CACHE), exist_ok=True)
    ox.save_graphml(G, config.GRAPH_CACHE)
    return G


def pick_depot_hospital(G):
    nodes_gdf, _ = ox.graph_to_gdfs(G)
    minx, miny, maxx, maxy = nodes_gdf.total_bounds
    depot_node = ox.distance.nearest_nodes(G, X=minx, Y=miny)
    centroid = nodes_gdf.geometry.unary_union.centroid
    hospital_node = ox.distance.nearest_nodes(G, X=centroid.x, Y=centroid.y)
    return depot_node, hospital_node


def astar_route(graph, orig, dest, weight="travel_time"):
    try:
        return nx.astar_path(graph, orig, dest, weight=weight)
    except nx.NetworkXNoPath:
        return None


def route_distance_km(graph, route):
    return sum(graph.edges[u, v, 0]["length"] for u, v in zip(route[:-1], route[1:])) / 1000


def heal_graph_with_damage(graph, damage_gdf, base_weight="travel_time"):
    healed = graph.copy()
    _, edges = ox.graph_to_gdfs(healed)
    edges = edges.reset_index()

    damage_union_by_level = {
        lvl: unary_union(damage_gdf.loc[damage_gdf.damage_level == lvl, "geometry"])
        for lvl in [1, 2, 3]
        if (damage_gdf.damage_level == lvl).any()
    }

    removed_edges = []
    penalized_edges = []

    for _, row in edges.iterrows():
        geom = row["geometry"]
        u, v, k = row["u"], row["v"], row["key"]
        worst_level = 0
        for lvl, union_geom in damage_union_by_level.items():
            if union_geom is not None and not union_geom.is_empty and geom.intersects(union_geom):
                worst_level = max(worst_level, lvl)
        if worst_level == 0:
            continue
        mult = config.DAMAGE_WEIGHT_MULTIPLIER[worst_level]
        if math.isinf(mult):
            if healed.has_edge(u, v, k):
                healed.remove_edge(u, v, k)
                removed_edges.append((u, v, k))
        else:
            if healed.has_edge(u, v, k):
                healed[u][v][k][base_weight] = healed[u][v][k].get(base_weight, 1.0) * mult
                healed[u][v][k]["damage_level"] = worst_level
                penalized_edges.append((u, v, k, worst_level))

    return healed, removed_edges, penalized_edges
