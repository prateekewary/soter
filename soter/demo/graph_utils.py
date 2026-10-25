import math
import os

import networkx as nx
import osmnx as ox
from shapely.ops import unary_union

import config

ox.settings.use_cache = True
ox.settings.cache_folder = config.OSMNX_CACHE


def _cache_path_for(place_name):
    """One cache file per place. Previously GRAPH_CACHE was a single fixed
    path, so after the first graph was cached (e.g. Kathmandu), every later
    call -- even with a different place_name -- silently returned that same
    cached graph instead of fetching/caching the requested place."""
    base, ext = os.path.splitext(config.GRAPH_CACHE)
    slug = "".join(c if c.isalnum() else "_" for c in place_name.lower()).strip("_")
    return f"{base}__{slug}{ext}"


def load_graph(place_name=None, refresh=False):
    place_name = place_name or config.PLACE_NAME
    cache_path = _cache_path_for(place_name)

    if os.path.exists(cache_path) and not refresh:
        return ox.load_graphml(cache_path)

    G = ox.graph_from_place(place_name, network_type="drive")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    ox.save_graphml(G, cache_path)
    return G


def build_edges_gdf(G):
    _, edges = ox.graph_to_gdfs(G)
    edges = edges.reset_index()[["u", "v", "key", "geometry"]]
    return edges


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


def heal_graph_with_damage(graph, damage_gdf, edges_gdf=None, base_weight="travel_time", severity=1.0):
    healed = graph.copy()

    if edges_gdf is not None:
        edges = edges_gdf.reset_index() if "u" not in edges_gdf.columns else edges_gdf
    else:
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
        base_mult = config.DAMAGE_WEIGHT_MULTIPLIER[worst_level]
        mult = base_mult if math.isinf(base_mult) else 1.0 + (base_mult - 1.0) * severity
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