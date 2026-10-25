def route_distance_km(graph, route):
    if not route:
        return 0.0
    return sum(graph.edges[u, v, 0]["length"] for u, v in zip(route[:-1], route[1:])) / 1000


def build_stats(G, healed, plan, removed, penalized, heal_ms, route_ms):
    truck_km = route_distance_km(healed, plan["truck_route"])
    drone_km = plan["drone_leg"]["distance_km"] if plan["drone_leg"] else 0.0

    lines = [
        f"**Mode:** {'Truck only' if plan['mode'] == 'truck_only' else 'Truck + Drone handoff'}",
        f"**Truck distance:** {truck_km:.2f} km",
    ]
    if plan["drone_leg"]:
        lines.append(f"**Drone leg:** {drone_km:.2f} km")
    lines.append(f"**Roads destroyed:** {len(removed)}")
    lines.append(f"**Roads slowed:** {len(penalized)}")
    lines.append(f"**Graph heal time:** {heal_ms:.1f} ms")
    lines.append(f"**Routing time:** {route_ms:.1f} ms")
    return "\n\n".join(lines)
