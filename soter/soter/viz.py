import folium

from . import config


def plot_plan_on_map(graph, damage_gdf, plan, depot, hospital):
    center_lat = graph.nodes[depot]["y"]
    center_lon = graph.nodes[depot]["x"]
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")

    for _, row in damage_gdf.iterrows():
        color = config.DAMAGE_COLORS.get(row["damage_level"], "#999999")
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda feat, c=color: {"fillColor": c, "color": c, "fillOpacity": 0.35, "weight": 1},
            tooltip=f"Damage level {row['damage_level']}",
        ).add_to(m)

    folium.Marker(
        [graph.nodes[depot]["y"], graph.nodes[depot]["x"]],
        tooltip="Depot / Cold-chain Warehouse",
        icon=folium.Icon(color="blue", icon="truck", prefix="fa"),
    ).add_to(m)
    folium.Marker(
        [graph.nodes[hospital]["y"], graph.nodes[hospital]["x"]],
        tooltip="Hospital",
        icon=folium.Icon(color="red", icon="plus", prefix="fa"),
    ).add_to(m)

    if plan["truck_route"]:
        coords = [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for n in plan["truck_route"]]
        folium.PolyLine(coords, color="#2980b9", weight=5, opacity=0.9, tooltip="Truck route").add_to(m)

    if plan["drone_leg"]:
        launch = plan["drone_leg"]["launch_latlon"]
        hosp = plan["drone_leg"]["hospital_latlon"]
        folium.Marker(launch, tooltip="Drone Launch Point", icon=folium.Icon(color="orange", icon="plane", prefix="fa")).add_to(m)
        folium.PolyLine(
            [launch, hosp],
            color="#c0392b",
            weight=4,
            dash_array="8,8",
            tooltip=f"Drone hand-off ({plan['drone_leg']['distance_km']:.2f} km)",
        ).add_to(m)

    return m


def save_map(m, out_path):
    m.save(out_path)
    return out_path
