"""
Bristol Safe Walking Router
============================
Adjust paths, waypoints and parameters in config.yaml.
Usage:
    python safe_route.py
    python safe_route.py --config my_config.yaml
"""

import argparse
import pickle
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import yaml


# ────────────────────────────────────────────────────────────
# 1. Config
# ────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    """Supports absolute paths and paths relative to the config file."""
    p = Path(raw_path)
    return p if p.is_absolute() else base_dir / p


# ────────────────────────────────────────────────────────────
# 2. Edge costs
# ────────────────────────────────────────────────────────────

def compute_edge_costs(G: nx.MultiDiGraph, cfg: dict):
    """Compute combined routing edge costs in-place using exponential model.

    w(e) = length(e) * exp( alpha * R̃(e) − beta * S̃(e) )

    Route A — length only         (weight="length",       no change needed)
    Route B — length + crime      (weight="crime_weight")
    Route C — length + crime + safety infrastructure (weight="safety_weight")
    """
    alpha = cfg["weights"]["alpha"]
    beta  = cfg["weights"]["beta"]

    node_crime = nx.get_node_attributes(G, "crime_score")

    lengths, crime_scores, safety_scores = [], [], []
    for u, v, data in G.edges(data=True):
        lengths.append(data.get("length", 1.0))
        crime_scores.append((node_crime.get(u, 0) + node_crime.get(v, 0)) / 2.0)
        safety_scores.append(data.get("safety_density", 0.0))

    def normalise(vals):
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        return [(v - lo) / span for v in vals]

    norm_crimes   = normalise(crime_scores)
    norm_safeties = normalise(safety_scores)

    for (u, v, data), length, nc, ns in zip(
        G.edges(data=True), lengths, norm_crimes, norm_safeties
    ):
        data["crime_weight"]  = length * np.exp(alpha * nc)
        data["safety_weight"] = length * np.exp(alpha * nc - beta * ns)


# ────────────────────────────────────────────────────────────
# 3. Routing
# ────────────────────────────────────────────────────────────

def find_routes(G: nx.MultiDiGraph, cfg: dict):
    """Return (start_node, goal_node, quickest, crime_route, safest)."""
    route_cfg = cfg["route"]
    start_lat, start_lon = route_cfg["start"]["lat"], route_cfg["start"]["lon"]
    goal_lat,  goal_lon  = route_cfg["end"]["lat"],   route_cfg["end"]["lon"]

    start_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
    goal_node  = ox.distance.nearest_nodes(G, goal_lon,  goal_lat)

    quickest    = nx.shortest_path(G, start_node, goal_node, weight="length")
    crime_route = nx.astar_path(G,   start_node, goal_node, weight="crime_weight")
    safest      = nx.astar_path(G,   start_node, goal_node, weight="safety_weight")

    return start_node, goal_node, quickest, crime_route, safest


def route_stats(G: nx.MultiDiGraph, route: list, speed_kmh: float):
    """Return (distance_m, walking_minutes)."""
    total_m = sum(
        G[u][v][0].get("length", 0)
        for u, v in zip(route[:-1], route[1:])
    )
    minutes = (total_m / 1000 / speed_kmh) * 60
    return total_m, minutes


def format_time(minutes: float) -> str:
    if minutes < 60:
        return f"{int(round(minutes))} min"
    h = int(minutes // 60)
    m = int(round(minutes % 60))
    return f"{h}h {m}min"


# ────────────────────────────────────────────────────────────
# 4. Visualisation
# ────────────────────────────────────────────────────────────

def plot_routes(G, start_node, goal_node, quickest, crime_route, safest,
                cctv_gdf, lights_gdf, cfg):
    speed = cfg["route"]["walking_speed_kmh"]

    quick_dist, quick_mins = route_stats(G, quickest,    speed)
    crime_dist, crime_mins = route_stats(G, crime_route, speed)
    safe_dist,  safe_mins  = route_stats(G, safest,      speed)

    fig, axes = plt.subplots(1, 3, figsize=(27, 9))
    fig.patch.set_facecolor("#1a1a2e")

    titles = [
        f"A  ·  Distance only\n{quick_dist:.0f}m  •  {format_time(quick_mins)} walking",
        f"B  ·  Distance + Crime\n{crime_dist:.0f}m  •  {format_time(crime_mins)} walking",
        f"C  ·  Distance + Crime + Safety\n{safe_dist:.0f}m  •  {format_time(safe_mins)} walking",
    ]
    routes  = [quickest, crime_route, safest]
    colours = ["#00d4ff", "#ff9f43", "#00ff99"]

    node_crime = nx.get_node_attributes(G, "crime_score")
    edge_crimes = [
        (node_crime.get(u, 0) + node_crime.get(v, 0)) / 2.0
        for u, v, _ in G.edges(data=True)
    ]
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=min(edge_crimes), vmax=max(edge_crimes))
    edge_colors = [cmap(norm(c)) for c in edge_crimes]

    for ax, route, title, colour in zip(axes, routes, titles, colours):
        ox.plot_graph(
            G, ax=ax, node_size=0,
            edge_color=edge_colors,
            edge_linewidth=0.8,
            bgcolor="#1a1a2e",
            show=False, close=False,
        )
        edge_xs, edge_ys = [], []
        for u, v in zip(route[:-1], route[1:]):
            edge_xs += [G.nodes[u]["x"], G.nodes[v]["x"], None]
            edge_ys += [G.nodes[u]["y"], G.nodes[v]["y"], None]
        ax.plot(edge_xs, edge_ys, color=colour, linewidth=3, zorder=4)

        ax.scatter(G.nodes[start_node]["x"], G.nodes[start_node]["y"],
                   c="white", s=120, zorder=6, marker="o")
        ax.scatter(G.nodes[goal_node]["x"],  G.nodes[goal_node]["y"],
                   c=colour,  s=150, zorder=6, marker="*")

        cctv_gdf.plot(ax=ax,   color="red",    markersize=4, zorder=5, alpha=0.6)
        lights_gdf.plot(ax=ax, color="yellow", markersize=2, zorder=4, alpha=0.4)

        xs = [G.nodes[n]["x"] for n in route]
        ys = [G.nodes[n]["y"] for n in route]
        margin = 0.003
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=12)
        ax.set_axis_off()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, location="right")
    cb.set_label("Crime risk", color="white", fontsize=11)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    legend_handles = [
        mpatches.Patch(color="#00d4ff", label="A · Distance only"),
        mpatches.Patch(color="#ff9f43", label="B · Distance + Crime"),
        mpatches.Patch(color="#00ff99", label="C · Full safety"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="white",   markersize=8,  label="Start"),
        plt.Line2D([0],[0], marker="*", color="w", markerfacecolor="#aaaaaa", markersize=10, label="End"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="red",     markersize=6,  label="CCTV"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="yellow",  markersize=6,  label="Street light"),
        mpatches.Patch(color=cmap(0.1), label="Road: low crime risk"),
        mpatches.Patch(color=cmap(0.9), label="Road: high crime risk"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=5,
        facecolor="#1a1a2e", labelcolor="white", fontsize=10, framealpha=0.8,
    )

    start_name = cfg["route"]["start"]["name"]
    end_name   = cfg["route"]["end"]["name"]
    plt.suptitle(
        f"{start_name}  →  {end_name}",
        color="white", fontsize=16, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    print("\n========== ROUTE SUMMARY ==========")
    print(f"  A  Distance only:             {quick_dist:.0f}m  |  {format_time(quick_mins)}")
    print(f"  B  Distance + Crime:          {crime_dist:.0f}m  |  {format_time(crime_mins)}")
    print(f"  C  Distance + Crime + Safety: {safe_dist:.0f}m  |  {format_time(safe_mins)}")
    print("====================================\n")

    plt.show()


# ────────────────────────────────────────────────────────────
# 5. Entry point
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bristol Safe Walking Router")
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    base_dir    = config_path.parent
    cfg         = load_config(config_path)

    graph_path = resolve_path(base_dir, cfg["paths"]["weighted_graph"])
    print(f"Loading weighted graph: {graph_path}")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    cctv_gdf   = gpd.read_file(resolve_path(base_dir, cfg["paths"]["cctv"]))
    lights_gdf = gpd.read_file(resolve_path(base_dir, cfg["paths"]["lights"]))

    print("\nComputing edge costs...")
    compute_edge_costs(G, cfg)

    print("\nFinding routes...")
    start_node, goal_node, quickest, crime_route, safest = find_routes(G, cfg)

    plot_routes(G, start_node, goal_node, quickest, crime_route, safest, cctv_gdf, lights_gdf, cfg)


if __name__ == "__main__":
    main()