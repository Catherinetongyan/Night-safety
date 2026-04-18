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
import time
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
        arr = np.array(vals)
        lo = 0.0
        hi = np.percentile(arr, 98)
        span = (hi - lo) or 1.0
        return np.clip((arr - lo) / span, 0, 1).tolist()

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
# 4. Safety analysis
# ────────────────────────────────────────────────────────────

def route_safety_stats(G: nx.MultiDiGraph, route: list):
    """Compute safety and crime metrics along a route."""
    node_crime = nx.get_node_attributes(G, "crime_score")
 
    total_cctv    = 0
    total_lights  = 0
    total_length  = 0
    dark_length   = 0
    crime_scores  = []
 
    for u, v in zip(route[:-1], route[1:]):
        data   = G[u][v][0]
        length = data.get("length", 1.0)
 
        cc = data.get("cctv_count", 0)
        lc = data.get("light_count", 0)
 
        total_cctv   += cc
        total_lights += lc
        total_length += length
 
        if lc == 0:
            dark_length += length
 
        edge_crime = (node_crime.get(u, 0) + node_crime.get(v, 0)) / 2.0
        crime_scores.append(edge_crime)
 
    total_km = total_length / 1000
 
    return {
        "total_length":   total_length,
        "light_density":  total_lights / total_km if total_km > 0 else 0,
        "cctv_density":   total_cctv / total_km if total_km > 0 else 0,
        "lit_coverage":   1 - dark_length / total_length if total_length > 0 else 0,
        "dark_length":    dark_length,
        "total_cctv":     total_cctv,
        "total_lights":   total_lights,
        "avg_crime_risk": float(np.mean(crime_scores)) if crime_scores else 0,
        "max_crime_risk": float(np.max(crime_scores)) if crime_scores else 0,
    }
 
 
def print_safety_analysis(stats_a, stats_b, stats_c):
    """Print a comparison table and improvement metrics."""
    print("\n========== SAFETY ANALYSIS ==========")
    print(f"{'Metric':<25} {'A (shortest)':>14} {'B (+crime)':>14} {'C (full)':>14}")
    print("-" * 70)
    print(f"{'Distance':<25} {stats_a['total_length']:>12.0f}m {stats_b['total_length']:>12.0f}m {stats_c['total_length']:>12.0f}m")
    print(f"{'Lights per km':<25} {stats_a['light_density']:>13.1f} {stats_b['light_density']:>13.1f} {stats_c['light_density']:>13.1f}")
    print(f"{'CCTV per km':<25} {stats_a['cctv_density']:>13.1f} {stats_b['cctv_density']:>13.1f} {stats_c['cctv_density']:>13.1f}")
    print(f"{'Lit coverage':<25} {stats_a['lit_coverage']:>13.1%} {stats_b['lit_coverage']:>13.1%} {stats_c['lit_coverage']:>13.1%}")
    print(f"{'Dark segments':<25} {stats_a['dark_length']:>12.0f}m {stats_b['dark_length']:>12.0f}m {stats_c['dark_length']:>12.0f}m")
    print(f"{'Avg crime risk':<25} {stats_a['avg_crime_risk']:>13.3f} {stats_b['avg_crime_risk']:>13.3f} {stats_c['avg_crime_risk']:>13.3f}")
    print(f"{'Max crime risk':<25} {stats_a['max_crime_risk']:>13.3f} {stats_b['max_crime_risk']:>13.3f} {stats_c['max_crime_risk']:>13.3f}")
 
    # C vs A improvements
    def pct_change(new, old):
        return (new - old) / max(abs(old), 1e-9) * 100
 
    light_d = pct_change(stats_c["light_density"], stats_a["light_density"])
    cctv_d  = pct_change(stats_c["cctv_density"],  stats_a["cctv_density"])
    lit_c   = pct_change(stats_c["lit_coverage"],   stats_a["lit_coverage"])
    dark_r  = pct_change(stats_a["dark_length"],    stats_c["dark_length"])
    crime_r = pct_change(stats_a["avg_crime_risk"], stats_c["avg_crime_risk"])
    detour = (stats_c["total_length"] - stats_a["total_length"]) / stats_a["total_length"] * 100 if stats_a["total_length"] > 0 else 0

    print(f"\n  Route C vs Route A:")
    print(f"    Detour:            {detour:.1f}%")
    print(f"    Lights per km:   {'+' if light_d >= 0 else ''}{light_d:.1f}%")
    print(f"    CCTV per km:     {'+' if cctv_d >= 0 else ''}{cctv_d:.1f}%")
    print(f"    Lit coverage:    {'+' if lit_c >= 0 else ''}{lit_c:.1f}%")
    print(f"    Dark segments:   {'-' if dark_r > 0 else '+'}{abs(dark_r):.1f}%")
    print(f"    Avg crime risk:  {'-' if crime_r > 0 else '+'}{abs(crime_r):.1f}%")
    print("=====================================\n")

# ────────────────────────────────────────────────────────────
# 5. Visualisation
# ────────────────────────────────────────────────────────────

def plot_routes(G, start_node, goal_node, quickest, crime_route, safest,
                cctv_gdf, lights_gdf, cfg):
    speed = cfg["route"]["walking_speed_kmh"]

    quick_dist, quick_mins = route_stats(G, quickest,    speed)
    crime_dist, crime_mins = route_stats(G, crime_route, speed)
    safe_dist,  safe_mins  = route_stats(G, safest,      speed)

    fig, axes = plt.subplots(1, 3, figsize=(27, 9))
    fig.patch.set_facecolor("#1a1a2e")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.90, wspace=0.05)

    titles = [
        f"A  ·  Distance only\n{quick_dist:.0f}m  •  {format_time(quick_mins)} walking",
        f"B  ·  Distance + Crime\n{crime_dist:.0f}m  •  {format_time(crime_mins)} walking",
        f"C  ·  Distance + Crime + Safety\n{safe_dist:.0f}m  •  {format_time(safe_mins)} walking",
    ]
    routes  = [quickest, crime_route, safest]
    colours = ["#FFFFFF", "#FFFFFF", "#FFFFFF"]

    node_crime = nx.get_node_attributes(G, "crime_score")
    edge_crimes = [
        (node_crime.get(u, 0) + node_crime.get(v, 0)) / 2.0
        for u, v, _ in G.edges(data=True)
    ]
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=min(edge_crimes), vmax=max(edge_crimes))
    edge_colors = [cmap(norm(c)) for c in edge_crimes]

    for ax, route, title, colour in zip(axes, routes, titles, colours):
        ox.plot_graph(
            G, ax=ax, node_size=0,
            edge_color=edge_colors,
            edge_linewidth=1.0,
            bgcolor="#1a1a2e",
            show=False, close=False,
        )
        edge_xs, edge_ys = [], []
        for u, v in zip(route[:-1], route[1:]):
            edge_xs += [G.nodes[u]["x"], G.nodes[v]["x"], None]
            edge_ys += [G.nodes[u]["y"], G.nodes[v]["y"], None]
        ax.plot(edge_xs, edge_ys, color=colour, linewidth=4, zorder=4)

        ax.scatter(G.nodes[start_node]["x"], G.nodes[start_node]["y"],
                   c="white", s=120, zorder=6, marker="o")
        ax.scatter(G.nodes[goal_node]["x"],  G.nodes[goal_node]["y"],
                   c=colour,  s=150, zorder=6, marker="*")

        cctv_gdf.plot(ax=ax,   color="#02CCFF7B", markersize=6, zorder=5, alpha=0.9,marker="s")
        lights_gdf.plot(ax=ax, color="#FFFB0078", markersize=1.5, zorder=4, alpha=0.3)

        xs = [G.nodes[n]["x"] for n in route]
        ys = [G.nodes[n]["y"] for n in route]
        margin = 0.003
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=12)
        ax.set_axis_off()

    # Colorbar: independent axis on the far left, no space stolen from subplots
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.015, 0.15, 0.012, 0.55])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Crime risk", color="white", fontsize=11)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=9)

    legend_handles = [
        mpatches.Patch(color="#FFFFFF", label="Route"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="white",   markersize=8,  label="Start"),
        plt.Line2D([0],[0], marker="*", color="w", markerfacecolor="#aaaaaa", markersize=10, label="End"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#02CCFF7B",     markersize=6,  label="CCTV"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#FFFB0078",  markersize=6,  label="Street light"),
        mpatches.Patch(color=cmap(0.1), label="Road: low risk"),
        mpatches.Patch(color=cmap(0.9), label="Road: high risk"),
    ]
    fig.legend(
        handles=legend_handles, loc = "lower center", ncol=7,
        facecolor="#1a1a2e", labelcolor="white", fontsize=10, framealpha=0.8,
    )

    start_name = cfg["route"]["start"]["name"]
    end_name   = cfg["route"]["end"]["name"]
    plt.suptitle(
        f"{start_name}  →  {end_name}",
        color="white", fontsize=16, fontweight="bold", y=0.97,
    )

    print("\n========== ROUTE SUMMARY ==========")
    print(f"  A  Distance only:             {quick_dist:.0f}m  |  {format_time(quick_mins)}")
    print(f"  B  Distance + Crime:          {crime_dist:.0f}m  |  {format_time(crime_mins)}")
    print(f"  C  Distance + Crime + Safety: {safe_dist:.0f}m  |  {format_time(safe_mins)}")
    print("====================================\n")

    # Safety analysis
    stats_a = route_safety_stats(G, quickest)
    stats_b = route_safety_stats(G, crime_route)
    stats_c = route_safety_stats(G, safest)
    print_safety_analysis(stats_a, stats_b, stats_c)

    plt.show()


# ────────────────────────────────────────────────────────────
# 5. Entry point
# ────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

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
    elapsed = time.time() - t_start
    print(f"Total runtime: {elapsed:.1f}s")

    plot_routes(G, start_node, goal_node, quickest, crime_route, safest, cctv_gdf, lights_gdf, cfg)
    
if __name__ == "__main__":
    main()