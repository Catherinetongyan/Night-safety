"""
Bristol Heat Method Router
==========================
Compares three routes on the same weighted graph built by Mapping_software.py:

A — Shortest path          (distance only)
B — A* full                (distance + crime + safety, exponential cost)
C — Heat full              (distance + crime + safety, linear cost + heat method)

Usage
-----
    python bristol_heat_routing_gradient.py
    python bristol_heat_routing_gradient.py --config config.yaml
"""

from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import yaml
from scipy.sparse import csr_matrix, diags, lil_matrix
from scipy.sparse.linalg import spsolve


# ────────────────────────────────────────────────────────────
# 1. Config
# ────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else base_dir / p


@dataclass
class UserProfile:
    w_distance: float
    w_crime: float
    w_safety: float
    require_lit: bool = False
    epsilon: float = 1.0e-6


def load_linear_profile(cfg: dict) -> UserProfile:
    linear_cfg = cfg.get("linear_cost", {})
    return UserProfile(
        w_distance=float(linear_cfg.get("w_distance", 0.4)),
        w_crime=float(linear_cfg.get("w_crime", 0.3)),
        w_safety=float(linear_cfg.get("w_safety", 0.3)),
        require_lit=bool(linear_cfg.get("require_lit", False)),
        epsilon=float(linear_cfg.get("epsilon", 1.0e-6)),
    )


# ────────────────────────────────────────────────────────────
# 2. Routing attributes
# ────────────────────────────────────────────────────────────

def compute_edge_norms(G: nx.MultiDiGraph, cfg: dict) -> None:
    """Store norm_crime and norm_safety on each edge using the same p98 clip as safe_route.py."""
    t0 = time.time()
    print("Normalising crime / safety for routing...")

    w_cctv = cfg["weights"].get("cctv_weight", 3)
    w_light = cfg["weights"].get("light_weight", 1)
    node_crime = nx.get_node_attributes(G, "crime_score")

    edge_keys, raw_crimes, raw_safeties = [], [], []
    for u, v, k, data in G.edges(keys=True, data=True):
        length = float(data.get("length", 1.0) or 1.0)
        edge_keys.append((u, v, k))

        crime = (node_crime.get(u, 0.0) + node_crime.get(v, 0.0)) / 2.0
        raw_crimes.append(crime)

        cc = data.get("cctv_count", 0)
        lc = data.get("light_count", 0)
        safety = (cc * w_cctv + lc * w_light) / max(length, 1.0)
        raw_safeties.append(safety)

    def p98_clip(vals):
        arr = np.asarray(vals, dtype=np.float64)
        hi = float(np.percentile(arr, 98)) if len(arr) else 1.0
        hi = hi or 1.0
        return np.clip(arr / hi, 0.0, 1.0)

    norm_c = p98_clip(raw_crimes)
    norm_s = p98_clip(raw_safeties)

    for (u, v, k), nc, ns in zip(edge_keys, norm_c, norm_s):
        G[u][v][k]["norm_crime"] = float(nc)
        G[u][v][k]["norm_safety"] = float(ns)

    print(f"  Normalised {len(edge_keys):,} edges ({time.time() - t0:.2f}s)")


def compute_full_astar_weights(G: nx.MultiDiGraph, cfg: dict) -> None:
    """Store baseline A* full cost on each edge."""
    t0 = time.time()
    print("Computing A* full edge weights...")

    alpha = float(cfg["weights"].get("alpha", 1.3))
    beta = float(cfg["weights"].get("beta", 1.1))

    for _, _, _, data in G.edges(keys=True, data=True):
        length = float(data.get("length", 1.0))
        nc = float(data.get("norm_crime", 0.0))
        ns = float(data.get("norm_safety", 0.0))
        data["astar_full_weight"] = length * np.exp(alpha * nc - beta * ns)

    print(f"  A* full weights ready ({time.time() - t0:.2f}s)")


def linear_cost(edge_data: dict, profile: UserProfile) -> float:
    """Linear cost used by the heat method."""
    if profile.require_lit and edge_data.get("light_count", 0) <= 0:
        return float("inf")

    length = float(edge_data.get("length", 50.0))
    nc = float(edge_data.get("norm_crime", 0.0))
    ns = float(edge_data.get("norm_safety", 0.0))

    cost = (
        profile.w_distance * length
        + profile.w_crime * nc * length
        + profile.w_safety * (1.0 - ns) * length
    )
    return max(cost, profile.epsilon)


# ────────────────────────────────────────────────────────────
# 3. Heat method
# ────────────────────────────────────────────────────────────

def build_laplacian(G: nx.MultiDiGraph, profile: UserProfile):
    t0 = time.time()
    nodes = list(G.nodes())
    node_idx = {node: i for i, node in enumerate(nodes)}
    rows, cols, vals = [], [], []

    for u, v, data in G.edges(data=True):
        c = linear_cost(data, profile)
        if c == float("inf"):
            continue
        i, j = node_idx[u], node_idx[v]
        w = 1.0 / c
        rows += [i, j, i, j]
        cols += [j, i, i, j]
        vals += [-w, -w, w, w]

    L = csr_matrix((vals, (rows, cols)), shape=(len(nodes), len(nodes)))
    print(f"  Laplacian built ({time.time() - t0:.2f}s)")
    return L, nodes, node_idx


def run_heat_method(G: nx.MultiDiGraph, origin_node, profile: UserProfile, cfg: dict) -> dict:
    heat_cfg = cfg.get("heat_method", {})
    t_factor = float(heat_cfg.get("t_factor", 1.0))

    print("Running heat method...")
    L, nodes, node_idx = build_laplacian(G, profile)
    n = len(nodes)
    src = node_idx[origin_node]

    t0 = time.time()
    costs = []
    for _, _, data in G.edges(data=True):
        c = linear_cost(data, profile)
        if c < float("inf"):
            costs.append(c)
    mean_c = float(np.mean(costs)) if costs else 1.0
    t = t_factor * (mean_c ** 2)
    print(f"  Heat time scale t={t:.6f} ({time.time() - t0:.2f}s)")

    t0 = time.time()
    delta = np.zeros(n)
    delta[src] = 1.0
    I = diags(np.ones(n), format="csr")
    u = spsolve(I + t * L, delta)
    print(f"  Heat diffusion solve ({time.time() - t0:.2f}s)")

    t0 = time.time()
    div = np.zeros(n)
    for eu, ev, data in G.edges(data=True):
        c = linear_cost(data, profile)
        if c == float("inf"):
            continue
        i, j = node_idx[eu], node_idx[ev]
        w = 1.0 / c
        grad = u[j] - u[i]
        norm = abs(grad)
        if norm < 1e-12:
            continue
        X = -grad / norm
        div[i] += w * X
        div[j] -= w * X
    print(f"  Gradient normalisation ({time.time() - t0:.2f}s)")

    t0 = time.time()
    L_mod = lil_matrix(L)
    div_mod = div.copy()
    L_mod[src, :] = 0
    L_mod[src, src] = 1
    div_mod[src] = 0
    phi = spsolve(L_mod.tocsr(), div_mod)
    phi -= phi[src]
    phi = np.abs(phi)
    print(f"  Poisson solve ({time.time() - t0:.2f}s)")

    return {nodes[i]: phi[i] for i in range(n)}


def recover_path(G: nx.MultiDiGraph, origin_node, dest_node, distances: dict, profile: UserProfile) -> list:
    t0 = time.time()
    if origin_node == dest_node:
        print(f"  Gradient descent recovery ({time.time() - t0:.2f}s)")
        return [origin_node]

    path = [dest_node]
    visited = {dest_node}
    current = dest_node

    for _ in range(len(G.nodes)):
        if current == origin_node:
            break
        phi_here = distances.get(current, float("inf"))
        neighbours = list(G.successors(current)) + list(G.predecessors(current))

        best_node, best_phi = None, phi_here
        for nb in neighbours:
            if nb in visited:
                continue
            phi_nb = distances.get(nb, float("inf"))
            if phi_nb < best_phi:
                best_phi, best_node = phi_nb, nb

        if best_node is None:
            print(f"  Gradient descent stuck at node {current} (phi={phi_here:.4f})")
            break

        visited.add(best_node)
        path.append(best_node)
        current = best_node

    if current != origin_node:
        print("  Falling back to Dijkstra on linear full cost...")
        try:
            route = nx.shortest_path(
                G,
                origin_node,
                dest_node,
                weight=lambda u, v, d: linear_cost(d, profile),
            )
            print(f"  Gradient descent recovery + fallback ({time.time() - t0:.2f}s)")
            return route
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            print(f"  Gradient descent recovery failed ({time.time() - t0:.2f}s)")
            return []

    path.reverse()
    print(f"  Gradient descent recovery ({time.time() - t0:.2f}s)")
    return path


# ────────────────────────────────────────────────────────────
# 4. Route stats and safety analysis
# ────────────────────────────────────────────────────────────

def route_stats(G: nx.MultiDiGraph, route: list, speed_kmh: float):
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


def route_safety_stats(G: nx.MultiDiGraph, route: list):
    node_crime = nx.get_node_attributes(G, "crime_score")

    total_cctv = 0
    total_lights = 0
    total_length = 0
    dark_length = 0
    crime_scores = []

    for u, v in zip(route[:-1], route[1:]):
        data = G[u][v][0]
        length = data.get("length", 1.0)

        cc = data.get("cctv_count", 0)
        lc = data.get("light_count", 0)

        total_cctv += cc
        total_lights += lc
        total_length += length

        if lc == 0:
            dark_length += length

        edge_crime = (node_crime.get(u, 0) + node_crime.get(v, 0)) / 2.0
        crime_scores.append(edge_crime)

    total_km = total_length / 1000

    return {
        "total_length": total_length,
        "light_density": total_lights / total_km if total_km > 0 else 0,
        "cctv_density": total_cctv / total_km if total_km > 0 else 0,
        "lit_coverage": 1 - dark_length / total_length if total_length > 0 else 0,
        "dark_length": dark_length,
        "total_cctv": total_cctv,
        "total_lights": total_lights,
        "avg_crime_risk": float(np.mean(crime_scores)) if crime_scores else 0,
        "max_crime_risk": float(np.max(crime_scores)) if crime_scores else 0,
    }


def print_safety_analysis(stats_a, stats_b, stats_c):
    print("\n========== SAFETY ANALYSIS ==========")
    header = (
        f"{'Metric':<25} {'A (shortest)':>14} {'B (A* full)':>14} {'C (heat full)':>16}"
    )
    print(header)
    print("-" * 76)
    print(
        f"{'Distance':<25} {stats_a['total_length']:>12.0f}m "
        f"{stats_b['total_length']:>12.0f}m {stats_c['total_length']:>14.0f}m"
    )
    print(
        f"{'Lights per km':<25} {stats_a['light_density']:>13.1f} "
        f"{stats_b['light_density']:>13.1f} {stats_c['light_density']:>15.1f}"
    )
    print(
        f"{'CCTV per km':<25} {stats_a['cctv_density']:>13.1f} "
        f"{stats_b['cctv_density']:>13.1f} {stats_c['cctv_density']:>15.1f}"
    )
    print(
        f"{'Lit coverage':<25} {stats_a['lit_coverage']:>13.1%} "
        f"{stats_b['lit_coverage']:>13.1%} {stats_c['lit_coverage']:>15.1%}"
    )
    print(
        f"{'Dark segments':<25} {stats_a['dark_length']:>12.0f}m "
        f"{stats_b['dark_length']:>12.0f}m {stats_c['dark_length']:>14.0f}m"
    )
    print(
        f"{'Avg crime risk':<25} {stats_a['avg_crime_risk']:>13.3f} "
        f"{stats_b['avg_crime_risk']:>13.3f} {stats_c['avg_crime_risk']:>15.3f}"
    )
    print(
        f"{'Max crime risk':<25} {stats_a['max_crime_risk']:>13.3f} "
        f"{stats_b['max_crime_risk']:>13.3f} {stats_c['max_crime_risk']:>15.3f}"
    )

    def pct_change(new, old):
        return (new - old) / max(abs(old), 1e-9) * 100

    def print_comparison(label, stats_x):
        detour = pct_change(stats_x["total_length"], stats_a["total_length"])
        light_d = pct_change(stats_x["light_density"], stats_a["light_density"])
        cctv_d = pct_change(stats_x["cctv_density"], stats_a["cctv_density"])
        lit_c = pct_change(stats_x["lit_coverage"], stats_a["lit_coverage"])
        dark_r = pct_change(stats_a["dark_length"], stats_x["dark_length"])
        crime_r = pct_change(stats_a["avg_crime_risk"], stats_x["avg_crime_risk"])

        print(f"\n  {label} vs Route A:")
        print(f"    Detour:          {detour:+.1f}%")
        print(f"    Lights per km:   {light_d:+.1f}%")
        print(f"    CCTV per km:     {cctv_d:+.1f}%")
        print(f"    Lit coverage:    {lit_c:+.1f}%")
        print(f"    Dark segments:   {'-' if dark_r > 0 else '+'}{abs(dark_r):.1f}%")
        print(f"    Avg crime risk:  {'-' if crime_r > 0 else '+'}{abs(crime_r):.1f}%")

    print_comparison("Route B", stats_b)
    print_comparison("Route C", stats_c)
    print("=====================================\n")


# ────────────────────────────────────────────────────────────
# 5. Plotting
# ────────────────────────────────────────────────────────────

def plot_routes(G, start_node, goal_node, route_a, route_b, route_c, cctv_gdf, lights_gdf, cfg):
    speed = cfg["route"]["walking_speed_kmh"]

    dist_a, mins_a = route_stats(G, route_a, speed)
    dist_b, mins_b = route_stats(G, route_b, speed)
    dist_c, mins_c = route_stats(G, route_c, speed)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.patch.set_facecolor("#1a1a2e")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.90, wspace=0.05)

    titles = [
        f"A  ·  Shortest path\n{dist_a:.0f}m  •  {format_time(mins_a)} walking",
        f"B  ·  A* full\n{dist_b:.0f}m  •  {format_time(mins_b)} walking",
        f"C  ·  Heat full\n{dist_c:.0f}m  •  {format_time(mins_c)} walking",
    ]
    routes = [route_a, route_b, route_c]
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
            G,
            ax=ax,
            node_size=0,
            edge_color=edge_colors,
            edge_linewidth=1.0,
            bgcolor="#1a1a2e",
            show=False,
            close=False,
        )

        edge_xs, edge_ys = [], []
        for u, v in zip(route[:-1], route[1:]):
            edge_xs += [G.nodes[u]["x"], G.nodes[v]["x"], None]
            edge_ys += [G.nodes[u]["y"], G.nodes[v]["y"], None]
        ax.plot(edge_xs, edge_ys, color=colour, linewidth=4, zorder=4)

        ax.scatter(
            G.nodes[start_node]["x"],
            G.nodes[start_node]["y"],
            c="white",
            s=120,
            zorder=6,
            marker="o",
        )
        ax.scatter(
            G.nodes[goal_node]["x"],
            G.nodes[goal_node]["y"],
            c=colour,
            s=150,
            zorder=6,
            marker="*",
        )

        cctv_gdf.plot(ax=ax, color="#02CCFF7B", markersize=6, zorder=5, alpha=0.9, marker="s")
        lights_gdf.plot(ax=ax, color="#FFFB0078", markersize=1.5, zorder=4, alpha=0.3)

        xs = [G.nodes[n]["x"] for n in route]
        ys = [G.nodes[n]["y"] for n in route]
        margin = 0.003
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=12)
        ax.set_axis_off()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.015, 0.15, 0.012, 0.55])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Crime risk", color="white", fontsize=11)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=9)

    legend_handles = [
        mpatches.Patch(color="#FFFFFF", label="Route"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markersize=8, label="Start"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#aaaaaa", markersize=10, label="End"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#02CCFF7B", markersize=6, label="CCTV"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#FFFB0078", markersize=6, label="Street light"),
        mpatches.Patch(color=cmap(0.1), label="Road: low risk"),
        mpatches.Patch(color=cmap(0.9), label="Road: high risk"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=7,
        facecolor="#1a1a2e",
        labelcolor="white",
        fontsize=10,
        framealpha=0.8,
    )

    start_name = cfg["route"]["start"]["name"]
    end_name = cfg["route"]["end"]["name"]
    plt.suptitle(
        f"{start_name}  →  {end_name}",
        color="white",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    print("\n========== ROUTE SUMMARY ==========")
    print(f"  A  Shortest path: {dist_a:.0f}m  |  {format_time(mins_a)}")
    print(f"  B  A* full:       {dist_b:.0f}m  |  {format_time(mins_b)}")
    print(f"  C  Heat full:     {dist_c:.0f}m  |  {format_time(mins_c)}")
    print("===================================\n")

    sa = route_safety_stats(G, route_a)
    sb = route_safety_stats(G, route_b)
    sc = route_safety_stats(G, route_c)
    print_safety_analysis(sa, sb, sc)

    plt.show()


# ────────────────────────────────────────────────────────────
# 6. Entry point
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bristol Heat Method Router")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    t_total = time.time()
    config_path = Path(args.config)
    base_dir = config_path.parent
    cfg = load_config(config_path)
    profile = load_linear_profile(cfg)

    print("=" * 55)
    print("  Bristol Heat Method Router")
    print("=" * 55)

    t0 = time.time()
    graph_path = resolve_path(base_dir, cfg["paths"]["weighted_graph"])
    print(f"\n[1/6] Loading weighted graph: {graph_path}")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    print(f"  Graph loaded ({G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges) ({time.time() - t0:.2f}s)")

    print("\n[2/6] Preparing routing attributes")
    compute_edge_norms(G, cfg)
    compute_full_astar_weights(G, cfg)

    t0 = time.time()
    print("\n[3/6] Loading safety overlays")
    cctv_gdf = gpd.read_file(resolve_path(base_dir, cfg["paths"]["cctv"]))
    lights_gdf = gpd.read_file(resolve_path(base_dir, cfg["paths"]["lights"]))
    print(f"  CCTV: {len(cctv_gdf):,} | Lights: {len(lights_gdf):,} ({time.time() - t0:.2f}s)")

    t0 = time.time()
    print("\n[4/6] Mapping origin and destination to graph")
    start_cfg = cfg["route"]["start"]
    end_cfg = cfg["route"]["end"]
    start_node = ox.distance.nearest_nodes(G, start_cfg["lon"], start_cfg["lat"])
    goal_node = ox.distance.nearest_nodes(G, end_cfg["lon"], end_cfg["lat"])
    print(f"  Origin node: {start_node} | Dest node: {goal_node} ({time.time() - t0:.2f}s)")

    print("\n[5/6] Routing")
    t0 = time.time()
    route_a = nx.shortest_path(G, start_node, goal_node, weight="length")
    print(f"  A shortest path: {len(route_a)} nodes ({time.time() - t0:.2f}s)")

    t0 = time.time()
    route_b = nx.astar_path(G, start_node, goal_node, weight="astar_full_weight")
    print(f"  B A* full:       {len(route_b)} nodes ({time.time() - t0:.2f}s)")

    print("\n[6/6] Heat method")
    distances = run_heat_method(G, start_node, profile, cfg)
    print(f"  Phi at destination: {distances.get(goal_node, float('inf')):.4f}")
    route_c = recover_path(G, start_node, goal_node, distances, profile)
    if not route_c:
        print("  No heat route found; using shortest-path fallback for plotting")
        route_c = route_a

    print(f"\nPlotting... Total pipeline: {time.time() - t_total:.2f}s")
    plot_routes(G, start_node, goal_node, route_a, route_b, route_c, cctv_gdf, lights_gdf, cfg)


if __name__ == "__main__":
    main()