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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler


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
# 2. Crime KDE
# ────────────────────────────────────────────────────────────

def build_crime_scores(G: nx.MultiDiGraph, cfg: dict, base_dir: Path) -> dict:
    """Read crime CSV, run weighted KDE, return {node_id: crime_score [0,1]}."""
    crime_path = resolve_path(base_dir, cfg["paths"]["crime_csv"])
    kde_cfg    = cfg["kde"]
    severity   = cfg["severity"]

    print("  Loading crime data...")
    df = pd.read_csv(crime_path)
    df["severity"] = df["crime_type"].map(severity).fillna(0.5)

    min_sev = kde_cfg.get("min_severity", 0.0)
    if min_sev > 0:
        before = len(df)
        df = df[df["severity"] >= min_sev]
        print(f"  Severity filter: {before:,} -> {len(df):,} records")

    # Flat local projection centred on Bristol (units: metres)
    LAT0, LON0 = 51.45, -2.60
    coords_m = np.stack([
        (df["lat"].values - LAT0) * 111320,
        (df["lon"].values - LON0) * 111320 * np.cos(np.radians(LAT0)),
    ], axis=1).astype(np.float32)
    weights = df["severity"].values.astype(np.float32)
    weights = weights / weights.sum()

    bw_metres = kde_cfg.get("bandwidth_metres", 150)

    # Node coords projected to the same flat space
    node_ids      = list(G.nodes)
    node_coords_m = np.stack([
        (np.array([G.nodes[n]["y"] for n in node_ids]) - LAT0) * 111320,
        (np.array([G.nodes[n]["x"] for n in node_ids]) - LON0) * 111320 * np.cos(np.radians(LAT0)),
    ], axis=1).astype(np.float32)

    try:
        import cupy as cp
        
        _ = cp.array([1.0], dtype=cp.float32) ** 2
        crime_gpu = cp.asarray(coords_m)       # (N_crime, 2)
        nodes_gpu = cp.asarray(node_coords_m)  # (N_nodes, 2)
        w_gpu     = cp.asarray(weights)        # (N_crime,)

        # Auto batch size: use 75 % of free VRAM.
        # Peak usage per batch = 3 × BATCH × N_crime × 4 bytes
        free_mem = cp.cuda.Device(0).mem_info[0]
        n_crime  = len(crime_gpu)
        BATCH    = int(free_mem * 0.75 / (3 * n_crime * 4))
        BATCH    = max(256, min(BATCH, len(nodes_gpu)))
        print(f"  GPU KDE: {n_crime:,} crime pts, {len(nodes_gpu):,} nodes, "
              f"batch={BATCH}, free VRAM={free_mem/1e9:.1f}GB, bw={bw_metres}m")

        t0             = time.time()
        bw2            = cp.float32(bw_metres ** 2)
        crime_sq       = cp.sum(crime_gpu ** 2, axis=1)
        density_chunks = []

        for i in range(0, len(nodes_gpu), BATCH):
            batch   = nodes_gpu[i : i + BATCH]
            node_sq = cp.sum(batch ** 2, axis=1, keepdims=True)
            cross   = batch @ crime_gpu.T
            sq_dist = node_sq + crime_sq[None, :] - 2 * cross
            gauss   = cp.exp(-0.5 * sq_dist / bw2)
            density_chunks.append(cp.asnumpy(gauss @ w_gpu))

        density = np.concatenate(density_chunks)
        print(f"  GPU KDE complete: {time.time() - t0:.1f}s")

    except (ImportError, Exception) as e:
        if not isinstance(e, ImportError):
            print(f"  CuPy runtime error ({type(e).__name__}) — falling back to scipy CPU mode...")
        else:
            print("  CuPy not found — falling back to scipy CPU mode...")
        from scipy.stats import gaussian_kde

        t0  = time.time()
        kde = gaussian_kde(
            coords_m.T,
            bw_method=bw_metres / coords_m.std(),
            weights=weights,
        )
        print(f"  CPU KDE fit complete: {time.time() - t0:.1f}s")

        print("  Scoring nodes on CPU...")
        t0      = time.time()
        density = kde(node_coords_m.T)
        print(f"  CPU node scoring complete: {time.time() - t0:.1f}s")

    crime_scores = MinMaxScaler().fit_transform(
        density.reshape(-1, 1)
    ).flatten()

    return dict(zip(node_ids, crime_scores))


# ────────────────────────────────────────────────────────────
# 3. Edge weights
# ────────────────────────────────────────────────────────────

def update_safety_weights(G: nx.MultiDiGraph, node_crime: dict, cfg: dict):
    """Recalculate edge weights in-place using additive model.

    weight = k1 * norm_length + k2 * norm_crime + k3 * (1 - norm_safety)
    k1 + k2 + k3 = 1

    Route A — length only         (weight="length",       no change needed)
    Route B — length + crime      (weight="crime_weight")
    Route C — length + crime + safety infrastructure (weight="safety_weight")
    """
    k1 = cfg["weights"]["k1"]
    k2 = cfg["weights"]["k2"]
    k3 = cfg["weights"]["k3"]

    assert abs(k1 + k2 + k3 - 1.0) < 1e-6, "k1 + k2 + k3 must equal 1.0"

    # ── Collect raw values for normalisation ──────────────────────────────
    lengths       = []
    crime_scores  = []
    safety_scores = []

    for u, v, data in G.edges(data=True):
        lengths.append(data.get("length", 1.0))
        crime_scores.append(
            (node_crime.get(u, 0) + node_crime.get(v, 0)) / 2.0
        )
        safety_scores.append(data.get("safety_density", 0.0))

    # ── Normalise each dimension to [0, 1] ────────────────────────────────
    def normalise(values):
        lo, hi = min(values), max(values)
        span = (hi - lo) or 1.0
        return [(v - lo) / span for v in values]

    norm_lengths  = normalise(lengths)
    norm_crimes   = normalise(crime_scores)
    norm_safeties = normalise(safety_scores)   # higher = denser lights/CCTV

    # ── Write weights ─────────────────────────────────────────────────────
    for (u, v, data), nl, nc, ns in zip(
        G.edges(data=True), norm_lengths, norm_crimes, norm_safeties
    ):
        # Route B: distance + crime only (k1 and k2 split evenly if k3=0,
        # but here we just use k1/(k1+k2) and k2/(k1+k2) to keep it honest)
        k12 = (k1 + k2) or 1.0
        data["crime_weight"] = (k1 / k12) * nl + (k2 / k12) * nc

        # Route C: all three factors
        # (1 - ns) because higher safety density = lower risk = lower weight
        data["safety_weight"] = k1 * nl + k2 * nc + k3 * (1 - ns)
# ────────────────────────────────────────────────────────────
# 4. Routing
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

    titles = [
        f"A  ·  Distance only\n{quick_dist:.0f}m  •  {format_time(quick_mins)} walking",
        f"B  ·  Distance + Crime\n{crime_dist:.0f}m  •  {format_time(crime_mins)} walking",
        f"C  ·  Distance + Crime + Safety\n{safe_dist:.0f}m  •  {format_time(safe_mins)} walking",
    ]
    routes  = [quickest, crime_route, safest]
    colours = ["#00d4ff", "#ff9f43", "#00ff99"]

    for ax, route, title, colour in zip(axes, routes, titles, colours):
        ox.plot_graph(
            G, ax=ax, node_size=0, edge_color="#333355",
            edge_linewidth=0.5, bgcolor="#1a1a2e", show=False, close=False,
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

    legend_handles = [
        mpatches.Patch(color="#00d4ff", label="A · Distance only"),
        mpatches.Patch(color="#ff9f43", label="B · Distance + Crime"),
        mpatches.Patch(color="#00ff99", label="C · Full safety"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="white",   markersize=8,  label="Start"),
        plt.Line2D([0],[0], marker="*", color="w", markerfacecolor="#aaaaaa", markersize=10, label="End"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="red",     markersize=6,  label="CCTV"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="yellow",  markersize=6,  label="Street light"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=7,
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
# 6. Entry point
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

    # Loading road network graph
    graph_path = resolve_path(base_dir, cfg["paths"]["graph"])
    print(f"Loading road network graph: {graph_path}")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    cctv_gdf   = gpd.read_file(resolve_path(base_dir, cfg["paths"]["cctv"]))
    lights_gdf = gpd.read_file(resolve_path(base_dir, cfg["paths"]["lights"]))

    # Computing crime density
    print("\nComputing crime density...")
    node_crime = build_crime_scores(G, cfg, base_dir)

    # Updating edge safety weights
    print("\nUpdating edge safety weights...")
    update_safety_weights(G, node_crime, cfg)

    # Save the fully weighted graph (length + crime + light + cctv)
    save_path = resolve_path(
        base_dir,
        cfg["paths"].get("weighted_graph", "bristol_weighted_graph.pkl")
    )
    print(f"\nSaving weighted graph to: {save_path}")
    with open(save_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("  Saved successfully.")

    # Finding routes
    print("\nFinding routes...")
    start_node, goal_node, quickest, crime_route, safest = find_routes(G, cfg)

    plot_routes(G, start_node, goal_node, quickest, crime_route, safest, cctv_gdf, lights_gdf, cfg)


if __name__ == "__main__":
    main()
