"""
Bristol Heat Method Router — Panel Visualisation
=================================================
Runs the full Heat Method pipeline from bristol_heat_routing.ipynb
and renders output using the 3-panel dark-theme style from safe_route.py.

Three routes are compared side by side:
  A — Distance only          (A star on edge length)
  B — Distance + Crime       (A star on length + crime penalty)
  C — Heat Method (full)     (gradient descent on φ field: distance + crime + safety)

Usage
-----
    python bristol_heat_routing2.py

Edit the coordinates and UserProfile at the top of main().

Dependencies
------------
    pip install osmnx networkx numpy scipy matplotlib geopandas scikit-learn
"""

from __future__ import annotations

import os
import pickle
import time
import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.sparse import csr_matrix, diags, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────────────────────────────────────
# FILE PATHS  (relative to this script)
# ─────────────────────────────────────────────────────────────────────────────

GRAPH_PKL  = "bristol_heat_graph.pkl"
CRIME_CSV  = "bristol_2023-01_to_2025-12.csv"
CCTV_CSV   = "Council_cctv_cameras_longlat.csv"
LIGHTS_CSV = "Council_streetlights_longlat.csv"

BUFFER_M      = 40     # metres radius for CCTV / light edge scoring
KDE_BANDWIDTH = 150    # metres for crime KDE bandwidth

SEVERITY = {
    "violent crime":         1.0,
    "robbery":               0.9,
    "burglary":              0.8,
    "vehicle crime":         0.6,
    "theft":                 0.5,
    "anti-social behaviour": 0.3,
    "other":                 0.4,
}


# ─────────────────────────────────────────────────────────────────────────────
# USER PROFILE  +  COST FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    """
    Routing preferences.  Weights are 0–1; 0 = ignore that factor.

    cost = w_distance * length
         + w_crime    * crime_score  * length
         + w_safety   * (1 - safety_score) * length
    """
    w_distance : float = 0.4
    w_crime    : float = 0.3
    w_safety   : float = 0.3
    require_lit: bool  = False

    def profile_key(self) -> str:
        d = {k: v for k, v in self.__dict__.items()}
        return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def cost_function(edge_data: dict, profile: UserProfile) -> float:
    if profile.require_lit and edge_data.get("lit", "no") == "no":
        return float("inf")
    length       = float(edge_data.get("length",       50.0))
    crime_score  = float(edge_data.get("crime_score",   0.0))
    safety_score = float(edge_data.get("safety_score",  0.0))
    return max(
        profile.w_distance * length
        + profile.w_crime  * crime_score  * length
        + profile.w_safety * (1.0 - safety_score) * length,
        1e-6,
    )


# ─────────────────────────────────────────────────────────────────────────────
# COORDINATE HELPER
# ─────────────────────────────────────────────────────────────────────────────

_LAT0, _LON0 = 51.45, -2.60

def _to_metres(lat_arr, lon_arr) -> np.ndarray:
    x = (np.asarray(lon_arr) - _LON0) * 111_320 * np.cos(np.radians(_LAT0))
    y = (np.asarray(lat_arr) - _LAT0) * 111_320
    return np.stack([x, y], axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH  —  build once, load every other time
# ─────────────────────────────────────────────────────────────────────────────

def build_or_load_graph() -> nx.MultiDiGraph:
    if os.path.exists(GRAPH_PKL):
        print(f"  Loading graph from {GRAPH_PKL} …")
        with open(GRAPH_PKL, "rb") as f:
            G = pickle.load(f)
        print(f"  {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges")
        return G

    print("  Graph not found — downloading from OSM …")
    G = ox.graph_from_place("Bristol, United Kingdom", network_type="walk")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    print(f"  Downloaded: {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges")

    # ── CCTV + lighting KD-tree scoring ──────────────────────────────────────
    print("  Scoring edges for CCTV + lighting …")
    cctv_df   = pd.read_csv(CCTV_CSV)
    lights_df = pd.read_csv(LIGHTS_CSV)
    cctv_tree  = cKDTree(_to_metres(cctv_df["latitude"].values,   cctv_df["longitude"].values))
    light_tree = cKDTree(_to_metres(lights_df["latitude"].values, lights_df["longitude"].values))

    raw_scores = {}
    total = G.number_of_edges()
    for i, (u, v, k, data) in enumerate(G.edges(keys=True, data=True)):
        if i % 10_000 == 0:
            print(f"    {i:,} / {total:,} …")
        mx  = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
        my  = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
        mid = _to_metres(np.array([my]), np.array([mx]))[0]
        n_cctv  = len(cctv_tree.query_ball_point(mid, BUFFER_M))
        n_light = len(light_tree.query_ball_point(mid, BUFFER_M))
        data["cctv_count"]  = n_cctv
        data["light_count"] = n_light
        raw_scores[(u, v, k)] = float(n_cctv * 2 + n_light)

    all_raw = list(raw_scores.values())
    span = (max(all_raw) - min(all_raw)) or 1.0
    s_min = min(all_raw)
    for u, v, k, data in G.edges(keys=True, data=True):
        data["safety_score"] = (raw_scores[(u, v, k)] - s_min) / span

    # ── Crime KDE ─────────────────────────────────────────────────────────────
    print("  Computing crime KDE …")
    crime_df = pd.read_csv(CRIME_CSV).dropna(subset=["lat", "lon"])
    crime_df["severity"] = crime_df["crime_type"].map(SEVERITY).fillna(0.4)
    weights  = crime_df["severity"].values.astype(np.float32)
    weights /= weights.sum()
    crime_m  = _to_metres(crime_df["lat"].values, crime_df["lon"].values)

    node_ids = list(G.nodes)
    node_m   = _to_metres(
        [G.nodes[n]["y"] for n in node_ids],
        [G.nodes[n]["x"] for n in node_ids],
    )
    kde     = gaussian_kde(crime_m.T, bw_method=KDE_BANDWIDTH / crime_m.std(), weights=weights)
    density = kde(node_m.T)
    scores  = MinMaxScaler().fit_transform(density.reshape(-1, 1)).flatten()
    node_crime = dict(zip(node_ids, scores))

    for u, v, data in G.edges(data=True):
        data["crime_score"] = (node_crime.get(u, 0.0) + node_crime.get(v, 0.0)) / 2.0

    # Store node-level crime scores for edge colouring later
    nx.set_node_attributes(G, node_crime, "crime_score")

    print(f"  Saving → {GRAPH_PKL}")
    with open(GRAPH_PKL, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    return G


# ─────────────────────────────────────────────────────────────────────────────
# LAPLACIAN
# ─────────────────────────────────────────────────────────────────────────────

def build_laplacian(G, profile):
    nodes    = list(G.nodes())
    n        = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}
    rows, cols, vals = [], [], []
    for u, v, data in G.edges(data=True):
        c = cost_function(data, profile)
        if c == float("inf"):
            continue
        i, j = node_idx[u], node_idx[v]
        w = 1.0 / c
        rows += [i, j, i, j]
        cols += [j, i, i, j]
        vals += [-w, -w, w, w]
    L = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return L, nodes, node_idx


# ─────────────────────────────────────────────────────────────────────────────
# HEAT METHOD  (three sparse solves)
# ─────────────────────────────────────────────────────────────────────────────

def run_heat_method(G, origin_node, profile, t_factor=1.0) -> dict:
    print("  Step 1 — heat diffusion …")
    L, nodes, node_idx = build_laplacian(G, profile)
    n   = len(nodes)
    src = node_idx[origin_node]

    costs  = [cost_function(d, profile) for _, _, d in G.edges(data=True)
              if cost_function(d, profile) < float("inf")]
    mean_c = float(np.mean(costs)) if costs else 1.0
    t      = t_factor * (mean_c ** 2)

    delta       = np.zeros(n);  delta[src] = 1.0
    I           = diags(np.ones(n), format="csr")
    t0          = time.time()
    u           = spsolve(I + t * L, delta)
    print(f"    done {time.time()-t0:.2f}s")

    print("  Step 2 — gradient normalisation …")
    div = np.zeros(n)
    for eu, ev, data in G.edges(data=True):
        c = cost_function(data, profile)
        if c == float("inf"):
            continue
        i, j = node_idx[eu], node_idx[ev]
        w    = 1.0 / c
        grad = u[j] - u[i]
        norm = abs(grad)
        if norm < 1e-12:
            continue
        X      = -grad / norm
        div[i] += w * X
        div[j] -= w * X

    print("  Step 3 — Poisson solve …")
    L_mod          = lil_matrix(L)
    div_mod        = div.copy()
    L_mod[src, :]  = 0
    L_mod[src, src] = 1
    div_mod[src]   = 0
    t0  = time.time()
    phi = spsolve(L_mod.tocsr(), div_mod)
    phi -= phi[src]
    phi  = np.abs(phi)
    print(f"    done {time.time()-t0:.2f}s")

    return {nodes[i]: phi[i] for i in range(n)}


# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT DESCENT PATH RECOVERY
# ─────────────────────────────────────────────────────────────────────────────

def recover_path(G, origin_node, dest_node, distances) -> list:
    if origin_node == dest_node:
        return [origin_node]
    path    = [dest_node]
    visited = {dest_node}
    current = dest_node
    for _ in range(len(G.nodes)):
        if current == origin_node:
            break
        phi_here   = distances.get(current, float("inf"))
        neighbours = list(G.successors(current)) + list(G.predecessors(current))
        best_node, best_phi = None, phi_here
        for nb in neighbours:
            if nb in visited:
                continue
            phi_nb = distances.get(nb, float("inf"))
            if phi_nb < best_phi:
                best_phi, best_node = phi_nb, nb
        if best_node is None:
            print(f"  ⚠  Gradient descent stuck at node {current} (φ={phi_here:.4f})")
            break
        visited.add(best_node)
        path.append(best_node)
        current = best_node
    if current != origin_node:
        print("  Falling back to Dijkstra on full cost function …")
        try:
            return nx.shortest_path(G, origin_node, dest_node,
                                    weight=lambda u, v, d: cost_function(d, UserProfile()))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    path.reverse()
    return path


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE STATS
# ─────────────────────────────────────────────────────────────────────────────

def route_stats(G, route, speed_kmh=5.0):
    total_m = sum(G[u][v][0].get("length", 0) for u, v in zip(route[:-1], route[1:]))
    minutes = (total_m / 1000 / speed_kmh) * 60
    return total_m, minutes


def format_time(minutes: float) -> str:
    if minutes < 60:
        return f"{int(round(minutes))} min"
    h, m = int(minutes // 60), int(round(minutes % 60))
    return f"{h}h {m}min"


def route_safety_stats(G, route) -> dict:
    node_crime   = nx.get_node_attributes(G, "crime_score")
    total_cctv   = total_lights = total_length = dark_length = 0
    crime_scores = []
    for u, v in zip(route[:-1], route[1:]):
        data   = G[u][v][0]
        length = data.get("length", 1.0)
        cc     = data.get("cctv_count",  0)
        lc     = data.get("light_count", 0)
        total_cctv   += cc
        total_lights += lc
        total_length += length
        if lc == 0:
            dark_length += length
        crime_scores.append((node_crime.get(u, 0) + node_crime.get(v, 0)) / 2.0)
    total_km = total_length / 1000
    return {
        "total_length":   total_length,
        "light_density":  total_lights / total_km if total_km > 0 else 0,
        "cctv_density":   total_cctv   / total_km if total_km > 0 else 0,
        "lit_coverage":   1 - dark_length / total_length if total_length > 0 else 0,
        "dark_length":    dark_length,
        "total_cctv":     total_cctv,
        "total_lights":   total_lights,
        "avg_crime_risk": float(np.mean(crime_scores)) if crime_scores else 0,
        "max_crime_risk": float(np.max(crime_scores))  if crime_scores else 0,
    }


def print_safety_analysis(stats_a, stats_b, stats_c):
    print("\n========== SAFETY ANALYSIS ==========")
    print(f"{'Metric':<25} {'A (shortest)':>14} {'B (+crime)':>14} {'C (heat)':>14}")
    print("-" * 70)
    print(f"{'Distance':<25} {stats_a['total_length']:>12.0f}m {stats_b['total_length']:>12.0f}m {stats_c['total_length']:>12.0f}m")
    print(f"{'Lights per km':<25} {stats_a['light_density']:>13.1f} {stats_b['light_density']:>13.1f} {stats_c['light_density']:>13.1f}")
    print(f"{'CCTV per km':<25} {stats_a['cctv_density']:>13.1f} {stats_b['cctv_density']:>13.1f} {stats_c['cctv_density']:>13.1f}")
    print(f"{'Lit coverage':<25} {stats_a['lit_coverage']:>13.1%} {stats_b['lit_coverage']:>13.1%} {stats_c['lit_coverage']:>13.1%}")
    print(f"{'Dark segments':<25} {stats_a['dark_length']:>12.0f}m {stats_b['dark_length']:>12.0f}m {stats_c['dark_length']:>12.0f}m")
    print(f"{'Avg crime risk':<25} {stats_a['avg_crime_risk']:>13.3f} {stats_b['avg_crime_risk']:>13.3f} {stats_c['avg_crime_risk']:>13.3f}")
    print(f"{'Max crime risk':<25} {stats_a['max_crime_risk']:>13.3f} {stats_b['max_crime_risk']:>13.3f} {stats_c['max_crime_risk']:>13.3f}")

    def pct(new, old):
        return (new - old) / max(abs(old), 1e-9) * 100

    light_d = pct(stats_c["light_density"], stats_a["light_density"])
    cctv_d  = pct(stats_c["cctv_density"],  stats_a["cctv_density"])
    lit_c   = pct(stats_c["lit_coverage"],  stats_a["lit_coverage"])
    dark_r  = pct(stats_a["dark_length"],   stats_c["dark_length"])
    crime_r = pct(stats_a["avg_crime_risk"],stats_c["avg_crime_risk"])
    detour  = pct(stats_c["total_length"],  stats_a["total_length"])

    print(f"\n  Route C vs Route A:")
    print(f"    Detour:            {detour:.1f}%")
    print(f"    Lights per km:   {'+' if light_d >= 0 else ''}{light_d:.1f}%")
    print(f"    CCTV per km:     {'+' if cctv_d  >= 0 else ''}{cctv_d:.1f}%")
    print(f"    Lit coverage:    {'+' if lit_c   >= 0 else ''}{lit_c:.1f}%")
    print(f"    Dark segments:   {'-' if dark_r  >  0 else '+'}{abs(dark_r):.1f}%")
    print(f"    Avg crime risk:  {'-' if crime_r >  0 else '+'}{abs(crime_r):.1f}%")
    print("=====================================\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3-PANEL PLOT  (safe_route.py dark-theme style)
# ─────────────────────────────────────────────────────────────────────────────

def plot_routes(G, origin_node, dest_node,
                route_a, route_b, route_c,
                cctv_gdf, lights_gdf,
                origin_name, dest_name,
                distances: dict = None):

    dist_a, mins_a = route_stats(G, route_a)
    dist_b, mins_b = route_stats(G, route_b)
    dist_c, mins_c = route_stats(G, route_c)

    # ── Crime colouring for both nodes and edges (RdYlGn_r: green→yellow→red) ─
    # Read crime scores from EDGE attributes — the graph builder stores them
    # on edges only (edge["crime_score"]).  Node attributes may not be present
    # if the pickle was built by the notebook rather than this script.
    node_list = list(G.nodes(data=True))

    # Edge crime scores straight from edge data
    edge_crime_vals = []
    for u, v, data in G.edges(data=True):
        edge_crime_vals.append(float(data.get("crime_score", 0.0)))

    # Per-node crime: average the crime scores of all adjacent edges
    node_crime_map = {}
    for n in G.nodes:
        adj = []
        for nb in list(G.successors(n)) + list(G.predecessors(n)):
            ed = G.get_edge_data(n, nb) or G.get_edge_data(nb, n)
            if ed:
                for d in ed.values():
                    adj.append(float(d.get("crime_score", 0.0)))
        node_crime_map[n] = float(np.mean(adj)) if adj else 0.0

    # Percentile-clipped normalisation — stretches contrast so hotspots
    # saturate to red instead of everything compressing into one green band
    all_crime_vals = list(node_crime_map.values()) + edge_crime_vals
    crime_arr = np.array(all_crime_vals)
    p2  = float(np.percentile(crime_arr, 2))
    p98 = float(np.percentile(crime_arr, 98))
    print(f"  Crime score range: p2={p2:.4f}  p98={p98:.4f}  max={crime_arr.max():.4f}")

    cmap_crime = plt.cm.RdYlGn_r
    norm_crime = mcolors.Normalize(vmin=p2, vmax=p98)

    node_colors = [cmap_crime(norm_crime(node_crime_map.get(n, 0.0))) for n, _ in node_list]
    edge_colors = [cmap_crime(norm_crime(c)) for c in edge_crime_vals]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(27, 9))
    fig.patch.set_facecolor("#1a1a2e")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.90, wspace=0.05)

    titles  = [
        f"A  ·  A* only\n{dist_a:.0f}m  •  {format_time(mins_a)} walking",
        f"B  ·  A* + Crime\n{dist_b:.0f}m  •  {format_time(mins_b)} walking",
        f"C  ·  Heat Method (full safety)\n{dist_c:.0f}m  •  {format_time(mins_c)} walking",
    ]
    routes  = [route_a, route_b, route_c]
    colours = ["#FFFFFF", "#FFFFFF", "#FFFFFF"]

    for ax, route, title, colour in zip(axes, routes, titles, colours):
        ox.plot_graph(
            G, ax=ax,
            node_color=node_colors,
            node_size=8,
            edge_color=edge_colors,
            edge_linewidth=0.6,
            bgcolor="#1a1a2e",
            show=False, close=False,
        )

        # Route polyline
        edge_xs, edge_ys = [], []
        for u, v in zip(route[:-1], route[1:]):
            edge_xs += [G.nodes[u]["x"], G.nodes[v]["x"], None]
            edge_ys += [G.nodes[u]["y"], G.nodes[v]["y"], None]
        ax.plot(edge_xs, edge_ys, color=colour, linewidth=4, zorder=4)

        # Start / end markers
        ax.scatter(G.nodes[origin_node]["x"], G.nodes[origin_node]["y"],
                   c="white", s=120, zorder=6, marker="o")
        ax.scatter(G.nodes[dest_node]["x"],   G.nodes[dest_node]["y"],
                   c=colour,  s=150, zorder=6, marker="*")

        # Safety infrastructure overlays
        cctv_gdf.plot(ax=ax,   color="#02CCFF7B", markersize=6,   zorder=5, alpha=0.9, marker="s")
        lights_gdf.plot(ax=ax, color="#FFFB0078", markersize=1.5, zorder=4, alpha=0.3)

        # Zoom to route bounding box
        xs = [G.nodes[n]["x"] for n in route]
        ys = [G.nodes[n]["y"] for n in route]
        margin = 0.003
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=12)
        ax.set_axis_off()

    # ── Colorbar — crime density (RdYlGn_r, far left) ────────────────────────
    sm_crime = plt.cm.ScalarMappable(cmap=cmap_crime, norm=norm_crime)
    sm_crime.set_array([])
    cbar_ax = fig.add_axes([0.015, 0.15, 0.012, 0.55])
    cb = fig.colorbar(sm_crime, cax=cbar_ax)
    cb.set_label("Crime density\n(green=low, red=high)", color="white", fontsize=10)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=9)

    # ── Legend (bottom centre, 7 columns) ────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="#FFFFFF",              label="Route"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="white",     markersize=8,  label="Start"),
        plt.Line2D([0],[0], marker="*", color="w", markerfacecolor="#aaaaaa",   markersize=10, label="End"),
        plt.Line2D([0],[0], marker="s", color="w", markerfacecolor="#02CCFF7B", markersize=6,  label="CCTV"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#FFFB0078", markersize=6,  label="Street light"),
        mpatches.Patch(color=cmap_crime(0.05),       label="Low crime"),
        mpatches.Patch(color=cmap_crime(0.95),       label="High crime"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=7,
        facecolor="#1a1a2e", labelcolor="white", fontsize=10, framealpha=0.8,
    )

    plt.suptitle(
        f"{origin_name}  →  {dest_name}",
        color="white", fontsize=16, fontweight="bold", y=0.97,
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n========== ROUTE SUMMARY ==========")
    print(f"  A  A* only:              {dist_a:.0f}m  |  {format_time(mins_a)}")
    print(f"  B  A* + Crime:           {dist_b:.0f}m  |  {format_time(mins_b)}")
    print(f"  C  Heat Method (full):   {dist_c:.0f}m  |  {format_time(mins_c)}")
    print("====================================\n")

    stats_a = route_safety_stats(G, route_a)
    stats_b = route_safety_stats(G, route_b)
    stats_c = route_safety_stats(G, route_c)
    print_safety_analysis(stats_a, stats_b, stats_c)

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Edit these ────────────────────────────────────────────────────────────
    ORIGIN_LAT, ORIGIN_LON = 51.4491, -2.5814   # Temple Meads
    DEST_LAT,   DEST_LON   = 51.4583, -2.6277   # Clifton Village
    ORIGIN_NAME = "Temple Meads"
    DEST_NAME   = "Clifton Village"

    profile = UserProfile(w_distance=0.4, w_crime=0.3, w_safety=0.3)
    # ─────────────────────────────────────────────────────────────────────────

    t_total = time.time()

    # ── Graph ─────────────────────────────────────────────────────────────────
    print("=" * 55)
    print("  Bristol Heat Method Router")
    print("=" * 55)
    print("\n[1/5] Graph …")
    G = build_or_load_graph()

    origin_node = ox.nearest_nodes(G, ORIGIN_LON, ORIGIN_LAT)
    dest_node   = ox.nearest_nodes(G, DEST_LON,   DEST_LAT)
    print(f"  Origin node: {origin_node}  |  Dest node: {dest_node}")

    # ── Safety infrastructure GeoDataFrames (for plotting overlays) ───────────
    print("\n[2/5] Loading safety overlays …")
    try:
        cctv_df   = pd.read_csv(CCTV_CSV)
        lights_df = pd.read_csv(LIGHTS_CSV)
        cctv_gdf = gpd.GeoDataFrame(
            cctv_df,
            geometry=gpd.points_from_xy(cctv_df["longitude"], cctv_df["latitude"]),
            crs="EPSG:4326",
        )
        lights_gdf = gpd.GeoDataFrame(
            lights_df,
            geometry=gpd.points_from_xy(lights_df["longitude"], lights_df["latitude"]),
            crs="EPSG:4326",
        )
        print(f"  CCTV: {len(cctv_gdf):,}  |  Lights: {len(lights_gdf):,}")
    except FileNotFoundError as e:
        print(f"  Warning: {e} — overlays will be empty")
        cctv_gdf   = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"))
        lights_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"))

    # ── Route A + B  (Dijkstra) ───────────────────────────────────────────────
    print("\n[3/5] Routes A + B (Dijkstra) …")
    try:
        route_a = nx.shortest_path(G, origin_node, dest_node, weight="length")
        print(f"  A (length):  {len(route_a)} nodes")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        print("  A: no path found")
        route_a = []

    try:
        _crime_vals = [d.get("crime_score", 0.0) for _, _, d in G.edges(data=True)]
        _hi = float(np.percentile(_crime_vals, 98)) or 1.0
        route_b = nx.astar_path(
            G, origin_node, dest_node,
            weight=lambda u, v, d: d.get("length", 50.0) * np.exp(
                2.0 * min(d.get("crime_score", 0.0) / _hi, 1.0)
            ),
        )
        print(f"  B (crime):   {len(route_b)} nodes")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        print("  B: no path found")
        route_b = route_a

    # ── Route C  (Heat Method + gradient descent) ─────────────────────────────
    print("\n[4/5] Route C — Heat Method …")
    distances = run_heat_method(G, origin_node, profile)
    print(f"  φ at destination: {distances.get(dest_node, float('inf')):.4f}")

    print("\n[5/5] Gradient descent path recovery …")
    t0      = time.time()
    route_c = recover_path(G, origin_node, dest_node, distances)
    print(f"  Done: {time.time()-t0:.3f}s  |  {len(route_c)} nodes")

    if not route_c:
        print("  No route C — using Dijkstra fallback for panel C")
        route_c = route_a

    # ── Plot ──────────────────────────────────────────────────────────────────
    print(f"\nTotal pipeline: {time.time()-t_total:.1f}s")
    plot_routes(G, origin_node, dest_node,
                route_a, route_b, route_c,
                cctv_gdf, lights_gdf,
                ORIGIN_NAME, DEST_NAME,
                distances=distances)


if __name__ == "__main__":
    main()
