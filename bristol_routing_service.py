"""
Bristol Road Network — On-Demand Routing Service (Heat Method)
==============================================================
Wraps the Heat Method geodesic solver in a caching query interface.

Usage:
    service = BristolRoutingService(risk_level="low")
    result  = service.query(
                  origin_lat=51.4545, origin_lon=-2.5879,
                  dest_lat=51.4655,   dest_lon=-2.6020,
              )
    print(result)

    # Interactive CLI:
    python bristol_routing_service.py

Dependencies:
    pip install osmnx networkx numpy scipy matplotlib
"""

import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.sparse import csr_matrix, diags, lil_matrix
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass
from typing import Optional
import time
import folium
import branca.colormap as cm

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

RISK_CONFIGS = {
    "low": {
        "network_type": "drive",
        "description" : "Full drive network, standard travel time",
        "penalties"   : {},
        "weight_attr" : "travel_time",
    },
    "medium": {
        "network_type": "drive",
        "description" : "Drive network — fast roads penalised",
        "penalties"   : {60: 3.0, 40: 1.5},
        "weight_attr" : "travel_time",
    },
    "high": {
        "network_type": "walk",
        "description" : "Pedestrian network only (safest routes)",
        "penalties"   : {},
        "weight_attr" : "length",
    },
}


# ─────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """Returned by BristolRoutingService.query()"""
    origin_node     : int
    dest_node       : int
    travel_time_s   : float
    travel_time_min : float
    risk_level      : str
    cache_hit       : bool          # True if distances were already cached

    def __str__(self):
        src = "cache" if self.cache_hit else "fresh solve"
        return (
            f"\n  Origin node   : {self.origin_node}"
            f"\n  Dest node     : {self.dest_node}"
            f"\n  Travel time   : {self.travel_time_min:.1f} min  "
            f"({self.travel_time_s:.0f} s)"
            f"\n  Risk level    : {self.risk_level}"
            f"\n  Distances from: {src}"
        )


# ─────────────────────────────────────────────────────────────
# ground helpers
# ─────────────────────────────────────────────────────────────

def _build_graph(risk_level: str):
    """Download Bristol graph and apply risk weighting."""
    cfg = RISK_CONFIGS[risk_level]
    print(f"  Downloading Bristol graph  (risk={risk_level}: {cfg['description']})")

    G = ox.graph_from_place("Bristol, England", network_type=cfg["network_type"])
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    for _, _, data in G.edges(data=True):
        base    = float(data.get(cfg["weight_attr"], 1.0))
        penalty = 1.0
        if cfg["penalties"]:
            speed = data.get("speed_kph", 30.0)
            for threshold, mult in sorted(cfg["penalties"].items(), reverse=True):
                if speed >= threshold:
                    penalty = mult
                    break
        data["risk_weight"] = max(base * penalty, 1e-6)

    return G


def _build_laplacian(G):
    """Conductance-weighted Laplacian: w_ij = 1 / risk_weight."""
    nodes    = list(G.nodes())
    n        = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}

    rows, cols, vals = [], [], []
    for u, v, data in G.edges(data=True):
        i = node_idx[u]
        j = node_idx[v]
        w = 1.0 / data["risk_weight"]

        rows += [i, j, i, j]
        cols += [j, i, i, j]
        vals += [-w, -w, w, w]

    L = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return L, nodes, node_idx


def _heat_method(G, source_node: int, t_factor: float = 1.0) -> dict:
    """
    Compute geodesic distances from source_node via the Heat Method.

    Returns {node_id: distance_float}
    """
    L, nodes, node_idx = _build_laplacian(G)
    n   = len(nodes)
    src = node_idx[source_node]

    # ── A: implicit Euler heat diffusion ────────────────────────────────
    delta  = np.zeros(n)
    delta[src] = 1.0

    mean_w = np.mean([d["risk_weight"] for _, _, d in G.edges(data=True)])
    t      = t_factor * (mean_w ** 2)

    I      = diags(np.ones(n), format="csr")
    u      = spsolve(I + t * L, delta)

    # ── B: normalised gradient → nodal divergence ────────────────────────
    div = np.zeros(n)
    for eu, ev, data in G.edges(data=True):
        i    = node_idx[eu]
        j    = node_idx[ev]
        w    = 1.0 / data["risk_weight"]
        grad = u[j] - u[i]
        norm = abs(grad)
        if norm < 1e-12:
            continue
        X      = -grad / norm
        div[i] += w * X
        div[j] -= w * X

    # ── C: Poisson solve ─────────────────────────────────────────────────
    L_mod         = lil_matrix(L)
    div_mod       = div.copy()
    L_mod[src, :] = 0
    L_mod[src, src] = 1
    div_mod[src]  = 0

    phi  = spsolve(L_mod.tocsr(), div_mod)
    phi -= phi[src]
    phi  = np.abs(phi)

    return {nodes[i]: phi[i] for i in range(n)}


def visualise_interactive(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: Optional[float] = None,
        dest_lon: Optional[float] = None,
        output_file: str = "bristol_heatmap.html",
):
    """
    Render an interactive Folium map with:
      - Nodes coloured by geodesic travel time (plasma palette)
      - Optional route polyline with tooltip
      - Origin (red) and destination (green) markers

    Open the output HTML in any browser — no server needed.
    Requires: pip install folium branca
    """

    origin_node = ox.nearest_nodes(self._G, origin_lon, origin_lat)

    # Ensure distances are computed
    if origin_node not in self._cache:
        self.query(origin_lat, origin_lon, origin_lat, origin_lon)

    distances = self._cache[origin_node]

    # ── Colour scale ─────────────────────────────────────────────────────
    dist_vals = [v for v in distances.values() if v < float("inf")]
    d_min, d_max = min(dist_vals), max(dist_vals)

    colormap = cm.LinearColormap(
        colors=["#0d0887", "#7e03a8", "#cc4778", "#f89540", "#f0f921"],
        vmin=d_min,
        vmax=d_max,
        caption="Travel time from origin (seconds)",
    )

    # ── Base map ──────────────────────────────────────────────────────────
    fmap = folium.Map(
        location=[origin_lat, origin_lon],
        zoom_start=13,
        tiles="CartoDB dark_matter",
    )
    colormap.add_to(fmap)

    # ── Node circles coloured by travel time ──────────────────────────────
    for node, data in self._G.nodes(data=True):
        dist = distances.get(node, None)
        if dist is None or dist == float("inf"):
            continue
        color = colormap(dist)
        folium.CircleMarker(
            location=(data["y"], data["x"]),
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=0,
            tooltip=f"Node {node}<br>{dist / 60:.1f} min ({dist:.0f} s)",
        ).add_to(fmap)

    # ── Route polyline ────────────────────────────────────────────────────
    if dest_lat is not None and dest_lon is not None:
        dest_node = ox.nearest_nodes(self._G, dest_lon, dest_lat)
        path_nodes = _recover_path(self._G, origin_node, dest_node, distances)

        if path_nodes:
            travel_s = distances.get(dest_node, float("inf"))
            travel_min = travel_s / 60.0

            # Collect lat/lon for each node in the path
            path_coords = [
                (self._G.nodes[n]["y"], self._G.nodes[n]["x"])
                for n in path_nodes
            ]

            # Glow effect — thick dim underline + bright topline
            folium.PolyLine(
                path_coords,
                color="#006666",
                weight=8,
                opacity=0.5,
            ).add_to(fmap)

            folium.PolyLine(
                path_coords,
                color="#00e5ff",
                weight=3,
                opacity=0.95,
                tooltip=f"Route: {travel_min:.1f} min ({travel_s:.0f} s)",
            ).add_to(fmap)

        # ── Destination marker ────────────────────────────────────────────
        folium.Marker(
            location=(dest_lat, dest_lon),
            tooltip=f"Destination<br>{distances.get(dest_node, float('inf')) / 60:.1f} min",
            icon=folium.Icon(color="green", icon="flag", prefix="fa"),
        ).add_to(fmap)

    # ── Origin marker ─────────────────────────────────────────────────────
    folium.Marker(
        location=(origin_lat, origin_lon),
        tooltip="Origin",
        icon=folium.Icon(color="red", icon="circle", prefix="fa"),
    ).add_to(fmap)

    fmap.save(output_file)
    print(f"  Saved → {output_file}  (open in any browser)")
    return fmap  # handy if running inside a Jupyter notebook

# ─────────────────────────────────────────────────────────────
# PATH RECOVERY — GRADIENT DESCENT ON φ
# ─────────────────────────────────────────────────────────────

def _recover_path(G, origin_node: int, dest_node: int,
                  distances: dict) -> list[int]:
    """
    Trace the suggested path from dest_node back to origin_node by
    following the steepest descent of the geodesic distance field φ.

    Why this instead of Dijkstra?
    ------------------------------
    The Heat Method already computed φ (the Poisson distance field)
    for every node. Since φ is a geodesic distance function, its
    gradient ∇φ points *away* from the source everywhere on the graph.
    The path from any destination back to the origin is therefore the
    integral curve of −∇φ — i.e. we just step greedily to whichever
    neighbour has the lowest φ value, weighted by edge conductance.

    On a graph edge (i → j) the discrete gradient is:

        ∇φ_ij = (φ_j − φ_i) / risk_weight_ij

    and steepest descent picks:

        n* = argmin_{j ∈ N(i)}  φ_j

    This keeps path extraction entirely within the Heat Method framework
    — no second algorithm, no extra traversal, and the path is
    geometrically consistent with the distance field.

    Guard against cycles
    --------------------
    In flat or near-flat regions of φ (nodes with nearly equal distances)
    the greedy step could revisit nodes. A visited set breaks any cycle.

    Returns
    -------
    list of node IDs from origin to destination (inclusive),
    or [] if the origin cannot be reached.
    """
    path    = [dest_node]
    visited = {dest_node}
    current = dest_node

    max_steps = len(G.nodes)   # hard cap — can't need more steps than nodes

    for _ in range(max_steps):
        if current == origin_node:
            break

        phi_current = distances.get(current, float("inf"))

        # Collect all neighbours reachable via graph edges
        # G.predecessors covers both directions for undirected-style traversal
        neighbours = list(G.successors(current)) + list(G.predecessors(current))

        best_node  = None
        best_phi   = phi_current   # only move to strictly lower φ

        for nb in neighbours:
            if nb in visited:
                continue

            # Edge weight: use the minimum risk_weight across parallel edges
            edge_data = G.get_edge_data(current, nb) or G.get_edge_data(nb, current)
            if edge_data is None:
                continue

            # MultiDiGraph stores edges as {0: data, 1: data, ...}
            risk_w = min(
                d.get("risk_weight", 1.0)
                for d in (edge_data.values()
                          if isinstance(edge_data, dict) else [edge_data])
            )

            phi_nb = distances.get(nb, float("inf"))

            # Normalised descent: steepest drop per unit risk
            descent = (phi_current - phi_nb) / risk_w

            if phi_nb < best_phi or (best_node is None and phi_nb < phi_current):
                best_phi  = phi_nb
                best_node = nb

        if best_node is None:
            # Stuck in a local minimum — no neighbour is closer to origin
            print(f"  ⚠ Gradient descent stuck at node {current} "
                  f"(φ={phi_current:.2f}). Path may be incomplete.")
            break

        path.append(best_node)
        visited.add(best_node)
        current = best_node

    if current != origin_node:
        return []   # failed to reach origin

    # Path was built destination → origin; reverse for origin → destination
    path.reverse()
    return path


# ─────────────────────────────────────────────────────────────
# ROUTING SERVICE
# ─────────────────────────────────────────────────────────────

class RoutingService:
    """
    On-demand geodesic routing service backed by the Heat Method.

    Cache behaviour
    ---------------
    Each unique (origin_node, risk_level) pair triggers one full
    Heat Method solve (~3 sparse linear systems).  Results are kept
    in memory; subsequent queries from the same origin are O(1)
    dictionary lookups.

    The cache is intentionally unbounded for simplicity. In production
    you would replace self._cache with an LRU cache (functools.lru_cache
    or cachetools.LRUCache) sized to your RAM budget.

    Parameters
    ----------
    risk_level : 'low' | 'medium' | 'high'
    t_factor   : Heat timestep scale (1.0 is a good default)
    """

    def __init__(self, risk_level: str = "low", t_factor: float = 1.0):
        if risk_level not in RISK_CONFIGS:
            raise ValueError(f"risk_level must be one of {list(RISK_CONFIGS)}")

        self.risk_level = risk_level
        self.t_factor   = t_factor
        self._cache: dict[int, dict] = {}   # {source_node_id: distances_dict}

        print(f"[BristolRoutingService] Loading graph (risk={risk_level}) ...")
        self._G = _build_graph(risk_level)
        print(f"  Graph ready: {len(self._G.nodes):,} nodes / "
              f"{len(self._G.edges):,} edges\n")

    # ── Public API ───────────────────────────────────────────────────────

    def query(
        self,
        origin_lat: float, origin_lon: float,
        dest_lat:   float, dest_lon:   float,
    ) -> QueryResult:
        """
        Return travel time between two (lat, lon) coordinates.

        Steps
        -----
        1. Snap both coordinates to nearest graph nodes.
        2. If distances from origin are not cached, run Heat Method.
        3. Look up destination in the cached distances dict.
        4. Return a QueryResult with time in seconds and minutes.
        """
        origin_node = ox.nearest_nodes(self._G, origin_lon, origin_lat)
        dest_node   = ox.nearest_nodes(self._G, dest_lon,   dest_lat)

        cache_hit = origin_node in self._cache

        if not cache_hit:
            print(f"  Cache MISS — solving Heat Method from node {origin_node} ...")
            t0 = time.perf_counter()
            self._cache[origin_node] = _heat_method(
                self._G, origin_node, self.t_factor
            )
            elapsed = time.perf_counter() - t0
            print(f"  Solve complete in {elapsed:.2f}s — result cached.\n")
        else:
            print(f"  Cache HIT  — distances from node {origin_node} ready.\n")

        travel_time_s = self._cache[origin_node].get(dest_node, float("inf"))

        return QueryResult(
            origin_node     = origin_node,
            dest_node       = dest_node,
            travel_time_s   = travel_time_s,
            travel_time_min = travel_time_s / 60.0,
            risk_level      = self.risk_level,
            cache_hit       = cache_hit,
        )

    def batch_query(
        self,
        origin_lat: float, origin_lon: float,
        destinations: list[tuple[float, float]],
    ) -> list[QueryResult]:
        """
        Query multiple destinations from ONE origin in a single Heat Method solve.

        This is the primary efficiency advantage of Option 1:
        one solve, N lookups.

        Parameters
        ----------
        destinations : list of (lat, lon) tuples
        """
        # Trigger solve once (or use cache)
        first_dest_lat, first_dest_lon = destinations[0]
        self.query(origin_lat, origin_lon, first_dest_lat, first_dest_lon)

        # All subsequent lookups are free cache hits
        return [
            self.query(origin_lat, origin_lon, dlat, dlon)
            for dlat, dlon in destinations
        ]

    def cache_info(self) -> dict:
        """Return a summary of what is currently cached."""
        return {
            "cached_origins" : len(self._cache),
            "node_ids"       : list(self._cache.keys()),
        }

    def clear_cache(self):
        """Evict all cached distance maps."""
        self._cache.clear()
        print("  Cache cleared.")

    def visualise_from(
        self,
        origin_lat:  float,
        origin_lon:  float,
        dest_lat:    Optional[float] = None,
        dest_lon:    Optional[float] = None,
        output_file: str = "bristol_heatmap.png",
    ):
        """
        Render a heatmap of geodesic travel times from the origin.

        If dest_lat / dest_lon are supplied the suggested path is overlaid:
          • The path edges are drawn as a bright cyan polyline.
          • Origin is marked with a red circle.
          • Destination is marked with a green diamond.
          • An info box shows the travel time for the route.

        Parameters
        ----------
        origin_lat / origin_lon : source coordinates
        dest_lat   / dest_lon   : (optional) destination coordinates
        output_file             : output PNG path
        """
        origin_node = ox.nearest_nodes(self._G, origin_lon, origin_lat)

        # ── Ensure distances are computed ────────────────────────────────
        if origin_node not in self._cache:
            self.query(origin_lat, origin_lon, origin_lat, origin_lon)

        distances = self._cache[origin_node]

        # ── Base heatmap ─────────────────────────────────────────────────
        node_list = list(self._G.nodes(data=True))
        dist_vals = np.array([distances.get(n, np.nan) for n, _ in node_list])

        valid = ~np.isnan(dist_vals)
        d_min, d_max = dist_vals[valid].min(), dist_vals[valid].max()
        norm = mcolors.Normalize(vmin=d_min, vmax=d_max)
        cmap = plt.cm.plasma

        node_colors = [cmap(norm(distances.get(n, d_max))) for n, _ in node_list]

        fig, ax = ox.plot_graph(
            self._G,
            node_color=node_colors,
            node_size=8,
            edge_color="#2a2a2a",
            edge_linewidth=0.4,
            bgcolor="#111111",
            show=False,
            close=False,
        )

        # ── Origin marker ────────────────────────────────────────────────
        sx = self._G.nodes[origin_node]["x"]
        sy = self._G.nodes[origin_node]["y"]
        ax.scatter([sx], [sy], c="#ff4444", s=160, zorder=8,
                   marker="o", edgecolors="white", linewidths=1.2)

        # ── Optional destination + path overlay ──────────────────────────
        legend_handles = [
            mpatches.Patch(color="#ff4444", label="Origin"),
        ]

        if dest_lat is not None and dest_lon is not None:
            dest_node = ox.nearest_nodes(self._G, dest_lon, dest_lat)

            # Destination marker
            dx = self._G.nodes[dest_node]["x"]
            dy = self._G.nodes[dest_node]["y"]
            ax.scatter([dx], [dy], c="#00ff99", s=220, zorder=8,
                       marker="D", edgecolors="white", linewidths=1.2)

            # Recover and draw path
            path_nodes = _recover_path(self._G, origin_node, dest_node, distances)

            if path_nodes:
                # Collect (x, y) for each node in the path
                xs = [self._G.nodes[n]["x"] for n in path_nodes]
                ys = [self._G.nodes[n]["y"] for n in path_nodes]

                # Glowing effect: thick dim underline + bright topline
                ax.plot(xs, ys, color="#006666", linewidth=5,
                        zorder=5, solid_capstyle="round")
                ax.plot(xs, ys, color="#00e5ff", linewidth=2.2,
                        zorder=6, solid_capstyle="round",
                        label="Suggested path")

                # Travel time annotation box
                travel_s   = distances.get(dest_node, float("inf"))
                travel_min = travel_s / 60.0
                info_text  = f"Route: {travel_min:.1f} min  ({travel_s:.0f} s)"
                ax.text(
                    0.5, 0.04, info_text,
                    transform=ax.transAxes,
                    ha="center", va="bottom",
                    fontsize=10, color="white",
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="#1a1a2e", edgecolor="#00e5ff",
                              linewidth=1.2, alpha=0.88),
                    zorder=10,
                )

                legend_handles.append(
                    mpatches.Patch(color="#00e5ff", label="Suggested path")
                )
            else:
                ax.text(0.5, 0.04, "No path found between nodes",
                        transform=ax.transAxes, ha="center",
                        color="#ff6666", fontsize=9)

            legend_handles.append(
                mpatches.Patch(color="#00ff99", label="Destination")
            )

        # ── Legend ───────────────────────────────────────────────────────
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            facecolor="#1a1a2e",
            edgecolor="#444",
            labelcolor="white",
            fontsize=9,
        )

        # ── Colourbar ────────────────────────────────────────────────────
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label("Travel time from origin (seconds)",
                     color="white", fontsize=10)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

        # ── Title ────────────────────────────────────────────────────────
        cfg = RISK_CONFIGS[self.risk_level]
        ax.set_title(
            f"Bristol — Geodesic Travel Times from Origin\n"
            f"Risk: {self.risk_level}  ({cfg['description']})",
            color="white", fontsize=11, pad=10,
        )

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight",
                    facecolor="#111111")
        print(f"  Saved → {output_file}")
        plt.show()

    def visualise_interactive(
            self,
            origin_lat: float,
            origin_lon: float,
            dest_lat: Optional[float] = None,
            dest_lon: Optional[float] = None,
            output_file: str = "bristol_heatmap.html",
    ):
        """
        Render an interactive Folium map with:
          - Nodes coloured by geodesic travel time (plasma palette)
          - Optional route polyline with tooltip
          - Origin (red) and destination (green) markers

        Open the output HTML in any browser — no server needed.
        Requires: pip install folium branca
        """

        origin_node = ox.nearest_nodes(self._G, origin_lon, origin_lat)

        # Ensure distances are computed
        if origin_node not in self._cache:
            self.query(origin_lat, origin_lon, origin_lat, origin_lon)

        distances = self._cache[origin_node]

        # ── Colour scale ─────────────────────────────────────────────────────
        dist_vals = [v for v in distances.values() if v < float("inf")]
        d_min, d_max = min(dist_vals), max(dist_vals)

        colormap = cm.LinearColormap(
            colors=["#0d0887", "#7e03a8", "#cc4778", "#f89540", "#f0f921"],
            vmin=d_min,
            vmax=d_max,
            caption="Travel time from origin (seconds)",
        )

        # ── Base map ──────────────────────────────────────────────────────────
        fmap = folium.Map(
            location=[origin_lat, origin_lon],
            zoom_start=13,
            tiles="CartoDB dark_matter",
        )
        colormap.add_to(fmap)

        # ── Node circles coloured by travel time ──────────────────────────────
        for node, data in self._G.nodes(data=True):
            dist = distances.get(node, None)
            if dist is None or dist == float("inf"):
                continue
            color = colormap(dist)
            folium.CircleMarker(
                location=(data["y"], data["x"]),
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                weight=0,
                tooltip=f"Node {node}<br>{dist / 60:.1f} min ({dist:.0f} s)",
            ).add_to(fmap)

        # ── Route polyline ────────────────────────────────────────────────────
        if dest_lat is not None and dest_lon is not None:
            dest_node = ox.nearest_nodes(self._G, dest_lon, dest_lat)
            path_nodes = _recover_path(self._G, origin_node, dest_node, distances)

            if path_nodes:
                travel_s = distances.get(dest_node, float("inf"))
                travel_min = travel_s / 60.0

                # Collect lat/lon for each node in the path
                path_coords = [
                    (self._G.nodes[n]["y"], self._G.nodes[n]["x"])
                    for n in path_nodes
                ]

                # Glow effect — thick dim underline + bright topline
                folium.PolyLine(
                    path_coords,
                    color="#006666",
                    weight=8,
                    opacity=0.5,
                ).add_to(fmap)

                folium.PolyLine(
                    path_coords,
                    color="#00e5ff",
                    weight=3,
                    opacity=0.95,
                    tooltip=f"Route: {travel_min:.1f} min ({travel_s:.0f} s)",
                ).add_to(fmap)

            # ── Destination marker ────────────────────────────────────────────
            folium.Marker(
                location=(dest_lat, dest_lon),
                tooltip=f"Destination<br>{distances.get(dest_node, float('inf')) / 60:.1f} min",
                icon=folium.Icon(color="green", icon="flag", prefix="fa"),
            ).add_to(fmap)

        # ── Origin marker ─────────────────────────────────────────────────────
        folium.Marker(
            location=(origin_lat, origin_lon),
            tooltip="Origin",
            icon=folium.Icon(color="red", icon="circle", prefix="fa"),
        ).add_to(fmap)

        fmap.save(output_file)
        print(f"  Saved → {output_file}  (open in any browser)")
        return fmap


# ─────────────────────────────────────────────────────────────
# INTERACTIVE CLI
# ─────────────────────────────────────────────────────────────

BRISTOL_LANDMARKS = {
    "temple meads"   : (51.4493, -2.5815),
    "clifton village": (51.4579, -2.6233),
    "broadmead"      : (51.4578, -2.5908),
    "harbourside"    : (51.4497, -2.5985),
    "whiteladies rd" : (51.4660, -2.6075),
    "bedminster"     : (51.4384, -2.5919),
    "stokes croft"   : (51.4626, -2.5923),
    "filton"         : (51.5072, -2.5688),
}


def _interactive_cli():
    print("=" * 60)
    print("  Bristol Routing Service — Interactive CLI")
    print("=" * 60)

    #risk = input("\nRisk level (low / medium / high) [low]: ").strip() or "low"
    risk = "low"
    service = RoutingService(risk_level=risk)

    print("\nKnown landmarks:")
    for name, (lat, lon) in BRISTOL_LANDMARKS.items():
        print(f"  {name:<20} ({lat}, {lon})")

    while True:
        print("\n" + "-" * 40)
        print("Enter origin (name or lat,lon) — or 'quit':")
        #origin_input = input("  Origin > ").strip().lower()
        origin_input = "51.45598, -h2.60469"
        if origin_input in ("quit", "q", "exit"):
            break

        print("Enter destination (name or lat,lon):")
        #dest_input = input("  Dest   > ").strip().lower()
        dest_input = "51.44906, -2.59425"

        def parse_location(s):
            if s in BRISTOL_LANDMARKS:
                return BRISTOL_LANDMARKS[s]
            try:
                lat, lon = map(float, s.split(","))
                return lat, lon
            except Exception:
                print(f"  ✗ Could not parse '{s}'. Use a landmark name or 'lat,lon'.")
                return None

        origin = parse_location(origin_input)
        dest   = parse_location(dest_input)
        if origin is None or dest is None:
            continue

        result = service.query(
            origin_lat=origin[0], origin_lon=origin[1],
            dest_lat=dest[0],     dest_lon=dest[1],
        )
        print(result)

        # Inside _interactive_cli(), replace the viz block:
        viz = input("\n  Render map? (png / html / n) [n]: ").strip().lower()
        if viz == "png":
            service.visualise_from(origin[0], origin[1], dest[0], dest[1])
        elif viz == "html":
            service.visualise_interactive(origin[0], origin[1], dest[0], dest[1])

        cache = service.cache_info()
        print(f"\n  Cache: {cache['cached_origins']} origin(s) stored.")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _interactive_cli()
