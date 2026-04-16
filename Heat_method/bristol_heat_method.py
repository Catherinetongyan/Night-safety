"""
Bristol Road Network — Heat Method for Geodesic Distances
==========================================================
Implements the discrete Heat Method (Crane et al., 2017) on the
OSMnx road graph of Bristol. The "manifold" is the weighted graph,
so switching risk levels simply rebuilds the Laplacian with different
edge weights — the algorithm itself is unchanged.

Dependencies:
    pip install osmnx networkx numpy scipy matplotlib

Usage:
    python bristol_heat_method.py
    # Edit RISK_LEVEL below to 'low', 'medium', or 'high'
"""

import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.sparse import csr_matrix, diags, lil_matrix
from scipy.sparse.linalg import spsolve

# ─────────────────────────────────────────────
# CONFIGURATION — change these as needed
# ─────────────────────────────────────────────

RISK_LEVEL   = "low"          # 'low' | 'medium' | 'high'
SOURCE_LAT   = 51.4545        # Bristol city centre (lat)
SOURCE_LON   = -2.5879        # Bristol city centre (lon)
T_FACTOR     = 1.0            # Heat timestep scale (larger = smoother)
OUTPUT_FILE  = "bristol_heatmap.png"


# ─────────────────────────────────────────────
# 1. DOWNLOAD & PREPARE GRAPH
# ─────────────────────────────────────────────

RISK_CONFIGS = {
    "low": {
        "network_type": "drive",
        "description" : "Full drive network, unweighted travel time",
        "penalties"   : {},         # speed_kph threshold → multiplier
        "weight_attr" : "travel_time",
    },
    "medium": {
        "network_type": "drive",
        "description" : "Drive network, fast roads penalised",
        "penalties"   : {60: 3.0, 40: 1.5},   # ≥60 kph → 3×, ≥40 → 1.5×
        "weight_attr" : "travel_time",
    },
    "high": {
        "network_type": "walk",
        "description" : "Pedestrian network only (safest routes)",
        "penalties"   : {},
        "weight_attr" : "length",   # walk graph may not have travel_time
    },
}


def get_bristol_graph(risk_level: str):
    """
    Download the Bristol street network and apply risk-level weighting.

    The 'manifold' is defined by how we weight graph edges:
      - low    →  normal travel-time weights (Euclidean-like geometry)
      - medium →  penalise arterial / motorway speeds (stretches fast roads)
      - high   →  pedestrian-only network (removes fast road topology entirely)

    Returns a projected MultiDiGraph with a 'risk_weight' edge attribute.
    """
    cfg = RISK_CONFIGS[risk_level]
    print(f"[1/4] Downloading Bristol graph  (risk={risk_level}: {cfg['description']})")

    G = ox.graph_from_place("Bristol, England", network_type=cfg["network_type"])
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    # Apply risk penalties to edge weights
    for _, _, data in G.edges(data=True):
        base = float(data.get(cfg["weight_attr"], 1.0))
        penalty = 1.0
        if cfg["penalties"]:
            speed = data.get("speed_kph", 30.0)
            for threshold, mult in sorted(cfg["penalties"].items(), reverse=True):
                if speed >= threshold:
                    penalty = mult
                    break
        data["risk_weight"] = max(base * penalty, 1e-6)

    return G


# ─────────────────────────────────────────────
# 2. BUILD WEIGHTED GRAPH LAPLACIAN
# ─────────────────────────────────────────────

def build_laplacian(G):
    """
    Construct the cotangent-style (conductance) weighted Laplacian:

        L[i,j] = -w(i,j)   if (i,j) is an edge
        L[i,i] = Σ_j w(i,j)

    where w(i,j) = 1 / risk_weight, so high-risk (long) edges get LOW
    conductance, which spreads heat slowly across them — accurately
    reflecting that they are "far" in the risk geometry.

    Returns (L, nodes, node_to_index).
    """
    nodes     = list(G.nodes())
    n         = len(nodes)
    node_idx  = {node: i for i, node in enumerate(nodes)}

    rows, cols, vals = [], [], []

    for u, v, data in G.edges(data=True):
        i = node_idx[u]
        j = node_idx[v]
        w = 1.0 / data["risk_weight"]   # conductance = inverse resistance

        # Off-diagonal entries
        rows += [i, j]
        cols += [j, i]
        vals += [-w, -w]

        # Diagonal accumulation
        rows += [i, j]
        cols += [i, j]
        vals += [w, w]

    L = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return L, nodes, node_idx


# ─────────────────────────────────────────────
# 3. HEAT METHOD (Crane et al., 2017)
# ─────────────────────────────────────────────

def heat_method(G, source_node: int, t_factor: float = 1.0):
    """
    Compute geodesic distances from `source_node` using the Heat Method.

    Algorithm
    ---------
    Let L = weighted graph Laplacian, n = number of nodes.

    Step A — Heat diffusion (implicit Euler):
        Solve  (I + t·L) u = δ_s
        where δ_s is a unit impulse at the source, t = t_factor·h²,
        h = mean edge weight (characteristic length of the manifold).

    Step B — Normalise gradient:
        For each directed edge (i→j):
            X_ij = -(u_j - u_i) / |u_j - u_i|   (unit vector away from source)

    Step C — Recover distance via Poisson solve:
        Compute nodal divergence: div_i = Σ_j w_ij · X_ij
        Solve  L φ = div  (with source pinned to 0)
        Return φ shifted so φ[source] = 0, φ ≥ 0.

    Why does this work?
    -------------------
    The heat kernel u(x, t) from a point source concentrates near the source
    for small t. Its gradient therefore points away from the source, and
    after normalisation forms a unit vector field X that mimics ∇φ/|∇φ|
    for the true distance function φ. The Poisson step recovers φ from X.

    Parameters
    ----------
    G           : NetworkX graph with 'risk_weight' edge attribute
    source_node : node ID to measure distances from
    t_factor    : timestep scaling (larger → smoother, less accurate; ~1 is good)

    Returns
    -------
    dict  {node_id: float geodesic_distance}
    """
    print("[2/4] Building graph Laplacian ...")
    L, nodes, node_idx = build_laplacian(G)
    n = len(nodes)

    src = node_idx[source_node]

    # ── Step A: heat diffusion ──────────────────────────────────────────
    print("[3/4] Running Heat Method (3 linear solves) ...")

    delta = np.zeros(n)
    delta[src] = 1.0

    mean_w = np.mean([d["risk_weight"] for _, _, d in G.edges(data=True)])
    t      = t_factor * (mean_w ** 2)

    I      = diags(np.ones(n), format="csr")
    A_heat = I + t * L
    u      = spsolve(A_heat, delta)  # implicit Euler discretisation

    # ── Step B: per-edge normalised gradient → nodal divergence ─────────
    div = np.zeros(n)
    for eu, ev, data in G.edges(data=True):
        i = node_idx[eu]
        j = node_idx[ev]
        w = 1.0 / data["risk_weight"]

        grad = u[j] - u[i]
        norm = abs(grad)
        if norm < 1e-12:
            continue

        # Unit vector pointing away from the source along this edge
        X = -grad / norm

        # Weighted divergence contribution (signed, so it sums correctly)
        div[i] += w * X
        div[j] -= w * X

    # ── Step C: Poisson solve for distance ───────────────────────────────
    # Pin source row to prevent singular system
    L_mod      = lil_matrix(L)
    div_mod    = div.copy()
    L_mod[src, :]   = 0
    L_mod[src, src] = 1
    div_mod[src]    = 0

    phi = spsolve(L_mod.tocsr(), div_mod)

    # Shift & rectify
    phi -= phi[src]
    phi  = np.abs(phi)

    return {nodes[i]: phi[i] for i in range(n)}


# ─────────────────────────────────────────────
# 4. VISUALISATION
# ─────────────────────────────────────────────

def visualise(G, distances: dict, source_node: int, risk_level: str):
    """
    Render the Bristol graph with nodes coloured by geodesic distance.
    Warm colours = close to source, cool colours = far.
    """
    print("[4/4] Rendering map ...")

    node_list = list(G.nodes(data=True))
    dist_vals = np.array([distances.get(n, np.nan) for n, _ in node_list])

    valid = ~np.isnan(dist_vals)
    d_min, d_max = dist_vals[valid].min(), dist_vals[valid].max()
    norm  = mcolors.Normalize(vmin=d_min, vmax=d_max)
    cmap  = plt.cm.plasma

    node_colors = [cmap(norm(distances.get(n, d_max))) for n, _ in node_list]

    fig, ax = ox.plot_graph(
        G,
        node_color=node_colors,
        node_size=8,
        edge_color="#2a2a2a",
        edge_linewidth=0.4,
        bgcolor="#111111",
        show=False,
        close=False,
    )

    # Mark the source node
    sx = G.nodes[source_node]["x"]
    sy = G.nodes[source_node]["y"]
    ax.scatter([sx], [sy], c="red", s=120, zorder=6, label="Source node")
    ax.legend(loc="upper left", facecolor="#222", labelcolor="white", fontsize=9)

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Geodesic distance (heat method)", color="white", fontsize=10)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    risk_label = RISK_CONFIGS[risk_level]["description"]
    ax.set_title(
        f"Bristol — Heat Method Geodesic Distances\nRisk level: {risk_level}  ({risk_label})",
        color="white", fontsize=11, pad=10
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor="#111111")
    print(f"    Saved → {OUTPUT_FILE}")
    plt.show()


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

def main():
    # ── Download & weight the graph (manifold) ─────────────────────────
    G = get_bristol_graph(RISK_LEVEL)

    # ── Snap source coordinates to nearest graph node ──────────────────
    source_node = ox.nearest_nodes(G, SOURCE_LON, SOURCE_LAT)
    print(f"    Source node ID : {source_node}")
    print(f"    Graph size     : {len(G.nodes):,} nodes  /  {len(G.edges):,} edges")

    # ── Run heat method ────────────────────────────────────────────────
    distances = heat_method(G, source_node, t_factor=T_FACTOR)

    # ── Stats ──────────────────────────────────────────────────────────
    vals = np.array(list(distances.values()))
    print(f"\n    Distance stats (risk={RISK_LEVEL}):")
    print(f"      Min  : {vals.min():.2f}")
    print(f"      Mean : {vals.mean():.2f}")
    print(f"      Max  : {vals.max():.2f}")

    # ── Visualise ──────────────────────────────────────────────────────
    visualise(G, distances, source_node, RISK_LEVEL)


if __name__ == "__main__":
    main()
