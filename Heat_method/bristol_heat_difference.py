"""
Bristol Heat Diffusion — Natural vs Safety-Weighted Comparison
==============================================================
Runs the Heat Method twice from the same origin node and plots the
resulting geodesic φ fields side by side:

  LEFT  — Unweighted (edge weight = length only)
           Heat diffuses purely by geography → concentric rings
  RIGHT — Safety-weighted (length + crime penalty + safety bonus)
           φ field is distorted: repelled from high-crime areas,
           attracted toward lit/CCTV-covered streets

Usage
-----
    python bristol_heat_comparison.py

Edit ORIGIN_LAT / ORIGIN_LON at the top of main() to change the source.

Dependencies
------------
    pip install osmnx networkx numpy scipy matplotlib
"""

from __future__ import annotations

import os
import pickle
import time
import json
import hashlib
from dataclasses import dataclass

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.sparse import csr_matrix, diags, lil_matrix
from scipy.sparse.linalg import spsolve

# ── Reuse paths from your main script ────────────────────────────────────────
GRAPH_PKL = "../Graph/bristol_heat_graph.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# USER PROFILE  (mirrors bristol_heat_routing2.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    w_distance : float = 0.4
    w_crime    : float = 0.3
    w_safety   : float = 0.3
    require_lit: bool  = False

    def profile_key(self) -> str:
        d = {k: v for k, v in self.__dict__.items()}
        return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def cost_safety(edge_data: dict, profile: UserProfile) -> float:
    """Full safety-aware cost (your existing cost_function)."""
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


def cost_natural(edge_data: dict, _profile=None) -> float:
    """Unweighted — pure geographic distance."""
    return max(float(edge_data.get("length", 50.0)), 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# LAPLACIAN  (shared helper)
# ─────────────────────────────────────────────────────────────────────────────

def build_laplacian(G, cost_fn, profile=None):
    nodes    = list(G.nodes())
    n        = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}
    rows, cols, vals = [], [], []
    for u, v, data in G.edges(data=True):
        c = cost_fn(data, profile) if profile is not None else cost_fn(data)
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

def run_heat_method(G, origin_node, cost_fn, profile=None, t_factor=1.0,
                    label="") -> dict:
    print(f"  [{label}] Building Laplacian …")
    L, nodes, node_idx = build_laplacian(G, cost_fn, profile)
    n   = len(nodes)
    src = node_idx[origin_node]

    # Representative edge costs for t selection
    costs = []
    for _, _, data in G.edges(data=True):
        c = cost_fn(data, profile) if profile is not None else cost_fn(data)
        if c < float("inf"):
            costs.append(c)
    mean_c = float(np.mean(costs)) if costs else 1.0
    t      = t_factor * (mean_c ** 2)

    # Step 1 — heat diffusion
    delta = np.zeros(n);  delta[src] = 1.0
    I     = diags(np.ones(n), format="csr")
    t0    = time.time()
    u     = spsolve(I + t * L, delta)
    print(f"    Heat solve:    {time.time() - t0:.2f}s")

    # Step 2 — normalised gradient divergence
    div = np.zeros(n)
    for eu, ev, data in G.edges(data=True):
        c = cost_fn(data, profile) if profile is not None else cost_fn(data)
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

    # Step 3 — Poisson solve for φ
    L_mod          = lil_matrix(L)
    div_mod        = div.copy()
    L_mod[src, :]  = 0
    L_mod[src, src] = 1
    div_mod[src]   = 0
    t0  = time.time()
    phi = spsolve(L_mod.tocsr(), div_mod)
    phi -= phi[src]
    phi  = np.abs(phi)
    print(f"    Poisson solve: {time.time() - t0:.2f}s")

    return {nodes[i]: phi[i] for i in range(n)}


# ─────────────────────────────────────────────────────────────────────────────
# SIDE-BY-SIDE PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(G, origin_node,
                    phi_natural: dict,
                    phi_safety:  dict):
    """
    Three-panel dark figure:
      Left   — Natural φ field (distance only)
      Centre — Safety-weighted φ field
      Right  — Difference map (safety_norm − natural_norm):
                 blue  = safety pulls this area closer (well-lit / low-crime)
                 red   = safety pushes this area further (high-crime / unlit)
    """

    node_list = list(G.nodes())
    xs = np.array([G.nodes[n]["x"] for n in node_list])
    ys = np.array([G.nodes[n]["y"] for n in node_list])

    vals_nat = np.array([phi_natural.get(n, 0.0) for n in node_list])
    vals_saf = np.array([phi_safety.get(n,  0.0) for n in node_list])

    # ── Per-panel normalisation [0, 1] using each field's 98th percentile ────
    p98_nat = float(np.percentile(vals_nat[vals_nat > 0], 98))
    p98_saf = float(np.percentile(vals_saf[vals_saf > 0], 98))
    print(f"  φ 98th-pct  natural={p98_nat:.2f}  safety={p98_saf:.2f}")

    nat_norm = np.clip(vals_nat / p98_nat, 0, 1)
    saf_norm = np.clip(vals_saf / p98_saf, 0, 1)

    # Difference: positive → safety makes the node feel further away (red)
    #             negative → safety makes the node feel closer       (blue)
    diff = saf_norm - nat_norm
    abs_max = float(np.percentile(np.abs(diff), 99))   # symmetric clip
    print(f"  Δφ range: {diff.min():.3f} – {diff.max():.3f}  "
          f"(±{abs_max:.3f} at 99th pct)")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(28, 9))
    fig.patch.set_facecolor("black")
    fig.subplots_adjust(left=0.02, right=0.93, bottom=0.10, top=0.78, wspace=0.06)

    # ── Left + Centre: plasma φ panels ───────────────────────────────────────
    plasma_panels = [
        (axes[0], vals_nat, p98_nat,
         "Natural graph  (distance only)",
         "φ grows as concentric rings\nno influence from crime or lighting",
         f"colour scale: 0 – {p98_nat:.0f}  (relative φ)"),
        (axes[1], vals_saf, p98_saf,
         "Safety-weighted graph  (distance + crime + lighting)",
         "φ distorted by safety costs: repelled from high-crime areas,\n"
         "drawn toward lit / CCTV-covered streets",
         f"colour scale: 0 – {p98_saf:.0f}  (relative φ)"),
    ]

    for ax, vals, vmax, title, subtitle, scale_note in plasma_panels:
        ax.set_facecolor("black")
        ax.scatter(xs, ys, c=vals, cmap="plasma",
                   vmin=0, vmax=vmax, s=4, linewidths=0)
        ax.scatter(G.nodes[origin_node]["x"], G.nodes[origin_node]["y"],
                   c="red", s=160, zorder=6)
        ax.set_title(title, color="white", fontsize=11,
                     fontweight="bold", pad=6)
        ax.text(0.5, -0.02, subtitle, transform=ax.transAxes,
                ha="center", va="top", color="#cccccc",
                fontsize=9, linespacing=1.4)
        ax.text(0.5, -0.10, scale_note, transform=ax.transAxes,
                ha="center", va="top", color="#888888",
                fontsize=8, style="italic")
        ax.set_axis_off()

        # Individual colourbar positioned just right of each panel
        pos = ax.get_position()
        cbar_ax = fig.add_axes([pos.x1 + 0.005, pos.y0 + 0.05,
                                 0.012, pos.height - 0.10])
        sm = plt.cm.ScalarMappable(cmap="plasma",
                                   norm=mcolors.Normalize(vmin=0, vmax=vmax))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.set_label("φ  (relative)", color="white", fontsize=8, labelpad=8)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

    # ── Right: diverging difference map ───────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor("black")

    sc3 = ax3.scatter(xs, ys, c=diff, cmap="RdBu_r",
                      vmin=-abs_max, vmax=abs_max,
                      s=4, linewidths=0)
    ax3.scatter(G.nodes[origin_node]["x"], G.nodes[origin_node]["y"],
                c="lime", s=160, zorder=6)

    ax3.set_title("Difference  (safety − natural,  normalised)",
                  color="white", fontsize=11, fontweight="bold", pad=6)
    ax3.text(0.5, -0.02,
             "Red  = safety routing pushes this area further (high crime / unlit)\n"
             "Blue = safety routing pulls this area closer  (low crime / well-lit)",
             transform=ax3.transAxes, ha="center", va="top",
             color="#cccccc", fontsize=9, linespacing=1.4)
    ax3.text(0.5, -0.10,
             f"symmetric scale: ±{abs_max:.2f}  (Δ normalised φ)",
             transform=ax3.transAxes, ha="center", va="top",
             color="#888888", fontsize=8, style="italic")
    ax3.set_axis_off()

    # Colourbar for difference panel
    pos3 = ax3.get_position()
    cbar_ax3 = fig.add_axes([pos3.x1 + 0.005, pos3.y0 + 0.05,
                              0.012, pos3.height - 0.10])
    cb3 = fig.colorbar(sc3, cax=cbar_ax3)
    cb3.set_label("Δφ  (safety − natural)", color="white",
                  fontsize=8, labelpad=8)
    cb3.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb3.ax.yaxis.get_ticklabels(), color="white", fontsize=7)
    # Label the poles of the diverging bar explicitly
    cb3.ax.text(2.2, 1.02, "further\n(red)", transform=cb3.ax.transAxes,
                color="#ff6666", fontsize=7, ha="left", va="bottom")
    cb3.ax.text(2.2, -0.02, "closer\n(blue)", transform=cb3.ax.transAxes,
                color="#6699ff", fontsize=7, ha="left", va="top")

    # ── Supra-title ───────────────────────────────────────────────────────────
    fig.text(0.47, 0.93,
             "Heat Method — Natural vs Safety-Weighted Geodesic Field  (Bristol)",
             color="white", fontsize=14, fontweight="bold",
             ha="center", va="bottom")
    fig.text(0.47, 0.89,
             "Left & centre: independent colour scales to reveal spatial shape  │  "
             "Right: difference on normalised [0–1] fields",
             color="#aaaaaa", fontsize=9, ha="center", va="bottom",
             style="italic")

    # ── Legend ────────────────────────────────────────────────────────────────
    fig.legend(
        handles=[
            plt.scatter([], [], c="red",  s=60, label="Origin (φ panels)"),
            plt.scatter([], [], c="lime", s=60, label="Origin (difference panel)"),
        ],
        loc="lower center", ncol=2,
        facecolor="#1a1a2e", labelcolor="white",
        fontsize=9, framealpha=0.7,
        bbox_to_anchor=(0.47, 0.01),
    )

    plt.savefig("heat_comparison.png", dpi=150, bbox_inches="tight",
                facecolor="black")
    print("  Saved → heat_comparison.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Edit origin here ──────────────────────────────────────────────────────
    ORIGIN_LAT, ORIGIN_LON = 51.4491, -2.5814   # Temple Meads
    # ─────────────────────────────────────────────────────────────────────────

    import osmnx as ox

    print("=" * 55)
    print("  Bristol Heat Diffusion Comparison")
    print("=" * 55)

    # Load the pre-built graph (must already exist)
    if not os.path.exists(GRAPH_PKL):
        raise FileNotFoundError(
            f"{GRAPH_PKL} not found — run bristol_heat_routing2.py once first "
            "to download and score the graph."
        )
    print(f"\nLoading graph from {GRAPH_PKL} …")
    with open(GRAPH_PKL, "rb") as f:
        G = pickle.load(f)
    print(f"  {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges")

    origin_node = ox.nearest_nodes(G, ORIGIN_LON, ORIGIN_LAT)
    print(f"  Origin node: {origin_node}")

    profile = UserProfile(w_distance=0.4, w_crime=0.3, w_safety=0.3)

    print("\n[1/2] Heat method — natural graph …")
    t0 = time.time()
    phi_natural = run_heat_method(
        G, origin_node,
        cost_fn=cost_natural,
        profile=None,
        label="natural",
    )
    print(f"  Done: {time.time() - t0:.1f}s")

    print("\n[2/2] Heat method — safety-weighted graph …")
    t0 = time.time()
    phi_safety = run_heat_method(
        G, origin_node,
        cost_fn=cost_safety,
        profile=profile,
        label="safety",
    )
    print(f"  Done: {time.time() - t0:.1f}s")

    print("\nPlotting …")
    plot_comparison(G, origin_node, phi_natural, phi_safety)


if __name__ == "__main__":
    main()
