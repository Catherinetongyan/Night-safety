"""
Bristol Safe Walking Router — Graph Builder
=============================================
Downloads the OSM walking network, scores safety infrastructure
and crime density on every edge, then saves a fully-weighted graph.

Usage:
    python Mapping_software.py
    python Mapping_software.py --config my_config.yaml
"""

import argparse
import pickle
import time
from collections import Counter
from pathlib import Path

import geopandas as gpd
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
    p = Path(raw_path)
    return p if p.is_absolute() else base_dir / p


# ────────────────────────────────────────────────────────────
# 2. Road network
# ────────────────────────────────────────────────────────────

def download_graph(place: str = "Bristol, United Kingdom") -> nx.MultiDiGraph:
    """Download the OSM walking network."""
    print(f"Downloading walking network for '{place}'...")
    t0 = time.time()
    G = ox.graph_from_place(place, network_type="walk")
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges  "
          f"({time.time() - t0:.1f}s)")
    return G


# ────────────────────────────────────────────────────────────
# 3. Safety infrastructure
# ────────────────────────────────────────────────────────────

def load_infrastructure(cctv_csv: str, lights_csv: str):
    """Read CCTV and streetlight CSVs, return GeoDataFrames in EPSG:3857."""
    print("Loading safety infrastructure data...")
    t0 = time.time()

    cctv_df   = pd.read_csv(cctv_csv)
    lights_df = pd.read_csv(lights_csv)

    cctv_gdf = gpd.GeoDataFrame(
        cctv_df,
        geometry=gpd.points_from_xy(cctv_df.longitude, cctv_df.latitude),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    lights_gdf = gpd.GeoDataFrame(
        lights_df,
        geometry=gpd.points_from_xy(lights_df.longitude, lights_df.latitude),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    print(f"  {len(cctv_gdf):,} CCTV cameras, {len(lights_gdf):,} streetlights  "
          f"({time.time() - t0:.1f}s)")
    return cctv_gdf, lights_gdf


def score_safety(G: nx.MultiDiGraph, cctv_gdf, lights_gdf,cfg: dict):
    """Snap each CCTV/light to its nearest edge, count per edge,
    and write safety_density = (cctv*2 + lights*1) / length."""
    w_cctv  = cfg["weights"]["cctv_weight"]
    w_light = cfg["weights"]["light_weight"]
    print("Scoring safety infrastructure on edges...")
    t0 = time.time()

    _nodes, edges_gdf = ox.graph_to_gdfs(G)
    edges_gdf = edges_gdf.to_crs(epsg=3857)

    print("  Snapping CCTV to nearest edges...")
    cctv_snapped = edges_gdf.sindex.nearest(
        cctv_gdf.geometry, return_all=False
    )[1]

    print("  Snapping streetlights to nearest edges...")
    lights_snapped = edges_gdf.sindex.nearest(
        lights_gdf.geometry, return_all=False
    )[1]

    cctv_counts  = Counter(edges_gdf.index[i] for i in cctv_snapped)
    light_counts = Counter(edges_gdf.index[i] for i in lights_snapped)

    for u, v, k, data in G.edges(keys=True, data=True):
        cc = cctv_counts.get((u, v, k), 0)
        lc = light_counts.get((u, v, k), 0)
        score = cc * w_cctv + lc * w_light
        data["cctv_count"]     = cc
        data["light_count"]    = lc
        data["safety_density"] = score / data.get("length", 1.0)

    elapsed = time.time() - t0
    print(f"  {sum(cctv_counts.values()):,} CCTV, "
          f"{sum(light_counts.values()):,} lights assigned  ({elapsed:.1f}s)")


# ────────────────────────────────────────────────────────────
# 4. Crime KDE with time-decay
# ────────────────────────────────────────────────────────────

def score_crime(G: nx.MultiDiGraph, cfg: dict, base_dir: Path):
    """Read crime CSV, apply severity × time-decay weighting,
    run KDE, and store crime_score as a node attribute."""
    print("Computing crime density...")
    t0_total = time.time()

    kde_cfg  = cfg["kde"]
    severity = cfg["severity"]

    # ── Load & filter ────────────────────────────────────────
    print("  Loading crime data...")
    df = pd.read_csv(resolve_path(base_dir, cfg["paths"]["crime_csv"]))
    print(f"  {len(df):,} records loaded")

    df["severity"] = df["crime_type"].map(severity).fillna(0.5)

    min_sev = kde_cfg.get("min_severity", 0.0)
    if min_sev > 0:
        before = len(df)
        df = df[df["severity"] >= min_sev]
        print(f"  Severity filter: {before:,} → {len(df):,} records")

    # ── Time-decay ───────────────────────────────────────────
    print("  Applying time-decay weighting...")
    half_life_days = kde_cfg.get("crime_half_life_days", 180)
    lam = np.log(2) / half_life_days

    df["Date"]   = pd.to_datetime(df["Date"], format="%d-%b-%y")
    days_ago     = (pd.Timestamp.now() - df["Date"]).dt.total_seconds() / 86400.0
    df["decay"]  = np.exp(-lam * days_ago)
    df["weight"] = df["severity"] * df["decay"]
    print(f"  half_life = {half_life_days} days  |  "
          f"date range: {df['Date'].min().date()} → {df['Date'].max().date()}")

    # ── Project to flat metres ───────────────────────────────
    LAT0, LON0 = 51.45, -2.60
    coords_m = np.stack([
        (df["lat"].values - LAT0) * 111320,
        (df["lon"].values - LON0) * 111320 * np.cos(np.radians(LAT0)),
    ], axis=1).astype(np.float32)

    weights = df["weight"].values.astype(np.float32)
    weights = weights / weights.sum()

    bw_metres = kde_cfg.get("bandwidth_metres", 150)

    node_ids = list(G.nodes)
    node_coords_m = np.stack([
        (np.array([G.nodes[n]["y"] for n in node_ids]) - LAT0) * 111320,
        (np.array([G.nodes[n]["x"] for n in node_ids]) - LON0)
        * 111320 * np.cos(np.radians(LAT0)),
    ], axis=1).astype(np.float32)

    # ── KDE (GPU → CPU fallback) ─────────────────────────────
    print(f"  Running KDE (bandwidth={bw_metres}m)...")
    density = _kde_gpu_or_cpu(coords_m, node_coords_m, weights, bw_metres)

    # ── Normalise & store ────────────────────────────────────
    crime_scores = MinMaxScaler().fit_transform(
        density.reshape(-1, 1)
    ).flatten()
    node_crime = dict(zip(node_ids, crime_scores))
    nx.set_node_attributes(G, node_crime, "crime_score")

    elapsed = time.time() - t0_total
    print(f"  Crime scoring complete  ({elapsed:.1f}s)")


def _kde_gpu_or_cpu(coords_m, node_coords_m, weights, bw_metres):
    """Try GPU (CuPy) KDE first; fall back to scipy CPU."""
    try:
        import cupy as cp

        _ = cp.array([1.0], dtype=cp.float32) ** 2
        crime_gpu = cp.asarray(coords_m)
        nodes_gpu = cp.asarray(node_coords_m)
        w_gpu     = cp.asarray(weights)

        free_mem = cp.cuda.Device(0).mem_info[0]
        n_crime  = len(crime_gpu)
        BATCH    = int(free_mem * 0.75 / (3 * n_crime * 4))
        BATCH    = max(256, min(BATCH, len(nodes_gpu)))
        print(f"  GPU KDE: {n_crime:,} crime pts, {len(nodes_gpu):,} nodes, "
              f"batch={BATCH}, free VRAM={free_mem / 1e9:.1f}GB")

        t0       = time.time()
        bw2      = cp.float32(bw_metres ** 2)
        crime_sq = cp.sum(crime_gpu ** 2, axis=1)
        chunks   = []

        for i in range(0, len(nodes_gpu), BATCH):
            batch   = nodes_gpu[i : i + BATCH]
            node_sq = cp.sum(batch ** 2, axis=1, keepdims=True)
            cross   = batch @ crime_gpu.T
            sq_dist = node_sq + crime_sq[None, :] - 2 * cross
            gauss   = cp.exp(-0.5 * sq_dist / bw2)
            chunks.append(cp.asnumpy(gauss @ w_gpu))

        density = np.concatenate(chunks)
        print(f"  GPU KDE complete  ({time.time() - t0:.1f}s)")
        return density

    except (ImportError, Exception) as e:
        if isinstance(e, ImportError):
            print("  CuPy not found — falling back to scipy CPU mode...")
        else:
            print(f"  CuPy runtime error ({type(e).__name__}) "
                  f"— falling back to scipy CPU mode...")

        from scipy.stats import gaussian_kde

        t0 = time.time()
        kde = gaussian_kde(
            coords_m.T,
            bw_method=bw_metres / coords_m.std(),
            weights=weights,
        )
        print(f"  CPU KDE fit complete  ({time.time() - t0:.1f}s)")

        t0      = time.time()
        density = kde(node_coords_m.T)
        print(f"  CPU node scoring complete  ({time.time() - t0:.1f}s)")
        return density


# ────────────────────────────────────────────────────────────
# 5. Save outputs
# ────────────────────────────────────────────────────────────

def save_outputs(G, cctv_gdf, lights_gdf, cfg, base_dir):
    """Save the weighted graph pickle and infrastructure GeoPackages."""
    weighted_path = resolve_path(
        base_dir,
        cfg["paths"].get("weighted_graph", "bristol_weighted_graph.pkl"),
    )
    print(f"Saving weighted graph to {weighted_path}...")
    t0 = time.time()
    with open(weighted_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Graph saved  ({time.time() - t0:.1f}s)")

    print("Saving infrastructure GeoPackages...")
    cctv_gdf.to_crs(epsg=4326).to_file("cctv.gpkg",   driver="GPKG")
    lights_gdf.to_crs(epsg=4326).to_file("lights.gpkg", driver="GPKG")
    print("  cctv.gpkg, lights.gpkg saved")


# ────────────────────────────────────────────────────────────
# 6. Entry point
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bristol Safe Walking Router — Graph Builder"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    base_dir    = config_path.parent
    cfg         = load_config(config_path)

    t_start = time.time()

    G = download_graph()

    cctv_gdf, lights_gdf = load_infrastructure(
        "Council_cctv_cameras_longlat.csv",
        "Council_streetlights_longlat.csv",
    )

    score_safety(G, cctv_gdf, lights_gdf,cfg)
    score_crime(G, cfg, base_dir)
    save_outputs(G, cctv_gdf, lights_gdf, cfg, base_dir)

    elapsed = time.time() - t_start
    print(f"\nAll done!  Total time: {elapsed:.1f}s")
    print("Run safe_route.py whenever you want a route.")


if __name__ == "__main__":
    main()