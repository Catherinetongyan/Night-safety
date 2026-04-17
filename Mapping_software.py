import pickle
import time
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import yaml
from pyproj import Transformer
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler

# Load config
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

base_dir = Path(".")

print("Downloading graph...")
G = ox.graph_from_place("Bristol, United Kingdom", network_type="walk")

cctv_df   = pd.read_csv(r"Council_cctv_cameras_longlat.csv")
lights_df = pd.read_csv(r"Council_streetlights_longlat.csv")

cctv_gdf   = gpd.GeoDataFrame(cctv_df,   geometry=gpd.points_from_xy(cctv_df.longitude,   cctv_df.latitude),   crs="EPSG:4326").to_crs(epsg=3857)
lights_gdf = gpd.GeoDataFrame(lights_df, geometry=gpd.points_from_xy(lights_df.longitude, lights_df.latitude), crs="EPSG:4326").to_crs(epsg=3857)

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def safety_score(point):
    buffer = point.buffer(40)
    cctv_count  = cctv_gdf[cctv_gdf.intersects(buffer)].shape[0]
    light_count = lights_gdf[lights_gdf.intersects(buffer)].shape[0]
    return (cctv_count * 2) + (light_count * 1)

print("Scoring edges... (this takes a while)")
total = G.number_of_edges()
for i, (u, v, k, data) in enumerate(G.edges(keys=True, data=True)):
    if i % 1000 == 0:
        print(f"  {i}/{total} edges done...")
    x1, y1 = transformer.transform(G.nodes[u]['x'], G.nodes[u]['y'])
    x2, y2 = transformer.transform(G.nodes[v]['x'], G.nodes[v]['y'])
    midpoint = Point((x1 + x2) / 2, (y1 + y2) / 2)
    safety = safety_score(midpoint)
    data["safety_density"] = safety / data.get("length", 1.0)

# ── Crime KDE with time-decay ─────────────────────────────────────────────────
print("\nComputing crime scores...")
kde_cfg  = cfg["kde"]
severity = cfg["severity"]

df = pd.read_csv(base_dir / cfg["paths"]["crime_csv"])
df["severity"] = df["crime_type"].map(severity).fillna(0.5)

min_sev = kde_cfg.get("min_severity", 0.0)
if min_sev > 0:
    before = len(df)
    df = df[df["severity"] >= min_sev]
    print(f"  Severity filter: {before:,} -> {len(df):,} records")

# Time-decay: more recent crimes weighted more heavily
half_life_days = kde_cfg.get("crime_half_life_days", 180)
lam = np.log(2) / half_life_days
df["Date"]  = pd.to_datetime(df["Date"], format="%d-%b-%y")
days_ago    = (pd.Timestamp.now() - df["Date"]).dt.total_seconds() / 86400.0
df["decay"] = np.exp(-lam * days_ago)
df["weight"] = df["severity"] * df["decay"]

LAT0, LON0 = 51.45, -2.60
coords_m = np.stack([
    (df["lat"].values - LAT0) * 111320,
    (df["lon"].values - LON0) * 111320 * np.cos(np.radians(LAT0)),
], axis=1).astype(np.float32)
weights = df["weight"].values.astype(np.float32)
weights = weights / weights.sum()

bw_metres = kde_cfg.get("bandwidth_metres", 150)

node_ids      = list(G.nodes)
node_coords_m = np.stack([
    (np.array([G.nodes[n]["y"] for n in node_ids]) - LAT0) * 111320,
    (np.array([G.nodes[n]["x"] for n in node_ids]) - LON0) * 111320 * np.cos(np.radians(LAT0)),
], axis=1).astype(np.float32)

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
          f"batch={BATCH}, free VRAM={free_mem/1e9:.1f}GB, bw={bw_metres}m")

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
    print(f"  GPU KDE complete: {time.time() - t0:.1f}s")

except (ImportError, Exception) as e:
    if not isinstance(e, ImportError):
        print(f"  CuPy runtime error ({type(e).__name__}) — falling back to scipy CPU mode...")
    else:
        print("  CuPy not found — falling back to scipy CPU mode...")
    from scipy.stats import gaussian_kde

    t0  = time.time()
    kde = gaussian_kde(coords_m.T, bw_method=bw_metres / coords_m.std(), weights=weights)
    print(f"  CPU KDE fit complete: {time.time() - t0:.1f}s")

    print("  Scoring nodes on CPU...")
    t0      = time.time()
    density = kde(node_coords_m.T)
    print(f"  CPU node scoring complete: {time.time() - t0:.1f}s")

crime_scores_arr = MinMaxScaler().fit_transform(density.reshape(-1, 1)).flatten()
node_crime       = dict(zip(node_ids, crime_scores_arr))
nx.set_node_attributes(G, node_crime, "crime_score")

# ── Save graph with individual weights ────────────────────────────────────────
weighted_path = base_dir / cfg["paths"]["weighted_graph"]
print(f"\nSaving graph to: {weighted_path}")
with open(weighted_path, "wb") as f:
    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

cctv_gdf.to_crs(epsg=4326).to_file(r"cctv.gpkg",   driver="GPKG")
lights_gdf.to_crs(epsg=4326).to_file(r"lights.gpkg", driver="GPKG")

print("Done! Run safe_route.py whenever you want a route.")
