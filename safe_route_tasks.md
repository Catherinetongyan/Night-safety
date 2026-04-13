# Safe Route — Implementation Tasks for Claude Code

> File to modify: `safe_route.py` + `config.yaml`
> All three tasks must be applied in the same editing session.

---

## Task 1 — GPU-accelerated KDE + save weighted graph at end of `main()`

### 1a. Rewrite `build_crime_scores` to use GPU (CuPy)

Replace the entire `build_crime_scores` function with the version below.

Key changes vs. the current implementation:
- **Project crime coords to metres** (flat local projection centred on Bristol). CuPy does not support the `haversine` metric, but Bristol's area is small enough that a flat projection introduces negligible error.
- **Use CuPy** to run a Gaussian KDE entirely on the GPU via matrix operations. cuML is not used — it does not support Windows. CuPy installs natively on Windows: `pip install cupy-cuda12x`.
- If CuPy is not installed, automatically fall back to `scipy.stats.gaussian_kde` on CPU — the function signature and return value stay identical either way.
- **`bandwidth_metres`** replaces the old `bandwidth` key in config (was in radians, now in metres — value ≈ 150).
- Remove the `np.repeat` row-explosion trick; scipy branch uses native `weights=` argument instead.
- **Auto batch size**: VRAM is queried at runtime and the batch size is calculated to use 75 % of free memory, so it adapts to whatever is available without manual tuning.
- **Optimised distance computation**: uses `‖a−b‖² = ‖a‖² + ‖b‖² − 2aᵀb` to avoid materialising the large `(BATCH, N, 2)` diff tensor, reducing peak VRAM by ~25 %.

```python
def build_crime_scores(G: nx.MultiDiGraph, cfg: dict, base_dir: Path) -> dict:
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
    weights = weights / weights.sum()   # normalise once here, used in both branches

    bw_metres = kde_cfg.get("bandwidth_metres", 150)

    # Node coords projected to the same flat space
    node_ids      = list(G.nodes)
    node_coords_m = np.stack([
        (np.array([G.nodes[n]["y"] for n in node_ids]) - LAT0) * 111320,
        (np.array([G.nodes[n]["x"] for n in node_ids]) - LON0) * 111320 * np.cos(np.radians(LAT0)),
    ], axis=1).astype(np.float32)

    try:
        import cupy as cp

        crime_gpu = cp.asarray(coords_m)       # (N_crime, 2)
        nodes_gpu = cp.asarray(node_coords_m)  # (N_nodes, 2)
        w_gpu     = cp.asarray(weights)        # (N_crime,)

        # Auto batch size: use 75 % of free VRAM.
        # Peak usage per batch = 3 × BATCH × N_crime × 4 bytes
        # (node_sq reuse means we never materialise the (BATCH, N, 2) diff tensor)
        free_mem = cp.cuda.Device(0).mem_info[0]
        n_crime  = len(crime_gpu)
        BATCH    = int(free_mem * 0.75 / (3 * n_crime * 4))
        BATCH    = max(256, min(BATCH, len(nodes_gpu)))
        print(f"  GPU KDE: {n_crime:,} crime pts, {len(nodes_gpu):,} nodes, "
              f"batch={BATCH}, free VRAM={free_mem/1e9:.1f}GB, bw={bw_metres}m")

        t0             = time.time()
        bw2            = cp.float32(bw_metres ** 2)
        crime_sq       = cp.sum(crime_gpu ** 2, axis=1)   # (N_crime,)  computed once
        density_chunks = []

        for i in range(0, len(nodes_gpu), BATCH):
            batch    = nodes_gpu[i : i + BATCH]            # (B, 2)
            node_sq  = cp.sum(batch ** 2, axis=1, keepdims=True)  # (B, 1)
            cross    = batch @ crime_gpu.T                         # (B, N_crime)
            sq_dist  = node_sq + crime_sq[None, :] - 2 * cross    # (B, N_crime)
            gauss    = cp.exp(-0.5 * sq_dist / bw2)               # (B, N_crime)
            density_chunks.append(cp.asnumpy(gauss @ w_gpu))      # (B,) → CPU

        density = np.concatenate(density_chunks)
        print(f"  GPU KDE complete: {time.time() - t0:.1f}s")

    except ImportError:
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
```

### 1b. Save the weighted graph at the end of `main()`

Insert the block below **after** `update_safety_weights(...)` is called and **before** `find_routes(...)`:

```python
    # Save the fully weighted graph (length + crime + light + cctv)
    save_path = resolve_path(
        base_dir,
        cfg["paths"].get("weighted_graph", "bristol_weighted_graph.pkl")
    )
    print(f"\nSaving weighted graph to: {save_path}")
    with open(save_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("  Saved successfully.")
```

### 1c. Update `config.yaml`

Add the `weighted_graph` path and replace `bandwidth` (radians) with `bandwidth_metres`:

```yaml
paths:
  graph:          bristol_safety_graph.pkl        # original read-only graph
  crime_csv:      bristol_2023-01_to_2025-12.csv
  cctv:           cctv.gpkg
  lights:         lights.gpkg
  weighted_graph: bristol_weighted_graph.pkl      # ADD THIS LINE

kde:
  bandwidth_metres: 150     # replaces old 'bandwidth' key (was in radians)
  kernel: gaussian
  min_severity: 0.0
```

**Remove** the old `bandwidth: 0.0015` key — it is no longer used.

---

## Task 2 — Three weight sets → three side-by-side route plots

### 2a. Rewrite `update_safety_weights`

Add a second edge field `crime_weight` (distance + crime only) alongside the existing `safety_weight` (all four factors):

```python
def update_safety_weights(G: nx.MultiDiGraph, node_crime: dict, cfg: dict):
    alpha = cfg["weights"]["alpha"]
    beta  = cfg["weights"]["beta"]
    gamma = cfg["weights"]["gamma"]

    for u, v, data in G.edges(data=True):
        edge_crime  = (node_crime.get(u, 0) + node_crime.get(v, 0)) / 2.0
        light_score = data.get("light_score", 0.0)
        cctv_score  = data.get("cctv_score",  0.0)
        length      = data.get("length", 1.0)

        # Route A weight: distance only (already "length", no new field needed)

        # Route B weight: distance + crime
        data["crime_weight"] = (
            length
            * (1 + alpha * edge_crime)
        )

        # Route C weight: distance + crime + street lights + CCTV
        data["safety_weight"] = (
            length
            * (1 + alpha * edge_crime)
            * max(0.1, 1 - beta  * light_score)
            * max(0.1, 1 - gamma * cctv_score)
        )
```

### 2b. Rewrite `find_routes`

Return three routes instead of two:

```python
def find_routes(G: nx.MultiDiGraph, cfg: dict):
    route_cfg = cfg["route"]
    start_lat, start_lon = route_cfg["start"]["lat"], route_cfg["start"]["lon"]
    goal_lat,  goal_lon  = route_cfg["end"]["lat"],   route_cfg["end"]["lon"]

    start_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
    goal_node  = ox.distance.nearest_nodes(G, goal_lon,  goal_lat)

    quickest     = nx.shortest_path(G, start_node, goal_node, weight="length")
    crime_route  = nx.astar_path(G,   start_node, goal_node, weight="crime_weight")
    safest       = nx.astar_path(G,   start_node, goal_node, weight="safety_weight")

    return start_node, goal_node, quickest, crime_route, safest
```

### 2c. Rewrite `plot_routes`

Change to three columns. New colour for route B: `#ff9f43` (amber).

```python
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
```

### 2d. Update the `main()` call site

Replace the two-value unpack and `plot_routes` call:

```python
    # Before (two routes):
    start_node, goal_node, quickest, safest = find_routes(G, cfg)
    plot_routes(G, start_node, goal_node, quickest, safest, cctv_gdf, lights_gdf, cfg)

    # After (three routes):
    start_node, goal_node, quickest, crime_route, safest = find_routes(G, cfg)
    plot_routes(G, start_node, goal_node, quickest, crime_route, safest, cctv_gdf, lights_gdf, cfg)
```

---

## Task 3 — Convert all Chinese comments to English

Replace every Chinese-language comment and print string in `safe_route.py` with English.

Reference table of strings that need changing:

| Location | Chinese | English replacement |
|---|---|---|
| Module docstring | `在 config.yaml 中调整路径...` | `Adjust paths, waypoints and parameters in config.yaml.` |
| Module docstring | `运行方式：` | `Usage:` |
| `load_config` | `支持绝对路径和相对于 config 文件的相对路径` | `Supports absolute paths and paths relative to the config file.` |
| `build_crime_scores` docstring | `读取犯罪 CSV，做加权 KDE...` | `Read crime CSV, run weighted KDE, return {node_id: crime_score [0,1]}.` |
| `update_safety_weights` docstring | `根据公式重新计算每条边的 safety_weight，直接修改图` | `Recalculate safety_weight for every edge in-place.` |
| `find_routes` docstring | `返回 (start_node, goal_node, ...)` | `Return (start_node, goal_node, quickest, crime_route, safest).` |
| `route_stats` docstring | `返回 (距离m, 步行分钟数)` | `Return (distance_m, walking_minutes).` |
| `main` argparse help | `YAML 配置文件路径（默认：config.yaml）` | `Path to YAML config file (default: config.yaml).` |
| `main` print | `加载路网图：` | `Loading road network graph:` |
| `main` print | `计算犯罪密度...` | `Computing crime density...` |
| `main` print | `更新路段安全权重...` | `Updating edge safety weights...` |
| `main` print | `计算路线...` | `Finding routes...` |
| Section divider comments | `# 1. 配置加载`, `# 2. 犯罪 KDE 计算`, etc. | `# 1. Config`, `# 2. Crime KDE`, `# 3. Edge weights`, `# 4. Routing`, `# 5. Visualisation`, `# 6. Entry point` |

---

## Dependency note

**CuPy** is used for GPU acceleration. It supports Windows natively and does not require WSL2. Install the build that matches your CUDA version (12.7 from `nvidia-smi`):

```bash
pip install cupy-cuda12x
```

If CuPy is not installed the code falls back to `scipy.stats.gaussian_kde` on CPU automatically — no code change needed.

> **Why not cuML?** RAPIDS cuML does not support Windows. CuPy achieves the same GPU speedup via direct matrix operations and works on Windows out of the box.
