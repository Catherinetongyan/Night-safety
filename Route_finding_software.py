import osmnx as ox
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle

# -------------------
# Load pre-built graph
# -------------------
with open(r"C:\Users\Oliver\Documents\bristol_safety_graph.pkl", "rb") as f:
    G = pickle.load(f)

cctv_gdf = gpd.read_file(r"C:\Users\Oliver\Documents\cctv.gpkg")
lights_gdf = gpd.read_file(r"C:\Users\Oliver\Documents\lights.gpkg")

# -------------------
# Set your start and end points here
# -------------------
start_lat, start_lon = 51.4576, -2.6053   # Wills Memorial
goal_lat,  goal_lon  = 51.4493, -2.5980   # Thekla

WALKING_SPEED_KMH = 5.0  # average walking speed

# -------------------
# Find nodes
# -------------------
start_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
goal_node  = ox.distance.nearest_nodes(G, goal_lon,  goal_lat)

# -------------------
# Quickest route (shortest distance)
# -------------------
quickest_route = nx.shortest_path(G, start_node, goal_node, weight="length")

# -------------------
# Safest route
# -------------------
safest_route = nx.astar_path(G, start_node, goal_node, weight="safety_weight")

# -------------------
# Calculate distances and walking times
# -------------------
def route_stats(G, route, weight="length"):
    total_m = sum(
        G[u][v][0].get("length", 0)
        for u, v in zip(route[:-1], route[1:])
    )
    total_km = total_m / 1000
    minutes = (total_km / WALKING_SPEED_KMH) * 60
    return total_m, minutes

quick_dist, quick_mins = route_stats(G, quickest_route)
safe_dist,  safe_mins  = route_stats(G, safest_route)

def format_time(minutes):
    if minutes < 60:
        return f"{int(round(minutes))} min"
    else:
        h = int(minutes // 60)
        m = int(round(minutes % 60))
        return f"{h}h {m}min"

# -------------------
# Plot side by side
# -------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 9))
fig.patch.set_facecolor("#1a1a2e")

titles = [
    f"⚡ Quickest Route\n{quick_dist:.0f}m  •  {format_time(quick_mins)} walking",
    f"🛡️ Safest Route\n{safe_dist:.0f}m  •  {format_time(safe_mins)} walking",
]
routes   = [quickest_route, safest_route]
colours  = ["#00d4ff", "#00ff99"]

for ax, route, title, colour in zip(axes, routes, titles, colours):

    # Draw base graph
    ox.plot_graph(G, ax=ax, node_size=0, edge_color="#333355",
                  edge_linewidth=0.5, bgcolor="#1a1a2e", show=False, close=False)

    # Draw route
    route_edges = list(zip(route[:-1], route[1:]))
    edge_xs, edge_ys = [], []
    for u, v in route_edges:
        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
        edge_xs += [x1, x2, None]
        edge_ys += [y1, y2, None]
    ax.plot(edge_xs, edge_ys, color=colour, linewidth=3, zorder=4)

    # Start and end markers
    ax.scatter(G.nodes[start_node]['x'], G.nodes[start_node]['y'],
               c="white", s=120, zorder=6, marker="o", label="Start")
    ax.scatter(G.nodes[goal_node]['x'],  G.nodes[goal_node]['y'],
               c=colour,  s=150, zorder=6, marker="*", label="End")

    # Safety features
    cctv_gdf.plot(ax=ax,   color="red",    markersize=4, zorder=5, alpha=0.6)
    lights_gdf.plot(ax=ax, color="yellow", markersize=2, zorder=4, alpha=0.4)

    # Zoom to route
    xs = [G.nodes[n]['x'] for n in route]
    ys = [G.nodes[n]['y'] for n in route]
    margin = 0.003
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=12)
    ax.set_axis_off()

# Legend
legend_handles = [
    mpatches.Patch(color="#00d4ff",  label="Quickest route"),
    mpatches.Patch(color="#00ff99",  label="Safest route"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",  markersize=8, label="Start"),
    plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#aaaaaa", markersize=10, label="End"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red",    markersize=6,  label="CCTV"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="yellow", markersize=6,  label="Street light"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=6,
           facecolor="#1a1a2e", labelcolor="white", fontsize=10, framealpha=0.8)

plt.suptitle("Wills Memorial  →  Thekla", color="white", fontsize=16,
             fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.show()

# -------------------
# Print summary
# -------------------
print("\n========== ROUTE SUMMARY ==========")
print(f"  Quickest:  {quick_dist:.0f}m  |  {format_time(quick_mins)}")
print(f"  Safest:    {safe_dist:.0f}m  |  {format_time(safe_mins)}")
extra_m    = safe_dist - quick_dist
extra_mins = safe_mins - quick_mins
if extra_m > 0:
    print(f"\n  Taking the safest route adds {extra_m:.0f}m ({format_time(extra_mins)}) to your journey.")
else:
    print(f"\n  The safest route is also the shortest — lucky!")
print("====================================\n")