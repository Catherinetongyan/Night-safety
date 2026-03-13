import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import networkx as nx
from pyproj import Transformer
import pickle

print("Downloading graph...")
G = ox.graph_from_place("Bristol, United Kingdom", network_type="walk")

cctv_df = pd.read_csv(r"C:\Users\Oliver\Documents\Council_cctv_cameras_longlat.csv")
lights_df = pd.read_csv(r"C:\Users\Oliver\Documents\Council_streetlights_longlat.csv")

cctv_gdf = gpd.GeoDataFrame(cctv_df, geometry=gpd.points_from_xy(cctv_df.longitude, cctv_df.latitude), crs="EPSG:4326").to_crs(epsg=3857)
lights_gdf = gpd.GeoDataFrame(lights_df, geometry=gpd.points_from_xy(lights_df.longitude, lights_df.latitude), crs="EPSG:4326").to_crs(epsg=3857)

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def safety_score(point):
    buffer = point.buffer(40)
    cctv_count = cctv_gdf[cctv_gdf.intersects(buffer)].shape[0]
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
    length = data.get("length", 1)
    data["safety_weight"] = max(1, length - (safety * 5))

print("Saving graph...")
with open(r"C:\Users\Oliver\Documents\bristol_safety_graph.pkl", "wb") as f:
    pickle.dump(G, f)

# Save GDFs for plotting later
cctv_gdf.to_crs(epsg=4326).to_file(r"C:\Users\Oliver\Documents\cctv.gpkg", driver="GPKG")
lights_gdf.to_crs(epsg=4326).to_file(r"C:\Users\Oliver\Documents\lights.gpkg", driver="GPKG")

print("Done! Run route.py whenever you want a route.")