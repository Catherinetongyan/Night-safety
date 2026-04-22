# Night-safety

All input parameters are in config.yaml, only change this file to use different parameters.

## Data loading

Run Mapping_software.py to load and save the weighted graph.

Output: bristol_weighted_graph.pkl, cctv.gpkg, lights.gpkg

## A* method

Run safe_route.py to get the result routes using A* search method

## Heat method

Run heat_routing.py to get the result routes using the heat method

The comparison is A: Shortest route using A*, B: A* full. C: heat full
