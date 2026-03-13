from Graph.graph_structures import Node, Edge
import pickle
import numpy as np
import matplotlib.pyplot as plt

GRAPH_FILE = ("Graph/graph_data.pkl") #Graph/graph_data.pkl
MAP_SIZE = 100
GRID_RESOLUTION = 200


danger_points = [

    (30, 40),
    (70, 70),
    (50, 20)

]

MAX_RADIUS = 20


def heat_function(d):

    if d > MAX_RADIUS:
        return 0

    return (MAX_RADIUS - d) / MAX_RADIUS


def generate_heatmap():

    grid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION))

    xs = np.linspace(0, MAP_SIZE, GRID_RESOLUTION)
    ys = np.linspace(0, MAP_SIZE, GRID_RESOLUTION)

    for i, x in enumerate(xs):

        for j, y in enumerate(ys):

            heat = 0

            for dx, dy in danger_points:

                d = np.sqrt((x-dx)**2 + (y-dy)**2)

                heat += heat_function(d)

            grid[j][i] = heat

    return grid


def plot_heatmap(grid):

    plt.imshow(
        grid,
        extent=[0, MAP_SIZE, 0, MAP_SIZE],
        origin="lower",
        cmap="hot",
        alpha=0.6
    )

    plt.colorbar(label="Danger Level")


def overlay_graph():

    with open(GRAPH_FILE , "rb") as f:
        graph = pickle.load(f)

    nodes = graph["nodes"]
    edges = graph["edges"]

    for e in edges:

        plt.plot(
            [e.node1.x, e.node2.x],
            [e.node1.y, e.node2.y],
            color="white",
            linewidth=1
        )

    for n in nodes:

        plt.scatter(n.x, n.y, color="cyan", s=10)


def main():

    grid = generate_heatmap()

    plot_heatmap(grid)

    overlay_graph()

    plt.title("Street Safety Heatmap")
    plt.show()


if __name__ == "__main__":
    main()