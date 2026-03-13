from Graph.graph_structures import Node, Edge
import random
import math
import pickle
import os  # <-- Add this import

GRAPH_FILE = "Graph/graph_data.pkl"

NUM_NODES = 40
MAP_SIZE = 100

MIN_NODE_DISTANCE = 8
K_NEIGHBORS = 3

# Delete the file if it already exists
if os.path.exists(GRAPH_FILE):
    os.remove(GRAPH_FILE)
    print(f"{GRAPH_FILE} already existed and was deleted.")


def generate_nodes():

    nodes = []

    while len(nodes) < NUM_NODES:

        x = random.uniform(0, MAP_SIZE)
        y = random.uniform(0, MAP_SIZE)

        valid = True

        for n in nodes:

            if math.sqrt((x-n.x)**2 + (y-n.y)**2) < MIN_NODE_DISTANCE:
                valid = False
                break

        if valid:
            nodes.append(Node(len(nodes), x, y))

    return nodes


def generate_edges(nodes):

    edges = []

    for node in nodes:

        distances = []

        for other in nodes:

            if node != other:

                dist = math.sqrt(
                    (node.x-other.x)**2 +
                    (node.y-other.y)**2
                )

                distances.append((dist, other))

        distances.sort(key=lambda x: x[0])

        count = 0

        for dist, other in distances:

            if count >= K_NEIGHBORS:
                break

            if any(
                (e.node1 == node and e.node2 == other) or
                (e.node1 == other and e.node2 == node)
                for e in edges
            ):
                continue

            edges.append(Edge(node, other))
            count += 1

    return edges


def generate_graph():

    nodes = generate_nodes()
    edges = generate_edges(nodes)

    graph = {
        "nodes": nodes,
        "edges": edges
    }

    with open(GRAPH_FILE, "wb") as f:
        pickle.dump(graph, f)

    print("Graph saved.")


if __name__ == "__main__":
    generate_graph()