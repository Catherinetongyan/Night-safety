from Graph.graph_structures import Node, Edge
import pickle
import heapq
import random
import math
import matplotlib.pyplot as plt

GRAPH_FILE = "Graph/graph_data.pkl"


def distance(a, b):

    return math.sqrt(
        (a.x - b.x)**2 +
        (a.y - b.y)**2
    )


def build_adjacency(nodes, edges):

    adjacency = {n: [] for n in nodes}

    for e in edges:

        adjacency[e.node1].append(e.node2)
        adjacency[e.node2].append(e.node1)

    return adjacency


def a_star(start, goal, adjacency):

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:

        current = heapq.heappop(open_set)[1]

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in adjacency[current]:

            tentative = g_score[current] + distance(current, neighbor)

            if neighbor not in g_score or tentative < g_score[neighbor]:

                came_from[neighbor] = current
                g_score[neighbor] = tentative

                f = tentative + distance(neighbor, goal)

                heapq.heappush(open_set, (f, neighbor))

    return None


def reconstruct_path(came_from, current):

    path = [current]

    while current in came_from:

        current = came_from[current]
        path.append(current)

    path.reverse()

    return path


def plot_graph(nodes, edges, path, start, goal):

    # draw edges
    for e in edges:

        x1, y1 = e.node1.x, e.node1.y
        x2, y2 = e.node2.x, e.node2.y

        plt.plot(
            [x1, x2],
            [y1, y2],
            color="gray"
        )

        # weight (distance)
        w = distance(e.node1, e.node2)

        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2

        plt.text(mx, my, f"{w:.1f}", fontsize=8, color="blue")

    # draw nodes
    for n in nodes:
        plt.scatter(n.x, n.y, color="black")

    # highlight start
    plt.scatter(start.x, start.y, color="green", s=120, label="Start")

    # highlight goal
    plt.scatter(goal.x, goal.y, color="red", s=120, label="Goal")

    # draw path
    if path:
        xs = [n.x for n in path]
        ys = [n.y for n in path]

        plt.plot(xs, ys, linewidth=3, color="orange", label="Path")

    plt.legend()
    plt.show()


def main():

    with open(GRAPH_FILE, "rb") as f:
        graph = pickle.load(f)

    nodes = graph["nodes"]
    edges = graph["edges"]

    adjacency = build_adjacency(nodes, edges)

    start = random.choice(nodes)
    goal = random.choice(nodes)

    path = a_star(start, goal, adjacency)

    plot_graph(nodes, edges, path, start, goal)


if __name__ == "__main__":
    main()