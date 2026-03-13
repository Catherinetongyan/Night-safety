import math

class Node:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Node({self.id}, {self.x}, {self.y})"


class Edge:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

    def length(self):
        return math.sqrt(
            (self.node1.x - self.node2.x)**2 +
            (self.node1.y - self.node2.y)**2
        )

    def __repr__(self):
        return f"Edge({self.node1.id} -> {self.node2.id})"