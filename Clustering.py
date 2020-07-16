class Node:

    def __init__(self, param1, param2, param3):
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3

    def __str__(self):
        return f'({round(self.param1, 2)}, {round(self.param2, 2)}, {round(self.param3, 2)})'

    def euclidean(self, other) -> float:
        return ((self.param1 - other.param1)**2 +
                (self.param2 - other.param2)**2 +
                (self.param3 - other.param3)**2) ** 0.5

class Cluster:

    def __init__(self, nodes, center):
        self.nodes = nodes
        self.center = center
