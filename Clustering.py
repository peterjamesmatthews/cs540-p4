import numpy as np


class Node:

    def __init__(self, param1, param2, param3):
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        self.cluster = None

    def __str__(self):
        return f'({round(self.param1, 2)}, {round(self.param2, 2)}, {round(self.param3, 2)})'

    def euclidean(self, other) -> float:
        return ((self.param1 - other.param1)**2 +
                (self.param2 - other.param2)**2 +
                (self.param3 - other.param3)**2) ** 0.5


class Cluster:

    def __init__(self, nodes):
        self.nodes = nodes
        self.center = Cluster.compute_center(nodes)

    @staticmethod
    def compute_center(nodes):
        x = 0.0
        y = 0.0
        z = 0.0
        for node in nodes:
            x += node.param1
            y += node.param2
            z += node.param3
        x /= len(nodes)
        y /= len(nodes)
        z /= len(nodes)
        return Node(x, y, z)

    def distortion(self):
        distortion = 0
        for node in self.nodes:
            distortion += node.euclidean(self.center)
        return distortion

    @staticmethod
    def single_linkage(cluster1, cluster2):
        distances = np.zeros(shape=(len(cluster1.nodes), len(cluster2.nodes)), dtype=float)
        for i, node1 in enumerate(cluster1.nodes):
            for j, node2 in enumerate(cluster2.nodes):
                distances[i, j] = node1.euclidean(node2)
        single = np.argmin(distances)
        single = np.unravel_index(single, distances.shape)
        return distances[single[0], single[1]]

    @staticmethod
    def merge(cluster1, cluster2):
        return Cluster(cluster1.nodes + cluster2.nodes)

    def __str__(self):
        return str(self.nodes)
