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
        self.distortion = Cluster.compute_distortion(nodes, self.center)

    @staticmethod
    def compute_center(nodes) -> Node:
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

    @staticmethod
    def compute_distortion(nodes, center):
        distortion = 0
        for node in nodes:
            distortion += node.euclidean(center)
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
    def complete_linkage(cluster1, cluster2):
        distances = np.zeros(shape=(len(cluster1.nodes), len(cluster2.nodes)), dtype=float)
        for i, node1 in enumerate(cluster1.nodes):
            for j, node2 in enumerate(cluster2.nodes):
                distances[i, j] = node1.euclidean(node2)
        complete = np.argmax(distances)
        complete = np.unravel_index(complete, distances.shape)
        return distances[complete[0], complete[1]]

    @staticmethod
    def merge(cluster1, cluster2):
        return Cluster(cluster1.nodes + cluster2.nodes)

    def __str__(self):
        return f'{round(self.center.param1,4)},{round(self.center.param2,4)},{round(self.center.param1,3)}'
