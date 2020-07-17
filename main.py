from Clustering import Node, Cluster
import re
import numpy as np
import pandas as pd
import random
import requests as r

globalDeaths = r.get(
    "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series"
    "/time_series_covid19_deaths_global.csv")
globalDeaths.raise_for_status()
globalDeaths = re.sub("\"(.+), (.+)\"", r'\g<2> \g<1>', globalDeaths.text)
globalDeaths = globalDeaths.split('\n')
for i in range(len(globalDeaths)):
    globalDeaths[i] = globalDeaths[i].split(',')
usDeaths = pd.read_csv(
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series"
    "/time_series_covid19_deaths_US.csv")
DEBUG = False


def question1() -> str:
    # fix "Korea, South" and other one
    canada = None
    for record in globalDeaths[1:-1]:  # skip header and new line at the end of file
        if record[1] == 'Canada':
            if canada is None:
                canada = np.array(record[4:], dtype=int)
            else:
                canada += np.array(record[4:], dtype=int)

    us = np.zeros((1, len(canada)), dtype=int)
    for i, row in usDeaths.iterrows():
        us += (row.iloc[12:]).values.astype(int)

    ret = ''
    for day in canada[:-1]:
        ret += f'{day},'
    ret += f'{canada[-1]}\n'
    for day in us[0][:-1]:
        ret += f'{day},'
    ret += f'{us[0][-1]}'
    return ret


def question2() -> str:
    # fix "Korea, South" and other one
    canada = None
    for record in globalDeaths[1:-1]:
        if record[1] == 'Canada':
            if canada is None:
                canada = np.array(record[4:], dtype=int)
            else:
                canada += np.array(record[4:], dtype=int)
    us = np.zeros((1, len(canada)), dtype=int)
    for i, row in usDeaths.iterrows():
        us += (row.iloc[12:]).values.astype(int)
    us = us[0]
    assert len(us) == len(canada), f'len(us) = {len(us)}\tlen(canada) = {len(canada)}'

    us_diff = np.zeros((1, len(us) - 1), dtype=int)
    canada_diff = np.zeros((1, len(canada) - 1), dtype=int)
    for i in range(len(us_diff[0])):
        us_diff[0][i] = us[i + 1] - us[i]
        canada_diff[0][i] = canada[i + 1] - canada[i]

    ret = ''
    for diff in canada_diff[0][:-1]:
        ret += f'{diff},'
    ret += f'{canada_diff[0][-1]}\n'
    for diff in us_diff[0][:-1]:
        ret += f'{diff},'
    ret += f'{us_diff[0][-1]}'
    return ret


def parameters() -> (str, list):
    june27th = globalDeaths[0].index('6/27/20')
    parameter_estimates = list()
    for record in globalDeaths[1:-1]:
        lat = float(record[2])
        long = float(record[3])
        j27 = int(record[june27th])
        parameter_estimates.append((lat, long, j27))
    ret = ""
    for estimate in parameter_estimates[:-1]:
        ret += str(round(estimate[0], 2)) + ',' + str(round(estimate[1], 2)) + ',' + str(estimate[2]) + '\n'
    ret += str(round(parameter_estimates[-1][0], 2)) + ',' + \
           str(round(parameter_estimates[-1][1], 2)) + ',' + \
           str(parameter_estimates[-1][2])
    return ret, parameter_estimates


def question5() -> str:
    k = 8
    m = parameters()[1]
    for i, x in enumerate(m):
        m[i] = Node(x[0], x[1], x[2])
    clusters = list()
    # make a cluster for each node
    for x in m:
        clusters.append(Cluster([x]))

    while len(clusters) > k:  # stop when there are 8 clusters
        # compute matrix of distances between clusters using single linkage
        distances = np.zeros(shape=(len(clusters), len(clusters)), dtype=float)
        for i in range(len(distances)):
            for j in range(len(distances[i])):
                if i >= j:
                    distances[i, j] = float('inf')
                else:
                    distances[i, j] = Cluster.single_linkage(clusters[i], clusters[j])
        # merge closest clusters
        merge = np.unravel_index(np.argmin(distances), distances.shape)
        cluster1 = clusters[merge[0]]
        cluster2 = clusters[merge[1]]
        clusters[merge[0]] = Cluster.merge(cluster1, cluster2)
        clusters = np.delete(clusters, merge[1])
    for i in range(len(clusters)):
        for node in clusters[i].nodes:
            node.cluster = i
    ret = ""
    for node in m:
        ret += f'{node.cluster},'
    return ret[:-1]  # cut off last ','


def question6() -> str:
    k = 8
    m = parameters()[1]
    for i, x in enumerate(m):
        m[i] = Node(x[0], x[1], x[2])
    clusters = list()
    # make a cluster for each node
    for x in m:
        clusters.append(Cluster([x]))
    while len(clusters) > k:  # stop when there are 8 clusters
        # compute matrix of distances between clusters using single linkage
        distances = np.zeros(shape=(len(clusters), len(clusters)), dtype=float)
        for i in range(len(distances)):
            for j in range(len(distances[i])):
                if i >= j:
                    distances[i, j] = float('inf')
                else:
                    distances[i, j] = Cluster.complete_linkage(clusters[i], clusters[j])
        # merge closest clusters
        merge = np.unravel_index(np.argmin(distances), distances.shape)
        cluster1 = clusters[merge[0]]
        cluster2 = clusters[merge[1]]
        clusters[merge[0]] = Cluster.merge(cluster1, cluster2)
        clusters = np.delete(clusters, merge[1])
    for i in range(len(clusters)):
        for node in clusters[i].nodes:
            node.cluster = i
    ret = ""
    for node in m:
        ret += f'{node.cluster},'
    return ret[:-1]  # cut off last ','


def question7() -> (str, list):
    k = 8
    m = parameters()[1]
    n = len(m)
    for i, x in enumerate(m):
        m[i] = Node(x[0], x[1], x[2])
    clusters = random.sample(m, k)
    for i, node in enumerate(clusters):
        clusters[i] = Cluster([node])
    lastDistortion = -1
    while True:
        # calculate distortion
        totalDistortion = 0
        for cluster in clusters:
            totalDistortion += cluster.distortion
        if lastDistortion == totalDistortion:
            break
        # assign each point to its closest center
        newClusters = [list() for i in range(k)]
        distances = np.zeros(k, dtype=float)
        for i in range(n):
            for j in range(k):
                distances[j] = m[i].euclidean(clusters[j].center)
            newCluster = int(np.argmin(distances))
            newClusters[newCluster].append(m[i])
        # update all cluster centers as the center of points
        clusters = list()
        for i in range(k):
            clusters.append(Cluster(newClusters[i]))
        lastDistortion = totalDistortion
    for i in range(len(clusters)):
        for node in clusters[i].nodes:
            node.cluster = i
    ret = ""
    for node in m:
        ret += f'{node.cluster},'
    return ret[:-1], clusters  # cut off last ','


def question8(clusters) -> str:
    ret = ""
    for cluster in clusters:
        ret += f'{str(cluster)}\n'
    return ret[:-1]


def question9(clusters) -> str:
    td = 0
    for cluster in clusters:
        td += cluster.q9_hack()
    return str(td)


def main():
    q7, clusters = question7()
    if DEBUG:
        return 1
    with open('P4.txt', 'w') as f:
        f.write('Outputs:\n')
        f.write('@id\n')
        f.write('pjmatthews\n')
        f.write('@original\n')
        f.write(f'{question1()}\n')
        f.write('@difference\n')
        f.write(f'{question2()}\n')
        f.write('@answer_3\n')
        f.write(f'Used TA\'s parameters\n')
        f.write('@parameters\n')
        f.write(f'{parameters()[0]}\n')
        f.write(f'@hacs\n')
        f.write(f'{question5()}\n')
        f.write('@hacc\n')
        f.write(f'{question6()}\n')
        f.write('@kmeans\n')
        f.write(f'{q7}\n')
        f.write('@centers\n')
        f.write(f'{question8(clusters)}\n')
        f.write('@answer_9\n')
        f.write(f'{question9(clusters)}\n')
        f.write('@answer_10\n')
        f.write('None\n')
    pass


if __name__ == "__main__":
    main()
