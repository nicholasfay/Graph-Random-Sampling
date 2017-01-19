import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import argparse
import powerlaw
from collections import Counter


def calculateNMSE(outd1, outd2):
    print(len(outd1))
    print(len(outd2))

# Indicator functions


def outDegreeIndicator(outdegree, j, node):
    if(outdegree[node] == j):
        return 1
    return 0


def inDegreeIndicator(indegree, j, node):
    if(indegree[node] == j):
        return 1
    return 0

# Calculating S for the steady state probability


def calculateS(degree, selected, w, n):
    ret = 0
    # Weird error that happens sometimes here, says that
    # some of the nodes in the selected nodes aren't nodes in
    # the sampled graph
    for item in selected:
        added = 0
        if item in degree:
            added = degree[item]
        else:
            added = 0
        ret += (1.0 / (w + added))
    return ret / n

# Steady state probability of sampling a node


def piFunc(degree, item, w):
    degree = degree[item]
    return (w + degree)

# Driver function to do in-degree and out-degree sample distribution estimators


def distributionEstimatorOut(outdegreeDict, dd, dd2, selected, inFile, w):
    phi = {}
    n = float(len(selected))
    S = calculateS(dd2, selected, w, n)
    out_degree_vals = sorted(set(outdegreeDict.values()))
    for item in out_degree_vals:
        ret = 0
        for item2 in selected:
            indicator = outDegreeIndicator(outdegreeDict, item, item2)
            pi = piFunc(dd, item2, w)
            pi = pi * S
            ret += (indicator / pi) / n
        phi[item] = ret
    return phi


def distributionEstimatorIn(indegreeDict, dd, dd2, selected, inFile, w):
    phi = {}
    n = float(len(selected))
    S = calculateS(dd2, selected, w, n)
    in_degree_vals = sorted(set(indegreeDict.values()))
    for item in in_degree_vals:
        ret = 0
        for item2 in selected:
            indicator = inDegreeIndicator(indegreeDict, item, item2)
            pi = piFunc(dd, item2, w)
            pi = pi * S
            ret += (indicator / pi) / n
        phi[item] = ret
    return phi


# Computes out_degree distribution, in_degree distribution
# and clustering coefficient of sampled graph after DURW
def graphSampleStatistics(origG, sampledG, selected, inFile, w, outdegree):
    outName = 'stats/stats-{}-sample-w{}.txt'.format(inFile, str(w))
    outFile = open(outName, 'w')
    print('Statistics for input graph sample-{}-w{}'.format(inFile, str(w)), file=outFile)

    od = origG.out_degree(selected)
    id = origG.in_degree(selected)
    dd = origG.degree(selected)
    dd2 = sampledG.degree(selected)
    out_degree = distributionEstimatorOut(
        od, dd, dd2, selected, inFile, w)
    in_degree = distributionEstimatorIn(
        id, dd, dd2, selected, inFile, w)
    plt.figure()
    plt.yscale('log')
    plt.xscale('log')
    outKeys = [float(x) for x in out_degree.keys()]
    outVals = list(out_degree.values())
    inKeys = [float(x) for x in in_degree.keys()]
    inVals = list(in_degree.values())
    plt.plot(outKeys, outVals, 'ro-')
    outKeys.remove(0)
    fit = powerlaw.Fit(outKeys)
    print("Alpha 1: {}".format(fit.alpha))
    plt.plot(inKeys, inVals, 'bv-')
    plt.legend(['Out-degree', 'In-degree'])
    plt.xlabel('Degree')
    plt.ylabel('Percentage of nodes')
    title = 'In-Degree and Out-Degree Distributions for {}'.format(inFile)
    plt.title(title)
    outGraph = 'stats/{}-degree-distribution-sample-{}.jpg'.format(inFile, w)
    plt.savefig(outGraph)
    plt.close()

    print('In-Degree and Out-Degree have been plotted and saved at {}'.format(outGraph), file=outFile)
    print('Clustering Coefficient:', file=outFile)
    cluster = nx.average_clustering(sampledG)
    print(cluster, file=outFile)

    calculateNMSE(outdegree, out_degree.values())

# Computes out_degree distribution, in_degree distribution
# and clustering coefficient of unsampled graph


def graphStatistics(G, inFile):
    outName = 'stats/stats-{}-original.txt'.format(inFile)
    outFile = open(outName, 'w')
    print('Statistics for input graph {}'.format(inFile), file=outFile)

    out_degree = G.out_degree()
    out_degree_vals = sorted(set(out_degree.values()))
    c = Counter(out_degree.values())
    out_degree_distr = [c[x] for x in out_degree_vals]
    n1 = float(sum(out_degree_distr))
    n1A = np.ones(len(out_degree_distr)) * n1
    norm_out_degree_distr = out_degree_distr / n1A
    in_degree = G.in_degree()
    in_degree_vals = sorted(set(in_degree.values()))
    c2 = Counter(in_degree.values())
    in_degree_distr = [c2[x] for x in in_degree_vals]
    n2 = float(sum(in_degree_distr))
    n2A = np.ones(len(in_degree_distr)) * n2
    norm_in_degree_distr = in_degree_distr / n2A
    # print(norm_in_degree_distr)
    fit = powerlaw.Fit(out_degree_vals)
    print("Alpha 2: {}".format(fit.alpha))
    plt.figure()
    plt.plot(out_degree_vals, norm_out_degree_distr, 'ro-')
    plt.plot(in_degree_vals, norm_in_degree_distr, 'bv-')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(['Out-degree', 'In-degree'])
    plt.xlabel('Degree')
    plt.ylabel('Percentage of nodes')
    title = 'In-Degree and Out-Degree Distributions for {}'.format(inFile)
    plt.title(title)
    outGraph = 'stats/{}-degree-distribution.jpg'.format(inFile)
    plt.savefig(outGraph)
    plt.close()

    print('In-Degree and Out-Degree have been plotted and saved at {}'.format(outGraph), file=outFile)
    cluster = nx.average_clustering(G.to_undirected())
    print('Clustering Coefficient:', file=outFile)
    print(cluster, file=outFile)

    return out_degree_distr


if __name__ == '__main__':
    # Argument parsing for various options
    parser = argparse.ArgumentParser(description="Generate Sampled Graph")
    parser.add_argument('-f', '--inFile', type=str, required=True,
                        help='Input graph file in form .gpickle (.txt support will be added)')
    parser.add_argument('-s', '--sample', type=bool, required=True,
                        help='True if Sampled Graph, False if full Graph')
    args = parser.parse_args()

    G = nx.read_gpickle(args.inFile)
    if(args.sample):
        graphSampleStatistics(G, args.inFile, 1)
    else:
        graphStatistics(G, args.inFile, 1)
