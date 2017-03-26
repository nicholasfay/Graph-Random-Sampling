import networkx as nx
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse
import powerlaw
from collections import Counter

def calculateNMSE(outd1, outd2):
    #outd1 and outd2 should be of form
    #{out-degreeVal : countWval}
    # Missing lists of keys that need 0 counts to satisfy distribution discrepancy
    missing1 = [k for k in outd1 if k not in outd2]
    missing2 = [k for k in outd2 if k not in outd1]
    # 0 buffering for distribution comparisons
    for item in missing1:
        outd2[item] = 0
    for item in missing2:
        outd1[item] = 0

    tuples1 = sorted(outd1.items())
    tuples2 = sorted(outd2.items())
    filtertuples1 = [k[1] for k in tuples1]

    n2 = float(sum(filtertuples1))
    n2A = np.ones(len(filtertuples1)) * n2
    normalfiltertuples = filtertuples1 / n2A

    filtertuples2 = [k[1] for k in tuples2]
    dist1 = np.array(normalfiltertuples)
    dist2 = np.array(filtertuples2)
    sub = dist2 - dist1
    square = sub ** 2
    expect = square.mean()
    rootexpect = math.sqrt(expect)
    nmse = []
    for i in range(len(normalfiltertuples)):
        if normalfiltertuples[i] == 0.0:
            nmse.append(0)
        else:
            nmse.append(rootexpect / normalfiltertuples[i])
    #nmse = [rootexpect / normalfiltertuples[i] for i in range(len(tuples2)) if not normalfiltertuples[i] == 0]
    degrees = [k[0] for k in tuples1]
    return nmse, degrees


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
    for item in selected:
        added = degree[item]
        ret += (1.0 / (w + added))
    return ret / n

# Steady state probability of sampling a node


def piFunc(degree, item, w):
    if item in degree:
        dg = degree[item]
    else:
        dg = 0
    return (w + dg)

# Driver function to do in-degree and out-degree sample distribution estimators


def distributionEstimatorOut(outdegreeDict, dd2, selected, w, maxout):
    phi = {}
    n = float(len(selected))
    S = calculateS(dd2, selected, w, n)
    for i in range(maxout + 1):
        ret = 0
        for item2 in selected:
            indicator = outDegreeIndicator(outdegreeDict, i, item2)
            pi = piFunc(dd2, item2, w)
            pi *= S
            ret += (indicator / pi)
        ret /= n
        phi[i] = ret
    return phi


def distributionEstimatorIn(indegreeDict, dd2, selected, w):
    phi = {}
    n = float(len(selected))
    S = calculateS(dd2, selected, w, n)
    in_degree_vals = sorted(set(indegreeDict.values()))
    for item in in_degree_vals:
        ret = 0
        for item2 in selected:
            indicator = inDegreeIndicator(indegreeDict, item, item2)
            pi = piFunc(dd2, item2, w)
            pi = pi * S
            ret += (indicator / pi)
        ret = ret / n
        phi[item] = ret
    return phi


# Computes out_degree distribution, in_degree distribution
# and clustering coefficient of sampled graph after DURW
def graphSampleStatistics(origG, sampledG, selected, inFile, w, outdegree, maxout):
    outName = 'stats/stats-{}-sample-w{}.txt'.format(inFile, str(w))
    outFile = open(outName, 'w')
    print('Statistics for input graph sample-{}-w{}'.format(inFile, str(w)), file=outFile)

    od = origG.out_degree()
    id = origG.in_degree()
    dd2 = sampledG.degree()
    out_degree = distributionEstimatorOut(
        od, dd2, selected, w, maxout)
    in_degree = distributionEstimatorIn(
        id, dd2, selected, w)
    outKeys = [float(x) for x in out_degree.keys()]
    outVals = list(out_degree.values())
    inKeys = [float(x) for x in in_degree.keys()]
    inVals = list(in_degree.values())
    plt.figure()
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(outKeys, outVals, 'ro-')
    plt.xlabel('Degree')
    plt.ylabel('Percentage of nodes')
    title = 'Out-Degree Distributions for {}'.format(inFile)
    plt.title(title)
    outGraph = 'stats/{}-outdegree-distribution-sample-{}.jpg'.format(inFile, w)
    plt.savefig(outGraph)
    plt.close()
    #outKeys.remove(0)
    fit = powerlaw.Fit(outKeys)
    print("Power Law Coefficient 2: {}".format(fit.alpha), file=outFile)
    plt.figure()
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(inKeys, inVals, 'bv-')
    plt.xlabel('Degree')
    plt.ylabel('Percentage of nodes')
    title = 'In-Degree Distributions for {}'.format(inFile)
    plt.title(title)
    outGraph2 = 'stats/{}-indegree-distribution-sample-{}.jpg'.format(inFile, w)
    plt.savefig(outGraph2)
    plt.close()

    print('In-Degree and Out-Degree have been plotted and saved at {}'.format(outGraph), file=outFile)
    #print('Clustering Coefficient:', file=outFile)
    #cluster = nx.average_clustering(sampledG)
    #print(cluster, file=outFile)

    NMSE, deg = calculateNMSE(outdegree, out_degree)
    return NMSE, deg

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

    #fit = powerlaw.Fit(out_degree_vals)
    #print("Power Law Coefficient 1: {}".format(fit.alpha), file=outFile)

    plt.figure()
    plt.plot(out_degree_vals, norm_out_degree_distr, 'ro-')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Degree')
    plt.ylabel('Percentage of nodes')
    title = 'Out-Degree Distributions for {}'.format(inFile)
    plt.title(title)
    outGraph = 'stats/{}-outdegree-distribution.jpg'.format(inFile)
    plt.savefig(outGraph)
    plt.close()

    plt.figure()
    plt.plot(in_degree_vals, norm_in_degree_distr, 'bv-')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Degree')
    plt.ylabel('Percentage of nodes')
    title = 'In-Degree Distributions for {}'.format(inFile)
    plt.title(title)
    outGraph2 = 'stats/{}-indegree-distribution.jpg'.format(inFile)
    plt.savefig(outGraph2)
    plt.close()

    print('In-Degree and Out-Degree have been plotted and saved at {}'.format(outGraph), file=outFile)
    #cluster = nx.average_clustering(G.to_undirected())
    #print('Clustering Coefficient:', file=outFile)
    #print(cluster, file=outFile)

    return c


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
