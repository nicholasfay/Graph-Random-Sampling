import networkx as nx
import numpy as np
import argparse
import sys
from categorical import Categorical as C
from datetime import datetime
from collections import deque, Counter
import matplotlib.pyplot as plt

startTime = datetime.now()

debug = open("test.txt", 'w')

def generateOutDegreeGraph(G, inFile, add, sample):
    if sample:
        out_degree = G.out_degree(sample)
    else:
        out_degree = G.out_degree()
    out_degree_vals = sorted(set(out_degree.values()))
    c = Counter(out_degree.values())
    out_degree_distr = [c[x] for x in out_degree_vals]
    temp = np.array(out_degree_distr)
    mean = temp.mean()
    n1 = float(sum(out_degree_distr))
    n1A = np.ones(len(out_degree_distr)) * n1
    norm_out_degree_distr = out_degree_distr / n1A
    plt.figure()
    plt.plot(out_degree_vals, norm_out_degree_distr, 'ro-')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Degree')
    plt.ylabel('Percentage of nodes')
    title = 'Out-Degree Distribution for {}'.format(inFile)
    plt.title(title)
    outGraph = 'stats/{}-{}-degree-distribution.jpg'.format(inFile, add)
    plt.savefig(outGraph)
    plt.close()


def labelNode(G, v):
    if 'collected' not in G.node[v]:
        G.node[v]['collected'] = False
    if 'sampled' not in G.node[v]:
        G.node[v]['sampled'] = False
    if 'seed' not in G.node[v]:
        G.node[v]['seed'] = False
    return G


def DURW(G, N1, w):
    outedges = G.out_edges([N1])
    #filter outedges for ones that have a v value that is already sampled/collected
    probarray = np.ones(len(outedges))
    #Dynamic random jump probability and -1 to represent virtual node
    outedges.append(-1)
    #probarray.append(w)
    #print(probarray)
    probarray = np.append(probarray, w)
    sumprob = np.sum(probarray)
    divisor = np.ones(len(probarray)) * sumprob
    probarray = probarray / divisor
    #print(probarray)
    #print("{} outedges length {} probarray length".format(len(outedges), len(probarray)))
    choice = np.random.choice(outedges, 1, p=probarray)

    if np.random.uniform() < (w / (w + deg)) or len(edgelist) == 0:
        idx2 = uni.sample()
        picked_node = nodes[idx2]
        temp = 0
    else:
        my_sampler = C(scores)
        idx = my_sampler.sample()
        picked_node = edgelist[idx][1]
        temp = 1
    if choice[0] == -1:
        return 'randomJump', None
    else:
        return 'navigate', choice[0][1]

def findNodesToCollect(G, Gsample, toNav, w):
    newNav = deque()
    while(toNav):
        #Pop left assumes FIFO is desired behavior
        N1 = toNav.popleft()
        #print(type(G.node))
        if G.node[N1]['collected']:
            #Add nodes to sample from
            Gsample.add_edges_from(G.out_edges([N1], data=True))
            G.node[N1]['sampled'] = True
            G = labelNode(G, N1)
            nextaction, nextnode = DURW(G, N1, w)
            #Adds the new node to the navigation queue at the top of the queue
            if nextaction is 'navigate':
                toNav.appendleft(nextnode)
                print("Nav")
            elif nextaction is 'randomJump':
                #makes sure that new jump only is a collected, not sampled, seed node
                #print(G.node.items())
                #for k,attrdict in G.node.items():
                #   print("{} k {} attrdict".format(k, attrdict), file=debug)
                collectedseed = [k for k,attrdict in G.node.items() if attrdict['collected'] is True and attrdict['sampled'] is False and attrdict['seed'] is True]
                if not collectedseed:
                    break
                N2 = np.random.choice(collectedseed, 1)
                #print(type(N2))
                #print(type(N2[0]))
                print("RJ")
                toNav.appendleft(N2[0])
            else:
                print("Didn't get proper return")
        else:
            newNav.appendleft(N1)

    return newNav

def main():
    # Argument parsing for various options
    parser = argparse.ArgumentParser(description="Stream Sampling")
    parser.add_argument('-st', '--stream', type=str, default=False,
        help='Stream Variable: True = Streaming, False = No Streaming')
    parser.add_argument('-se', '--seedf', type=str, required=True, help='File that has random set of nodes for the SEED')
    parser.add_argument('-k', '--kval', type=int, required=True, help='Number of nodes that can be collected at once with streaming method')
    parser.add_argument('-nn', '--numnodes', type=str, required=True, help='File that has random set of nodes for the SEED')
    parser.add_argument('-tg', '--testgraphf', type=str, default=None, help='File for testing without streaming capabilities')
    parser.add_argument('-w', '--weight', type=int, default=10, help='Random Jump Weight')

    args = parser.parse_args()
    if not args.testgraphf and args.stream:
        print("Please supply test graph to substitue streaming capabilities")
        sys.exit()
    #Random Seed Input
    seed = []
    with open(args.seedf) as f:
        for line in f:
            seed.append(int(line))

    if args.testgraphf:
        testG = nx.DiGraph()
        with open(args.testgraphf) as f:
            for line in f:
                # Edge format is <nodeID> <nodeID>
                edge = line.split()
                # Skips Comments
                if edge[0] == '#':
                    continue
                node1 = int(edge[0])
                node2 = int(edge[1])
                # Add edges as they are parsed
                testG.add_edge(node1, node2, weight=1)
        #print("This is testG length {} and this is numnodes {}".format(len(testG.nodes()), int(args.numnodes)))
        if(len(testG.nodes()) <= int(args.numnodes)):
            print("Can't sample more nodes than the graph provided has.")
            sys.exit()
    else:
        print("Need test graph without streaming capabilities enabled")
        sys.exit()

    #Initial sample graph
    Gsample = nx.DiGraph()

    #Unvisited Nodes with seed indicator added and collected inidcator added
    G = nx.DiGraph()
    for node in seed:
        G.add_node(node, collected=False, seed=True, sampled=False)

    #Set of K random nodes specified by user
    IS = np.random.choice(seed, args.kval)

    #Nodes that need to be navigated TODO: How does toNav ever have anything within the queue? navigations are unknown until labeled so and find_nodes_to_collect relies on this being not empty should I just add initial seed nodes?
    toNav = deque()
    #Continuous sampling function
    finished = False
    while(not finished):
        if args.stream:
            data = collect(IS)
            edges = computeEdges(data)
        else:
            #Edge collection of each node in IS
            for node in IS:
                #TODO: Mark edges that are being added as not collected, not sampled, not seed
                edges = testG.out_edges([node], data=True)
                #print(edges)
                G.add_edges_from(edges)
                for u, v, data in edges:
                    G = labelNode(G,v)
                #Signifies that the node has been collected
                G.node[node]['collected'] = True
                G = labelNode(G, node)
                #set to nav to IS here.
                toNav.append(node)
            #toNav = IS
            #print(toNav)
            #print(G.node.items(), file=debug)
            #print("TODODODODODODO", file=debug)
            toNav = findNodesToCollect(G, Gsample, toNav, args.weight)
            #print("{} Gsample length, {} NN length".format(len(Gsample.nodes()), args.numnodes))
            #???????Is >= good here since edges being added, there are nodes being added along with it
            sampled = [n for n,attrdict in G.node.items() if attrdict['sampled'] == True ]
            if len(Gsample.nodes()) >= int(args.numnodes):
                finished = True
                generateOutDegreeGraph(Gsample, args.testgraphf, "sample", sampled)
                generateOutDegreeGraph(G, args.testgraphf, "collected", None)
                generateOutDegreeGraph(testG, args.testgraphf, "test", None)
            #Set the collected nodes
            IS = np.array(toNav)
            if len(IS) < args.kval:
                k1 = args.kval - len(IS)
                #Filters out nodes in original seed that have already been collected or sampled
                filterseed = [k for k in seed if G.node[k]['collected'] is False and G.node[k]['sampled'] is False]
                #if not filterseed:
                 #   continue
                #Selects k1 new random nodes
                newNodes = np.random.choice(filterseed, k1)
                #print(IS)
                #
                #print(newNodes)
                IS = np.hstack((IS,newNodes))



    print(datetime.now() - startTime)

if __name__ == '__main__':
    main()
