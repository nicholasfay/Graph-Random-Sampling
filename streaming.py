import networkx as nx
import numpy as np
import argparse
import sys
from datetime import datetime
from collections import deque

startTime = datetime.now()

def DURW(G, N1, w):
    outedges = G.out_edges([N1])
    probarray = np.ones(len(outedges))
    #Dynamic random jump probability and -1 to represent virtual node
    outedges.append(-1)
    probarray.append(w)
    choice = np.random.choice(outedges, 1, p=probarray)
    print(choice)
    if choice == -1:
        return 'randomJump', None
    else:
        return 'navigate', choice

def findNodesToCollect(G, Gsample, toNav, w):
    newNav = deque()
    nodes = G.nodes()
    while(toNav):
        #Pop left assumes FIFO is desired behavior
        N1 = toNav.popleft()
        if nodes[N1]['collected']:
            #Add nodes to sample from
            Gsample.add_edges_from(G.out_edges([N1], data=True))
            G.node[N1]['sampled'] = True
            nextaction, nextnode = DURW(G, N1, w)
            #Adds the new node to the navigation queue at the top of the queue
            if nextaction is 'navigate':
                toNav.appendleft(nextnode)
            elif nextaction is 'randomJump':
                #makes sure that new jump only is a collected, not sampled, seed node
                collectedseed = [k for k in nodes if k['collected'] is True and k['sampled'] is False and k['seed'] is True]
                N2 = np.random.choice(collectedseed, 1)
                toNav.appendleft(N2)
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
    else:
        print("Need test graph without streaming capabilities enabled")
        sys.exit()

    #Initial sample graph
    Gsample = nx.Graph()

    #Unvisited Nodes with seed indicator added and collected inidcator added
    G = nx.DiGraph()
    for node in seed:
        G.add_node(node, collected=False, seed=True, sampled=False)

    #Set of K random nodes specified by user
    IS = np.random.choice(seed, args.kval)

    #Nodes that need to be navigated
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
                G.add_edges_from(testG.out_edges([node], data=True))
                #Signifies that the node has been collected
                G.node[node]['collected'] = True
            toNav = findNodesToCollect(G, Gsample, toNav, args.weight)
            if len(Gsample.nodes()) == args.numnodes:
                finished = True
            #Set the collected nodes 
            IS = toNav
            if len(IS) < args.kval:
                k1 = args.kval - len(IS)
                #Filters out nodes in original seed that have already been collected or sampled
                filterseed = [k for k in seed if G.node[k]['collected'] is False and G.node[k]['sampled'] is False]
                #Selects k1 new random nodes
                newNodes = np.random.choice(filterseed, k1)



    print(datetime.now() - startTime)

if __name__ == '__main__':
    main()
