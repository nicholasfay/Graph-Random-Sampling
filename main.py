from sampleDURW import sample
from generateStats import graphSampleStatistics, graphStatistics
import argparse
import os
import sys
import networkx as nx
from datetime import datetime
startTime = datetime.now()


def main():
    # Argument parsing for various options
    parser = argparse.ArgumentParser(
        description="Main Sampling and Statistics Driver Program - Supply relative or absolute path where the Graph(s) reside")
    parser.add_argument('-d', '--inDir', type=str, required=True,
                        help='Name of directory where input graphs reside')
    parser.add_argument('-oG', '--outFileG', type=bool, default=False,
                        help='Whether or not output gpickles of graphs are wanted (May be very taxing with huge graphs)')
    parser.add_argument('-oP', '--outFileP', type=bool, default=False,
                        help='Whether or not output images of graphs are wanted (May be very taxing with huge graphs)')
    parser.add_argument('-it', '--iternum', type=int, default=200000,
                        help='Number of nodes sampled - Default is 20')
    parser.add_argument('-bw', '--bweight', type=float, default=1,
                        help='Weight of edges to determine frequency of random jumps (1:less -->  inf:more, w>0) - default is 1')
    parser.add_argument('-incr', '--increment', type=int, default=1,
                        help='Amount to increment w by for each iteration of sampling')
    parser.add_argument('-amt', '--amount', type=int, default=1,
                        help='Amount of iterations for different samplings based on increment amount')
    parser.add_argument('-ws', '--wset', nargs='+', type=int, default=[False],
                        help='Amount of iterations for different samplings based on increment amount')
    args = parser.parse_args()

    # loop indir
    # perform sampleDURW on each graph, for number of times specified by amount, starting at bw, incrementing by increment each time
    # number of sampling rounds for underlying sampling function
    path = args.inDir
    if(os.path.isdir(path)):
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        count = 0
        for file in os.listdir(path):
            end = 0
            if args.wset[0] is False:
                end = args.amount
            else:
                end = len(args.wset)
            inFile = os.path.dirname(os.path.abspath(
                    __file__)) + "\\testGraphs\\" + file
            # Creates a Directed Graph to load the data into
            G1 = nx.DiGraph()
            print("Beginning Processing of input file")
            sys.stdout.flush()
            # Parses each line and adds an edge.
            # Since the underlying structure is a dictionary
            # there are no duplicates created
            # Creates the initial directed graph, Gd, from the given data.
            with open(inFile) as f:
                for line in f:
                    # Edge format is <nodeID> <nodeID>
                    edge = line.split()
                    # Skips Comments
                    if edge[0] == '#':
                        continue
                    node1 = int(edge[0])
                    node2 = int(edge[1])
                    # Add edges as they are parsed
                    G1.add_edge(node1, node2, weight=1)

            print("Done Processing Input File")
            sys.stdout.flush()
            graphStatistics(G1, file.split('.')[0])
            for i in range(0, end):
                if args.wset[0] is False:
                    if count == 0:
                        weight = args.bweight
                    else:
                        weight = weight + args.increment
                        count += 1
                else:
                    weight = args.wset[count]
                    count += 1

                graphs = sample(G1, args.outFileG, args.outFileP,
                                args.iternum, weight, count)
                print('I made it after the sampling')
                sys.stdout.flush()
                graphSampleStatistics(G1, graphs[0], graphs[
                                      1], file.split('.')[0], weight)
                print('Graphed sample statistics')
                sys.stdout.flush()
            count = 0
            print(datetime.now() - startTime)
    else:
        print("Input Directory Doesn't Exist - Insert New Directory")


if __name__ == '__main__':
    main()
