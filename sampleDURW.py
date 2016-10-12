from __future__ import print_function
from random import choice
import networkx as nx
import matplotlib.pyplot as plt
import numpy.random as rnd
import argparse


def sample(inFile, outFileG, outFileP, iternum, weight, count):

	#Creates a Directed Graph to load the data into
	G1 = nx.DiGraph()
	print(weight)

	#Parses each line and adds an edge. 
	#Since the underlying structure is a dictionary
	#there are no duplicates created
	#Creates the initial directed graph, Gd, from the given data.
	with open(inFile) as f:
		for line in f:
			#Edge format is <nodeID> <nodeID>
			edge = line.split()
			node1 = int(edge[0])
			node2 = int(edge[1])
			#Add edges as they are parsed
			G1.add_edge(node1, node2, weight=1)

	#Directed Sampling Algorithm Begin

	#Create the undirected graph, Gu
	Gu = nx.Graph()
	virtNode = 'virtNode'

	#Use choice to select a random node from the set of all nodes in the graph
	v = choice(G1.nodes())
	#f = open('./testfile', 'w')
	for i in range(0, iternum):
		uniform = rnd.uniform(0,1,1000)
		alpha = choice(uniform)
		x = alpha #uniform distribution random number

		#Get the set of out-edges and add them to our undirected graph
		#Add an edge from v to the virtual node with weight = U[0,1]
		Nv = G1.out_edges([v], data = True)
		Gu.add_edges_from(Nv)
		Gu.add_edge(v,virtNode, weight = weight)
		
		#To ensure that the next node isn't the same node
		t = choice(G1.nodes())
		while(v==t):
			t = choice(G1.nodes())
		v = t

	#Creates a custom labels dictionary for specific nodes
	custom_labels = {}
	custom_labels['virtNode'] = 'VIRTUAL NODE'

	#Creates a custom node size dictionary for specific nodes
	custom_node_sizes = {}
	custom_node_sizes['virtNode'] = 100

	#Code to test that the graph being passed out is the same that the stats is getting
	'''out = G1.out_degree(Gu.nodes())
	sortedList = [(k, out[k]) for k in sorted(out)]
	print(sortedList, file=f)'''

	#Draws the graph and saves it to the name specified in savefig() then displays it
	nx.draw(Gu, labels=custom_labels, node_list = Gu.nodes(), node_size = custom_node_sizes.values())

	#Exports the original graph (G1) and sampled graph (Gu) to a serialized GPickle
	if(outFileG):
		splitN = inFile.split('.')[0]
		outFile1 = 'DURW/outputSample/' + splitN + '-' + str(count) + '-out.gpickle'
		outFile2 = 'DURW/outputSample/' + splitN + '-' + str(count) + '-outSample.gpickle'
		nx.write_gpickle(G1, outFile1)
		nx.write_gpickle(Gu, outFile2)
	if(outFileP):
		splitN2 = inFile.split('.')[0]
		plt.savefig('DURW/outputSample/' + splitN2 + '-' + str(count) + '-sampled.png')
		plt.clf()
	#plt.show()
	graphs = []
	graphs.append(G1)
	graphs.append(Gu)
	return graphs

if __name__ == '__main__':
	#Argument parsing for various options
	parser = argparse.ArgumentParser(description="Generate Sampled Graph")
	parser.add_argument('-f', '--inFile', type=str, required=True,
											help='Input graph file in form <edge>, <edge> describing graph structure')
	parser.add_argument('-oG', '--outFileG', type=bool, default=False,
											help='Whether or not the output gpickle files should be generated.')
	parser.add_argument('-oP', '--outFileP', type=bool, default=False,
											help='Whether or not the output sampled png should be generated.')
	parser.add_argument('-it', '--iternum', type=int, default=20, 
											help='Number of sampling rounds - Default is 20')
	parser.add_argument('-w', '--weight', type=int, default=10,
											help='Weight of edges to determine frequency of random jumps (1:less -->  inf:more)')
	args = parser.parse_args()
	sample(args.inFile, args.outFileG, args.outFileP, args.iternum, args.weight)