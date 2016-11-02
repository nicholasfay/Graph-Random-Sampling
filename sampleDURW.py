from __future__ import print_function
from random import choice
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy.random as rnd
import numpy as numpy
import argparse
import sys

#This function implements the next node probability distribution calculations
#It takes the graph and the current node, then calculates a distribution for all
#current edges that are within the undirected graph
def generateProbArray(Gu,v, selected, edgelist):
	#print(seconds)
	ret = []
	sum1=0
	for edge in edgelist:
		#print(edge)
		prob = edge[2]['weight']
		#prob = W(v,node, Gu)
		sum1 += prob
		ret.append(prob)
	#normalize the distribution to allow for probability to equal to 1
	return [x/float(sum1) for x in ret]

def pickNextNode(Gu, v, selected, edgelist, seconds):
	#Picking next node to sample using calculated distribution from
	#generateProbArray above
	#print(edgelist)
	prob = generateProbArray(Gu, v, selected, edgelist)
	#print(edgelist)
	#print(prob)
	picked_node = rnd.choice(seconds, p=prob)
	return picked_node

def sample(inFile, outFileG, outFileP, iternum, weight, count, debug):
	#Creates a Directed Graph to load the data into
	G1 = nx.DiGraph()
	#Needed since added edges also add the end node to the graph
	#and an indicator to only sampled nodes need to be kept
	selected = []
	
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

	#Not possible to sample more nodes than given
	if iternum > len(G1.nodes()):
		sys.exit("Trying to sample more nodes than are given")
	#Directed Sampling Algorithm Begin
	#Assume that converging Gi (graph at ith step) to infinity gets you Gu (your final undirected sampled graph)

	#Create the undirected graph, Gu
	Gu = nx.Graph()
	virtNode = 'virtNode'

	#Use choice to select a random node from the set of all nodes in the graph
	v = choice(G1.nodes())
	selected.append(v)
	#f = open('./testfile', 'w')
	#iternum is how many sampling steps do you want to make
	#this starts at Gi and goes to G-inf
	for i in range(0, iternum):
		#Get the set of out-edges and add them to our undirected graph
		#Add an edge from v to the virtual node with weight = w
		#and between all other edges weight = 1
		Nv = G1.out_edges([v], data = True)
		Nv = [x for x in Nv if not x[1] in selected]
		Gu.add_edges_from(Nv)
		Gu.add_edge(v,virtNode, weight = weight)
		#print(Gu.edges())


		#Randomly choose a number from a uniform distribution across the
		#edges indices
		#Picking next node to sample using uniform distribution above across 
		#all edges
		edgelist = Gu.edges(v, data=True)
		#print(edgelist)
		seconds = [x[1] for x in edgelist]
		picked_node = pickNextNode(Gu, v, selected, edgelist, seconds)

		#Check that this node hasn't been selected before
		while(picked_node in selected or picked_node == 'virtNode'):
			picked_node = pickNextNode(Gu, v, selected, edgelist, seconds)
			#print(picked_node)
			if(picked_node == 'virtNode'):
				#print(seconds)
				seconds = [x[1] for x in edgelist]
				#TODO: random uniform choice from all nodes or only nodes in Gu?
				picked_node = rnd.choice(G1.nodes())
		v = picked_node
		#NOTE: For some reason int(v) needed to be used because picked_node was sometimes a string
		#and sometimes an integer which was breaking the out_degree function
		selected.append(int(v))
		#print(selected)

	#Exports the original graph (G1) and sampled graph (Gu) to a serialized GPickle
	if(outFileG):
		splitN = os.path.basename(inFile).split('.')[0]
		outFile1 = 'DURW/outputSample/{}-{}-out.gpickle'.format(splitN,str(count))
		outFile2 = 'DURW/outputSample/{}-{}-outSample.gpickle'.format(splitN,str(count))
		nx.write_gpickle(G1, outFile1)
		nx.write_gpickle(Gu, outFile2)
	if(outFileP):
		#Creates a custom labels dictionary for specific nodes
		custom_labels = {}
		custom_labels['virtNode'] = 'VIRTUAL NODE'

		#Creates a custom node size dictionary for specific nodes
		custom_node_sizes = {}
		custom_node_sizes['virtNode'] = 100
		#Draws the graph and saves it to the name specified in savefig()
		nx.draw(Gu, labels=custom_labels, node_list = Gu.nodes(), node_size = custom_node_sizes.values())
		splitN2 = os.path.basename(inFile).split('.')[0]
		plt.savefig('DURW/outputSample/{}-{}-sampled.png'.format(splitN2,str(count)))
		#plt.show() #Uncomment first part if user wants to automatically see graph 
		plt.clf()
	graphs = []
	graphs.append(G1)
	graphs.append(Gu)
	graphs.append(selected)
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