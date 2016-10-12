from __future__ import print_function
from random import choice
import networkx as nx
import matplotlib.pyplot as plt
import numpy.random as rnd
import argparse


def main():
	#Argument parsing for various options
	parser = argparse.ArgumentParser(description="Generate Sampled Graph")
	parser.add_argument('-f', '--inFile', type=str, required=True,
											help='Input graph file in form .gpickle (.txt support will be added)')
	args = parser.parse_args()

	G = nx.read_gpickle(args.inFile)

	outName = 'stats-' + args.inFile.split('.')[0] + '.txt'
	outFile = open(outName, 'w')

	print('Statistics for input graph', args.inFile, file=outFile)
	'''G2 = nx.read_gpickle('out-IncomingGraph.gpickle')

	#Creates a custom labels dictionary for specific nodes
	custom_labels = {}
	custom_labels['virtNode'] = 'VIRTUAL NODE'

	#Creates a custom node size dictionary for specific nodes
	custom_node_sizes = {}
	custom_node_sizes['virtNode'] = 100

	#Draws the graph and saves it to the name specified in savefig() then displays it
	nx.draw(G, labels=custom_labels, node_list = G.nodes(), node_size = custom_node_sizes.values())

	plt.savefig("G.png") 
	plt.show()'''

	#Test code to make sure gpickle was being loaded properly
	'''f = open('./testfile2', 'w')
	out = G2.out_degree(G.nodes())
	sortedList = [(k, out[k]) for k in sorted(out)]
	print(sortedList, file=f)'''

if __name__ == '__main__':
	main()