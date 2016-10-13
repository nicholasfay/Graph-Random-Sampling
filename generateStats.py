from __future__ import print_function
import networkx as nx
import matplotlib.pyplot as plt
import argparse

#TODO: Create outdegree distribution estimator based on paper
def sampleOutDegree(G):
	return G

#Computes out_degree distribution, in_degree distribution
#and clustering coefficient of sampled graph after DURW
def graphSampleStatistics(G, inFile, w):
	outName = 'stats/stats-{}-sample-w{}.txt'.format(inFile, str(w))
	outFile = open(outName, 'w')
	print('Statistics for input graph sample-{}-w{}'.format(inFile,str(w)), file=outFile)

	nodes = G.nodes(data=True)
	#for u in nodes:
		#print(u, file=outFile)
	#print(beta.rvs(2.31,0.627,size=1000))

	#out_degree = 
	#in_degree = 
	print('Clustering Coefficient:', file=outFile)
	cluster = nx.average_clustering(G)
	print(cluster, file=outFile)

#Computes out_degree distribution, in_degree distribution
#and clustering coefficient of unsampled graph
def graphStatistics(G, inFile, w):
	outName = 'stats/stats-{}-original.txt'.format(inFile)
	outFile = open(outName, 'w')
	print('Statistics for input graph {}'.format(inFile), file=outFile)

	out_degree = G.out_degree()
	out_degree_vals = sorted(set(out_degree.values()))
	out_degree_distr = [out_degree.values().count(x) for x in out_degree_vals]
	in_degree = G.in_degree()
	in_degree_vals = sorted(set(in_degree.values()))
	in_degree_distr = [in_degree.values().count(x) for x in in_degree_vals]
	plt.figure()
	plt.loglog(out_degree_vals,out_degree_distr, 'ro-')
	plt.loglog(in_degree_vals, in_degree_distr, 'bv-')
	plt.legend(['Out-degree', 'In-degree'])
	plt.xlabel('Degree')
	plt.ylabel('Number of nodes')
	plt.title('Test')
	outGraph = 'stats/{}-degree-distribution.jpg'.format(inFile)
	plt.savefig(outGraph)
	plt.close()

	print('In-Degree and Out-Degree have been plotted and saved at {}'.format(outGraph), file=outFile)
	#TODO: Is there error introduced from the undirected conversion? If so, how can we account
	#for that or how do we fix it
	cluster = nx.average_clustering(G.to_undirected())
	print('Clustering Coefficient:', file=outFile)
	print(cluster, file=outFile)

if __name__ == '__main__':
	#Argument parsing for various options
	parser = argparse.ArgumentParser(description="Generate Sampled Graph")
	parser.add_argument('-f', '--inFile', type=str, required=True,
											help='Input graph file in form .gpickle (.txt support will be added)')
	parser.add_argument('-s', '--sample', type=bool, required=True,
											help='True if Sampled Graph, False if full Graph')
	args = parser.parse_args()

	G = nx.read_gpickle(args.inFile)
	if(sample):
		graphSampleStatistics(G, args.inFile, 1)
	else:
		graphStatistics(G, args.inFile, 1)