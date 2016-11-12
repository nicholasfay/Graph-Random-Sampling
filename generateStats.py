import networkx as nx
import matplotlib.pyplot as plt
import argparse
from collections import Counter

#Indicator functions
def outDegreeIndicator(G, j, node):
	if(G.out_degree(node) == j):
		return 1
	return 0

def inDegreeIndicator(G, j, node):
	if(G.in_degree(node) == j):
		return 1
	return 0

#Calculating S for the steady state probability
def calculateS(G, selected, w, n):
	ret = 0
	#Weird error that happens sometimes here, says that
	#some of the nodes in the selected nodes aren't nodes in 
	#the sampled graph
	for item in selected:
		ret += (1.0/(w + G.degree(item)))
	return ret / n

#Steady state probability of sampling a node
def piFunc(G, item, w):
	degree = G.degree(item)
	return (w + degree)

#Driver function to do in-degree and out-degree sample distribution estimators
def distributionEstimator(topFunc, G, Gu, selected, inFile, w):
	phi = {}
	n = float(len(selected))
	#print(n)
	out_degree = G.out_degree(selected)
	S = calculateS(G, selected, w, n)
	out_degree_vals = sorted(set(out_degree.values()))
	for item in out_degree_vals:
		ret = 0
		for item2 in selected:
			indicator = topFunc(G, item, item2)
			pi = piFunc(G, item2, w)
			pi = pi * S
			ret += (indicator / pi) / n
		phi[item] = ret
	return phi


#Computes out_degree distribution, in_degree distribution
#and clustering coefficient of sampled graph after DURW
def graphSampleStatistics(origG, sampledG, selected, inFile, w):
	outName = 'stats/stats-{}-sample-w{}.txt'.format(inFile, str(w))
	outFile = open(outName, 'w')
	print('Statistics for input graph sample-{}-w{}'.format(inFile,str(w)), file=outFile)

	out_degree = distributionEstimator(outDegreeIndicator, origG, sampledG, selected, inFile, w) 
	in_degree =  distributionEstimator(inDegreeIndicator, origG, sampledG, selected, inFile, w)
	plt.figure()
	plt.plot([float(x) for x in out_degree.keys()], list(out_degree.values()), 'ro-')
	plt.plot([float(x) for x in in_degree.keys()], list(in_degree.values()), 'bv-')
	#plt.yscale('log')
	#plt.xscale('log')
	plt.legend(['Out-degree', 'In-degree'])
	plt.xlabel('Degree')
	plt.ylabel('Percentage of nodes')
	title = 'In-Degree and Out-Degree Distributions for {}'.format(inFile)
	plt.title(title)
	outGraph = 'stats/{}-degree-distribution-sample.jpg'.format(inFile)
	plt.savefig(outGraph)
	plt.close()

	print('In-Degree and Out-Degree have been plotted and saved at {}'.format(outGraph), file=outFile)
	print('Clustering Coefficient:', file=outFile)
	cluster = nx.average_clustering(sampledG)
	print(cluster, file=outFile)

#Computes out_degree distribution, in_degree distribution
#and clustering coefficient of unsampled graph
def graphStatistics(G, inFile, w):
	outName = 'stats/stats-{}-original.txt'.format(inFile)
	outFile = open(outName, 'w')
	print('Statistics for input graph {}'.format(inFile), file=outFile)

	out_degree = G.out_degree()
	out_degree_vals = sorted(set(out_degree.values()))
	c = Counter(out_degree.values())
	out_degree_distr = [c[x] for x in out_degree_vals]
	n1 = float(sum(out_degree_distr))
	norm_out_degree_distr = [x/n1 for x in out_degree_distr]
	in_degree = G.in_degree()
	in_degree_vals = sorted(set(in_degree.values()))
	c2 = Counter(in_degree.values())
	in_degree_distr = [c2[x] for x in in_degree_vals]
	n2 = float(sum(in_degree_distr))
	norm_in_degree_distr = [x/n2 for x in in_degree_distr]
	#print(norm_in_degree_distr)
	plt.figure()
	plt.plot(out_degree_vals, norm_out_degree_distr, 'ro-')
	plt.plot(in_degree_vals, norm_in_degree_distr, 'bv-')
	#plt.yscale('log')
	#plt.xscale('log')
	plt.legend(['Out-degree', 'In-degree'])
	plt.xlabel('Degree')
	plt.ylabel('Percentage of nodes')
	title = 'In-Degree and Out-Degree Distributions for {}'.format(inFile)
	plt.title(title)
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