from __future__ import print_function
from sampleDURW import sample
from generateStats import graphSampleStatistics,graphStatistics
import argparse
import os
from datetime import datetime
startTime = datetime.now()

def main():
	#Argument parsing for various options
	parser = argparse.ArgumentParser(description="Main Sampling and Statistics Driver Program - Supply relative or absolute path where the Graph(s) reside")
	parser.add_argument('-d', '--inDir', type=str, required=True,
											help='Name of directory where input graphs reside')
	parser.add_argument('-oG', '--outFileG', type=bool, default=False,
											help='Whether or not output gpickles of graphs are wanted (May be very taxing with huge graphs)')
	parser.add_argument('-oP', '--outFileP', type=bool, default=False,
											help='Whether or not output images of graphs are wanted (May be very taxing with huge graphs)')
	parser.add_argument('-it', '--iternum', type=int, default=20, 
											help='Number of sampling rounds - Default is 20')
	parser.add_argument('-bw', '--bweight', type=int, default=1,
											help='Weight of edges to determine frequency of random jumps (1:less -->  inf:more, w>0) - default is 10')
	parser.add_argument('-incr', '--increment', type=int, default=1,
											help='Amount to increment w by for each iteration of sampling')
	parser.add_argument('-amt', '--amount', type=int, default=3,
											help='Amount of iterations for different samplings based on increment amount')
	args = parser.parse_args()

	debug = open('debug.txt', 'w')

	#loop indir
	#perform sampleDURW on each graph, for number of times specified by amount, starting at bw, incrementing by increment each time
	#number of sampling rounds for underlying sampling function
	path = args.inDir
	if(os.path.isdir(path)):
		if not os.path.isabs(path):
			path = os.path.abspath(path)
		count = 0
		for file in os.listdir(path):	
			for i in range(0,args.amount):
				if count == 0:
					weight = args.bweight
				else:
					weight = weight + args.increment
				inFile = os.path.dirname(os.path.abspath(__file__)) + "\\testGraphs\\" + file
				#graphs.append(sample(file, args.outFileG, args.outFileP, args.iternum, weight, count))
				graphs = sample(inFile, args.outFileG, args.outFileP, args.iternum, weight, count, debug)
				if count == 0:
					graphStatistics(graphs[0], file.split('.')[0], weight)
				count += 1
				graphSampleStatistics(graphs[0], graphs[1], graphs[2], file.split('.')[0], weight)
			count = 0
			print(datetime.now() - startTime)
	else:
		print("Input Directory Doesn't Exist - Insert New Directory")

if __name__ == '__main__':
	main()

	#Degree-proportional jumps
	#Performs a jump (each iteration) to a uniformly chosen node(random.choise(uniform))
	#This happens with probability w/(w+deg(v)) where deg(v) is the out
	#degree of node v in Gu
	#Once a node is visited at the i-th step it has no additional edges will be added to that node
	#this means that if end node of an edge is one of the nodes in a dictionary(dictionary.find())
	#then do not add to that one.