from __future__ import print_function
from sampleDURW import sample
import argparse
import os

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
	parser.add_argument('-bw', '--bweight', type=int, default=10,
											help='Weight of edges to determine frequency of random jumps (1:less -->  inf:more) - default is 10')
	parser.add_argument('-incr', '--increment', type=int, default=100,
											help='Amount to increment w by for each iteration of sampling')
	parser.add_argument('-amt', '--amount', type=int, default=3,
											help='Amount of iterations for different samplings based on increment amount')
	args = parser.parse_args()

	#loop indir
	#perform sampleDURW on each graph, for number of times specified by amount, starting at bw, incrementing by increment each time
	#number of sampling rounds for underlying sampling function
	path = args.inDir
	graphs = []
	if(os.path.isdir(path)):
		if not os.path.isabs(path):
			path = os.path.abspath(path)
		count = 0
		for file in os.listdir(path):	
			for i in range(0,args.amount):
				if count == 0:
					weight = args.bweight
					count += 1
				else:
					weight = weight + args.increment
					count +=1
				graphs.append(sample(file, args.outFileG, args.outFileP, args.iternum, weight, count))
	else:
		print("Input Directory Doesn't Exist - Insert New Directory")
	#print(len(sum(graphs, [])))

if __name__ == '__main__':
	main()