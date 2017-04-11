import numpy as np

with open("WikiTalk.txt") as f:
    seed = set()
    for line in f:
        split = line.split()
        seed.add(split[0])
        seed.add(split[1])
arrayseed = list(seed)
output = np.random.choice(arrayseed, 200000)

outfile = open("seedout.txt", "w")
for item in output:
    print(item, file=outfile)
