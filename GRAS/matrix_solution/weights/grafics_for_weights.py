import matplotlib.pyplot as plt
import csv

n = 0
a = 33000

file = open('weights.dat', "r")
rdr = csv.reader(file, delimiter = ' ')

time = []
weight = []

weight = file.read().split(' ')

for i in range(a):
	time.append(n)
	n += 0.025
	weight[i] = float(weight[i])

del(weight[a])

fig = plt.figure()

plt.title('100 synapse')

plt.xlabel('time (ms)')
plt.ylabel('weight')

plt.plot(time, weight)

plt.savefig('100.png')