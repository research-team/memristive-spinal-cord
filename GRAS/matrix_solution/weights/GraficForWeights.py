import matplotlib.pyplot as plt
import csv

n = 0
a = 33000

file = open('0.dat', "r")
rdr = csv.reader(file, delimiter = ' ')

time = []
weight = []

weight = file.read().split(' ')

for i in range(a):
	time.append(n)
	n += 0.025
	weight[i] = float(weight[i])

del(weight[a])

# weight = weight[100:a]

# time = time[100:a]

fig = plt.figure()

plt.title('0 synapse')

plt.xlabel('time (ms)')
plt.ylabel('weight')

plt.plot(time, weight)

plt.savefig('01.png')