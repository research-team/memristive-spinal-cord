import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

path = "/home/alex/PINE64/LESHA/fast"
names = ["носок (d1)", "большой (d2)", "средний (d3)", "мизинец (d4)", "у пятки (d5)", "пятка (d6)"]
logical_order = [3, 0, 5, 4, 1, 2]

with open(path) as f:
	data = np.array(list(map(str.split, f)))

# ok for Yura's 6cm/s
# data = data[500:-150, :]
# ok for Yura's 21cm/s
# data = data[-150:, :]
# ok for Yura's 13.5cm/s
# data = data[-180:-70, :]

# check a small group (faster)
data = data[100:700, :]
# convert string time to datetime
times = [datetime.fromtimestamp(float(d)) for d in data[:, 0]]
# start from 0s 0ms
times = [(t - times[0]).total_seconds() for t in times]
# conver str to float and reorder sensors
values = np.array(data[:, 1:]).astype(float)[:, logical_order]

# list for a new data that depends on true dt
new_data = []
for d in values.T:
	row = []
	for (dpar_l, dpar_r), (tpar_l, tpar_r) in zip(zip(d, d[1:]), zip(times, times[1:])):
		# dt = 100um
		dt = int((tpar_r - tpar_l) * 100)
		row += np.linspace(dpar_l, dpar_r, dt).tolist()[:-1]
	new_data.append(row)
new_data = np.array(new_data)


ax = plt.subplot(211)
xticks = list(range(len(new_data[0])))
for d, name in zip(new_data, names):
	if name == 'носок (d1)':
		d[:] = 0
	plt.plot(xticks, d, label=name)
plt.legend()

plt.subplot(212, sharex=ax)
# recognized CV
CVs = []
prev_CV = 0
for d1, d2, d3, d4, d5, d6 in new_data.T:
	CV = 0
	"""
	if all(d < 3000 for d in [d1, d2, d3, d4]):
		CV = 0
	if d6 > 3800 and d5 > 3750:
		CV = 1
	if d1 == 4000 and d3 > d6 and d4 > d6:
		CV = 5
	"""
	if all(d < 1500 for d in [d2, d3, d4]):
		CV = 0
	elif d5 > 3500 and d5 > d3:
		CV = 1
	else:
		if all(d > 1500 for d in [d2, d3, d4]) and d6 > d4:
			CV = 5
		if d3 > d5 and d2 > d5:
			CV = 2
		if d5 < d4:
			CV = 3
	prev_CV = CV
	CVs.append(CV)

plt.plot(xticks, CVs)
plt.show()
