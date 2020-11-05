import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

path = "/home/alex/PINE64/LESHA/slow"
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

CVs = []
prev_CV = 0
for d1, d2, d3, d4, d5, d6 in new_data.T:
	CV = 0
	if all(d < 1000 for d in [d2, d3, d4]) and d6 < 3800:
		CV = 0
	if d5 > 3500 and d4 > d3 or d2 > d4 and d6 > 3700:
		CV = 1
	if d5 > d4 and all(d > 2300 for d in [d2, d3]):
		CV = 2
	if all(d > 1500 for d in [d2, d3, d4]) and d6 > d4 and d4 > d2:
		CV = 5
	if d5 < d4:
		CV = 3
	CVs.append(CV)


def plot_cv_ticks(plot, ticks):
	for tick in ticks:
		plot.axvline(x=tick, color='gray', alpha=0.5)

cv_ticks = [i for i, (cv1, cv2) in enumerate(zip(CVs, CVs[1:])) if cv1 != cv2]
cv_ticks += [i for i, (cv1, cv2) in enumerate(zip(CVs[1:], CVs)) if cv1 != cv2]

ax = plt.subplot(211)
xticks = list(range(len(new_data[0])))
for d, name in zip(new_data, names):
	if name == 'носок (d1)':
		d[:] = 0
	plt.plot(xticks, d, label=name)
plot_cv_ticks(plt, cv_ticks)
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)

plt.subplot(212, sharex=ax)
# recognized CV
plt.plot(xticks, CVs)
# plot_cv_ticks(plt, cv_ticks)

plt.show()
