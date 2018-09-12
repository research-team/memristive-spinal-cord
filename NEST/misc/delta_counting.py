"""
Is not finished as independent script

"""

list_of_deltas = []

mins = []
maxs = []
means = []

for index in range(1, len(list_slices) - 1):
	# take mean of previous tests
	previous_mean = list(map(lambda elements: sum(elements) / len(elements), zip(*list_slices[:index])))
	latest_experiment = list(map(lambda elements: sum(elements) / len(elements), zip(*list_slices[:index+1])))

	delta = list(map(lambda x: abs(x[0]-x[1]), zip(latest_experiment, previous_mean)))

	list_of_deltas.append(delta)
	mins.append(min(delta))
	maxs.append(max(delta))
	means.append(sum(delta) / len(delta))

for index, value in enumerate(maxs):
	print(index, "{:.20f}".format(value))
print("\n\n")
for index, value in enumerate(means):
	print(index, "{:.20f}".format(value))