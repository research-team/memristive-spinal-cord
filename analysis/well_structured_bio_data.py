from analysis.real_data_slices import read_data, slice_myogram
from matplotlib import pylab as plt
path = "control rat_threshold_bipedal_40Hz_LTA.mat"
data = read_data('../bio-data/{}'.format(path))
processed_data = slice_myogram(data)
real_data = processed_data[0]
slices_begin_time = processed_data[1]
# for slice_begin_time in slices_begin_time:
#     plt.axvline(x=slice_begin_time / 0.25, linestyle='--', color='gray')
plt.plot(real_data)
offset = len(real_data) % 1000
num_of_slices = len(real_data) // 1000
slices_of_real_data = []
for j in range(num_of_slices):    # 104
    slice_of_real_data = []
    for i in range(offset, offset + 1000):
        slice_of_real_data.append(real_data[i])
    slices_of_real_data.append(slice_of_real_data)
    offset += 1000
slices_of_real_data.append(slice_of_real_data)
ticks = []
labels = []
for i in range(0, len(real_data), 5000):
    ticks.append(i)
    labels.append(i * 0.025)
yticks = []
for index, sl in enumerate(range(len(slices_of_real_data))):
    offset = index
    yticks.append(slices_of_real_data[sl][0] + offset)
    # plt.plot([data + offset for data in slices_of_real_data[sl]], color='gray')
ticks_for_slices = []
labels_for_slices = []
for i in range(0, len(slices_of_real_data[0]), 50):
    ticks_for_slices.append(i)
    labels_for_slices.append(round((i * 0.025), 3))
ticks = []
labels = []
for i in range(0, len(real_data), 6000):
    ticks.append(i)
    labels.append(round((i * 0.025), 3))
plt.xticks(ticks, labels)
plt.show()