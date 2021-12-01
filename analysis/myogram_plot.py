import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, argrelextrema
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import normalize

# def normalize(arr, t_min, t_max):
# 	norm_arr = []
# 	diff = t_max - t_min
# 	diff_arr = max(arr) - min(arr)
# 	for i in arr:
# 		temp = (((i - min(arr)) * diff) / diff_arr) + t_min
# 		norm_arr.append(temp)
# 	return norm_arr


fs = 10000.0
lowcut = 200.0
highcut = 1000.0


def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y


def get_mean(data):
	meaned = []
	for i, a in enumerate(data):
		if i < 10:
			meaned.append(a)
		else:
			s = 0
			for j in range(1, 11):
				s += data[i - j]
			meaned.append(s / 10)

	return meaned


def outline(array):
	diff_array = np.diff(array)
	diff_array = np.append(diff_array, 0)

	# points of crossing of myogram and its diff
	cross_idx = np.argwhere(np.diff(np.sign(array - diff_array))).flatten()

	max_x_arr = []
	max_y_arr = []
	min_x_arr = []
	min_y_arr = []
	total_dict = {max: [max_x_arr, max_y_arr], min: [min_x_arr, min_y_arr]}
	for i in range(len(cross_idx) - 1):
		part = array[cross_idx[i]: cross_idx[i + 1]]
		if all(np.sign(part) == 1):
			max_val = max(part)
			total_dict[max][1].append(max_val)
			total_dict[max][0].append((np.where(array == max_val)[0][0]))
		else:
			min_val = min(part)
			total_dict[min][1].append(min_val)
			total_dict[min][0].append((np.where(array == min_val)[0][0]))
	return total_dict


# useless
# def adding_points(x_arr, y_arr):
#     merged_points = []
#     for i in range(x_arr[0]):
#         if i < x_arr[0]:
#             merged_points.append(y_arr[0])
#     for idx in range(len(x_arr) - 1):
#         x_val = x_arr[idx]
#         y_val = y_arr[idx]
#         next_x_val = x_arr[idx + 1]
#         next_y_val = y_arr[idx + 1]
#         merged_points.append(y_val)
#         must_have_val = next_x_val - x_val
#         max_difference_btw_y = next_y_val - y_val
#         tan = max_difference_btw_y / must_have_val
#
#         for i in range(1, must_have_val):
#             plus_val = y_val + (i * tan)
#             merged_points.append(plus_val)
#
#     return np.asarray(merged_points)


if __name__ == "__main__":
	with open("/home/b-rain/Desktop/plot_mio/8.txt") as f:
		all_lines = f.readlines()

	all_lines = all_lines[100000:140000]

	arr = []
	for line in all_lines:
		s = line.split(" ")

		res_arr = []

		for i in s:
			if i != '':
				res_arr.append(float(i))
		arr.append(res_arr)

	arr = np.array(arr)
	arr = arr.T
	left_ankle_angle_index = 4
	left_hip_angle_index_1 = 2
	left_hip_angle_index_2 = 5
	left_ex_ankle_index = 13
	left_fl_ankle_index = 12
	left_ex_hip_index = 11
	left_fl_hip_index = 10

	right_ankle_angle_index = 3
	right_hip_angle_index_1 = 0
	right_hip_angle_index_2 = 1
	right_ex_ankle_index = 9
	right_fl_ankle_index = 8
	right_ex_hip_index = 7
	right_fl_hip_index = 6

	left_ankle_angle = get_mean(arr[left_ankle_angle_index])
	left_hip_angle = get_mean(arr[left_hip_angle_index_1] - arr[left_hip_angle_index_2])
	left_ex_ankle = butter_bandpass_filter(np.array(arr[left_ex_ankle_index]), lowcut, highcut, fs)
	left_fl_ankle = butter_bandpass_filter(np.array(arr[left_fl_ankle_index]), lowcut, highcut, fs)
	left_ex_hip = butter_bandpass_filter(np.array(arr[left_ex_hip_index]), lowcut, highcut, fs)
	left_fl_hip = butter_bandpass_filter(np.array(arr[left_fl_hip_index]), lowcut, highcut, fs)

	right_ankle_angle = get_mean(- arr[right_ankle_angle_index])
	right_hip_angle = get_mean(arr[right_hip_angle_index_1] - arr[right_hip_angle_index_2])
	right_ex_ankle = butter_bandpass_filter(np.array(arr[right_ex_ankle_index]), lowcut, highcut, fs)
	right_fl_ankle = butter_bandpass_filter(np.array(arr[right_fl_ankle_index]), lowcut, highcut, fs)
	right_ex_hip = butter_bandpass_filter(np.array(arr[right_ex_hip_index]), lowcut, highcut, fs)
	right_fl_hip = butter_bandpass_filter(np.array(arr[right_fl_hip_index]), lowcut, highcut, fs)

	gaussian_angle = gaussian_filter(left_hip_angle, sigma=50)
	max_val = max(gaussian_angle)
	dispersion = 0.3
	range_ = []
	array_of_maximum = []
	index_of_maximum = []
	for i in gaussian_angle:
		if max_val + dispersion >= i > max_val - dispersion:
			range_.append(i)
		if range_ and i < (max_val-1):
			max_in_diapason = max(range_)
			array_of_maximum.append(max_in_diapason)
			index_of_maximum.append((np.where(gaussian_angle == max_in_diapason)[0][0]))
			range_ = []

	# plt.plot(gaussian_angle, linestyle=" ", marker="o")
	# plt.plot(index_of_maximum, array_of_maximum, linestyle=" ", marker="o", color = "red")

	plt.subplot(3, 1, 1)
	plt.plot(gaussian_angle, linestyle=" ", marker="o")
	plt.plot(index_of_maximum, array_of_maximum, linestyle=" ", marker="o", color="red")

	plt.legend()

	total_dict_myo_outline = outline(left_ex_ankle)
	max_arr_myo_outline = total_dict_myo_outline[max]
	min_arr_myo_outline = total_dict_myo_outline[min]

	plt.subplot(3, 1, 2)
	# plt.plot(left_ex_ankle)
	plt.plot(max_arr_myo_outline[0], gaussian_filter(max_arr_myo_outline[1], sigma=2, mode="wrap"), color="black")
	plt.plot(min_arr_myo_outline[0], gaussian_filter(min_arr_myo_outline[1], sigma=2, mode="wrap"), color="black")

	total_dict_myo_outline2 = outline(left_fl_ankle)
	max_arr_myo_outline2 = total_dict_myo_outline2[max]
	min_arr_myo_outline2 = total_dict_myo_outline2[min]

	plt.subplot(3, 1, 3)
	# plt.plot(left_fl_ankle)
	plt.plot(max_arr_myo_outline2[0], gaussian_filter(max_arr_myo_outline2[1], sigma=2, mode="wrap"), color="black")
	plt.plot(min_arr_myo_outline2[0], gaussian_filter(min_arr_myo_outline2[1], sigma=2, mode="wrap"), color="black")

	plt.show()

	# plt.subplot(12, 1, 1)
	# p = plt.plot(left_ankle_angle, label="ankle left")
	# plt.legend()
	#
	# plt.subplot(12, 1, 2)
	# p = plt.plot(left_hip_angle, label="hip left")
	# plt.legend()
	#
	# plt.subplot(12, 1, 3)
	# plt.plot(left_ex_ankle, label="ext ankle left")
	# plt.legend()
	#
	# plt.subplot(12, 1, 4)
	# plt.plot(left_fl_ankle, label="flex ankle left")
	# plt.legend()
	#
	# plt.subplot(12, 1, 5)
	# plt.plot(left_ex_hip, label="ext hip left")
	# plt.legend()
	#
	# plt.subplot(12, 1, 6)
	# plt.plot(left_fl_hip, label="flex hip left")
	# plt.legend()
	#
	# plt.subplot(12, 1, 7)
	# p = plt.plot(right_ankle_angle, label="ankle right")
	# plt.legend()
	#
	# plt.subplot(12, 1, 8)
	# p = plt.plot(right_hip_angle, label="hip right")
	# plt.legend()
	#
	# plt.subplot(12, 1, 9)
	# plt.plot(right_ex_ankle, label="ext ankle right")
	# plt.legend()
	#
	# plt.subplot(12, 1, 10)
	# plt.plot(right_fl_ankle, label="flex ankle right")
	# plt.legend()
	#
	# plt.subplot(12, 1, 11)
	# plt.plot(right_ex_hip, label="ext hip right")
	# plt.legend()
	#
	# plt.subplot(12, 1, 12)
	# plt.plot(right_fl_hip, label="flex hip right")
	# plt.legend()
	#
	# plt.savefig("res.png")
	#
	# plt.show()
