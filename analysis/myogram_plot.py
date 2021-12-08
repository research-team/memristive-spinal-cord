import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
from scipy.ndimage import gaussian_filter
import warnings
import matplotlib.cbook


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

	ceiling_arr_myo_outline = gaussian_filter(adding_points(total_dict[max][0], total_dict[max][1]), sigma=10)
	floor_arr_myo_outline = gaussian_filter(adding_points(total_dict[min][0], total_dict[min][1]), sigma=10)

	return [ceiling_arr_myo_outline, floor_arr_myo_outline]


def adding_points(x_arr, y_arr):
	merged_points = []
	for i in range(x_arr[0]):
		if i < x_arr[0]:
			merged_points.append(y_arr[0])
	for idx in range(len(x_arr) - 1):
		x_val = x_arr[idx]
		y_val = y_arr[idx]
		next_x_val = x_arr[idx + 1]
		next_y_val = y_arr[idx + 1]
		merged_points.append(y_val)
		must_have_val = next_x_val - x_val
		max_difference_btw_y = next_y_val - y_val
		tan = max_difference_btw_y / must_have_val

		for i in range(1, must_have_val):
			plus_val = y_val + (i * tan)
			merged_points.append(plus_val)

	return np.asarray(merged_points)


def read(path, start, end):
	with open(path) as f:
		all_lines = f.readlines()

	all_lines = all_lines[start:end]

	names_dict = {"right_hip_angle_index_1": [], "right_hip_angle_index_2": [], "left_hip_angle_index_1": [],
	              "right_ankle_angle_index": [],
	              "left_ankle_angle_index": [], "left_hip_angle_index_2": [], "right_fl_hip_index": [],
	              "right_ex_hip_index": [],
	              "right_fl_ankle_index": [], "right_ex_ankle_index": [], "left_fl_hip_index": [],
	              "left_ex_hip_index": [],
	              "left_fl_ankle_index": [], "left_ex_ankle_index": []}
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

	right_hip_angle = []
	left_hip_angle = []
	for indx, name in enumerate(names_dict.keys()):
		if indx == 3 or indx == 4:
			names_dict[name] = np.array(get_mean(arr[indx]))
		if indx == 0:
			right_hip_angle = gaussian_filter(get_mean(arr[0] - arr[1]), sigma=50)
		if indx == 2:
			left_hip_angle = gaussian_filter(get_mean(arr[2] - arr[5]), sigma=50)
		if indx > 5:
			names_dict[name].append(gaussian_filter(butter_bandpass_filter(arr[indx], lowcut, highcut, fs), sigma=10))

	names_dict.update({"right_hip_angle": right_hip_angle})
	names_dict.update({"left_hip_angle": left_hip_angle})

	return names_dict


def peak_search(angle):
	max_val = max(angle)
	sign = np.sign(max_val)
	percent_for_dispersion = 0.25
	percent_for_limiting = 0.75
	dispersion_for_max_peak = (max_val - min(angle)) * percent_for_dispersion
	range_limiting = (max_val - min(angle)) * percent_for_limiting
	range_of_peaks = []
	array_of_maximum = []
	index_of_maximum = []
	for i in angle:
		if sign * (abs(max_val) + dispersion_for_max_peak) <= i < sign * (abs(max_val) - dispersion_for_max_peak):
			range_of_peaks.append(i)
		if range_of_peaks and i < (max_val - range_limiting):
			max_in_range = max(range_of_peaks)
			array_of_maximum.append(max_in_range)
			index_of_maximum.append((np.where(angle == max_in_range)[0][0]))
			range_of_peaks = []

	return [index_of_maximum, array_of_maximum]


def plot(names_dict, index_of_maximum, first_idx, last_idx, angle_idx, peka_name):
	if first_idx == 5:
		leg_name = "right"
	else:
		leg_name = "left"

	warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
	step_count = 0
	for num, k in enumerate(names_dict.keys()):
		if first_idx < num < last_idx or num == angle_idx:  # special indexes for legs
			for i in range(len(index_of_maximum) - 1):
				start_step = index_of_maximum[i]
				end_step = index_of_maximum[i + 1]
				step_count += 1
				if num == angle_idx:
					plt.subplot(5, 1, 1)
					plt.title(f"{leg_name} \n hip angle, ankle ex, ankle fl, hip ex, hip fl")
					plt.plot(names_dict[k][start_step:end_step], color="teal")
				else:
					plt.subplot(5, 1, (first_idx + 6) - num)
					ceiling_arr_myo_outline, floor_arr_myo_outline = outline(names_dict[k][0])
					plt.plot(ceiling_arr_myo_outline[start_step:end_step], color="cadetblue")
					plt.plot(floor_arr_myo_outline[start_step:end_step], color="cadetblue")

			print(step_count, leg_name)
			step_count = 0
	plt.savefig(f"/home/{peka_name}/Desktop/plot_mio/{leg_name}_{filename}_{number}.png")
	plt.show()
	plt.close()


if __name__ == "__main__":
	fs = 10000.0
	lowcut = 200.0
	highcut = 1000.0

	peka_name = "ann"
	# peka_name = "b-rain"
	path = f"/home/{peka_name}/Desktop/plot_mio/"
	filename = "8.txt"
	start = 480000
	end = 620000
	number = round(end / start, 2)
	names_dict = read(path + filename, start, end)

	index_of_maximum_right, array_of_maximum_right = peak_search(names_dict["right_hip_angle"])
	index_of_maximum_left, array_of_maximum_left = peak_search(names_dict["left_hip_angle"])

	plot(names_dict=names_dict, index_of_maximum=index_of_maximum_right, first_idx=5, last_idx=10, angle_idx=14,
	     peka_name=peka_name)
	plot(names_dict=names_dict, index_of_maximum=index_of_maximum_left, first_idx=9, last_idx=14, angle_idx=15,
	     peka_name=peka_name)
