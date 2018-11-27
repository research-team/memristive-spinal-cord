from analysis.max_min_values_neuron_nest import calc_max_min
from analysis.real_data_slices import *


def init():
    raw_real_data = read_data('../bio-data//SCI_Rat-1_11-22-2016_RMG_40Hz_one_step.mat')

    myogram_data = slice_myogram(raw_real_data)
    slices_begin_time = myogram_data[1]
    slices_begin_time = [int(t / real_data_step) for t in slices_begin_time]
    volt_data = myogram_data[0]

    data = calc_max_min(slices_begin_time, volt_data, data_step=0.25)
    data_with_deleted_ees = remove_ees_from_min_max(data[0], data[1], data[2], data[3])

    durations = calc_durations(data_with_deleted_ees[0], data_with_deleted_ees[2])

    dels = delays(data_with_deleted_ees[0], data_with_deleted_ees[2])


def remove_ees_from_min_max(slices_max_time, slices_max_value, slices_min_time, slices_min_value):
    """

	Parameters
	----------
	slices_max_time: dict
		key is index of slice, value is the list of max times in slice
	slices_max_value: dict
		key is index of slice, value is the list of max values in slice
	slices_min_time: dict
		key is index of slice, value is the list of min times in slice
	slices_min_value: dict
		key is index of slice, value is the list of min values in slice

	Returns
	-------
	slices_max_time: dict
		key is index of slice, value is the list of max times in slice without the time of EES
	slices_max_value: dict
		key is index of slice, value is the list of max values in slice without the value of EES
	slices_min_time: dict
		key is index of slice, value is the list of min times in slice without the time of EES
	slices_min_value: dict
		key is index of slice, value is the list of min values in slice without the value of EES

	"""
    for slice_index in range(1, len(slices_max_time)):
        del slices_max_time[slice_index][0]
        del slices_max_value[slice_index][0]
        del slices_min_time[slice_index][0]
        del slices_min_value[slice_index][0]
    return slices_max_time, slices_max_value, slices_min_time, slices_min_value


def calc_durations_nest(slices_max_time, slices_min_time):
    """
	Parameters
	----------
	slices_max_time: dict
		key is index of slice, value is the list of max times in slice without the time of EES
	slices_min_time: dict
		key is index of slice, value is the list of min times in slice without the time of EES
	Returns
	-------
	duration_maxes: list
		durations (difference between the last and the first time of max peak) in each slice
	duration_mins: list
		durations (difference between the last and the first time of min peak) in each slice
	"""
    duration_maxes = []
    duration_mins = []
    for index in slices_max_time.values():
        duration_maxes.append(round(index[-1] - index[0], 3))
    for index in slices_min_time.values():
        duration_mins.append(round(index[-1] - index[0], 3))
    return duration_maxes, duration_mins


def calc_durations_neuron(slices_max_time, slices_min_time, slices_min_value):
    """

	Parameters
	----------
	slices_max_time: dict
		key is index of slice, value is the list of max times in slice without the time of EES
	slices_min_time: dict
		key is index of slice, value is the list of min times in slice without the time of EES

	Returns
	-------
	duration_maxes: list
		durations (difference between the last and the first time of max peak) in each slice
	duration_mins: list
		durations (difference between the last and the first time of min peak) in each slice
	"""
    duration_maxes = []
    duration_mins = []
    list_of_true_min_peaks = []
    tmp_list_of_true_min_peaks = []
    list_of_true_min_peak_times = []
    list_of_true_min_peak_times_tmp = []
    for index in slices_max_time.values():
        duration_maxes.append(round(index[-1] - index[0], 3))
    for index in slices_min_value.values():
        for i in range(len(index)):
            if -1.5 * 10 ** (-9) < index[i] < -8 * 10 ** (-10):
                tmp_list_of_true_min_peaks.append(i)
                # print("here")
            # print("i = ", i)
            # print("tmp_list_of_true_min_peaks.append(i) = ", tmp_list_of_true_min_peaks)
        list_of_true_min_peaks.append(tmp_list_of_true_min_peaks.copy())
        tmp_list_of_true_min_peaks.clear()
    print(list_of_true_min_peaks[26][0])
    del list_of_true_min_peaks[26][0]
    print("list_of_true_min_peaks = ", list_of_true_min_peaks)
    for sl in range(len(list_of_true_min_peaks)):
        for key in slices_min_time:
            # print("key = ", key)
            # print("sl = ", sl)
            if key == sl + 1:
                # print("len(list_of_true_min_peaks[{}]) = ".format(sl), len(list_of_true_min_peaks[sl]))
                for i in range(len(list_of_true_min_peaks[sl])):
                    # print("i = ", i)
                    list_of_true_min_peak_times_tmp.append(round(slices_min_time[key][list_of_true_min_peaks[sl][i]], 3))
                list_of_true_min_peak_times.append(list_of_true_min_peak_times_tmp.copy())
                list_of_true_min_peak_times_tmp.clear()
            # delays_mins.append(round(slices_min_time[key][list_of_true_min_peaks[sl][0]], 3))
    # duration_mins.append(round(index[-1] - index[0], 3))
    print("list_of_true_min_peak_times = ", list_of_true_min_peak_times)
    for sl in list_of_true_min_peak_times:
        # print("sl = ", sl)
        duration_mins.append(round(sl[-1] - sl[0], 3))
    return duration_maxes, duration_mins


def delays_neuron(slices_max_time, slices_min_time, slices_min_value):
    """

	Parameters
	----------
	slices_max_time: dict
		key is index of slice, value is the list of max times in slice without the time of EES
	slices_min_time: dict
		key is index of slice, value is the list of min times in slice without the time of EES


	Returns
	-------
	delays_maxes: list
		times of the first max peaks without EES in each slice
	delays_mins: list
		times of the first min peaks without EES in each slice

	"""
    delays_maxes = []
    delays_mins = []
    min_index = 0
    value_id = 0
    list_of_true_min_peaks = []
    tmp_list_of_true_min_peaks = []
    for index in slices_max_time.values():
        delays_maxes.append(round(index[0], 3))
    for index in slices_min_value.values():
        for i in range(len(index)):
            if -1.5 * 10 ** (-9) < index[i] < -8 * 10 ** (-10):
                # -1.5 * 10 ** (-9) < index[i] < -8 * 10 ** (-10) for 6 cm/s
                tmp_list_of_true_min_peaks.append(i)
            # print("i = ", i)
            # print("tmp_list_of_true_min_peaks.append(i) = ", tmp_list_of_true_min_peaks)
        list_of_true_min_peaks.append(tmp_list_of_true_min_peaks.copy())
        tmp_list_of_true_min_peaks.clear()
    del list_of_true_min_peaks[26][0]

    print("list_of_true_min_peaks_delays = ", list_of_true_min_peaks)
    for sl in range(len(list_of_true_min_peaks)):
        for key in slices_min_time:
            # print("key = ", key)
            # print("sl = ", sl)
            if key == sl + 1:
                # for slice in slices_min_time.values():
                #         print("sl = ", sl)
                # delays_mins.append(sl[0])
                # print("slices_min_time[{}] = ".format(key), slices_min_time[key])
                # print("list_of_true_min_peaks[sl][0] = ", list_of_true_min_peaks[sl][0])
                delays_mins.append(round(slices_min_time[key][list_of_true_min_peaks[sl][0]], 3))
    return delays_maxes, delays_mins


def delays_nest(slices_max_time, slices_min_time):
    delays_maxes = []
    delays_mins = []
    for index in slices_max_time.values():
        delays_maxes.append(round(index[0], 3))
    for index in slices_min_time.values():
        delays_mins.append(round(index[0], 3))
    # for value in slices_min_value.values():
    #     value_id += 1
    #     for i in range(len(value)):
    #         if value[i] < -0.5 * 10 ** (-10):
    #             min_index = i
    #             for key in slices_min_time:
    #                 if key == value_id:
    #                     delays_mins.append(slices_min_time[key][min_index])
    #                 break
    # for index in range(len(slices_min_time.values())):
    #     print("slices_min_time.values() = ", slices_min_time.values())
    #     for value_index in slices_min_value.values():
    #         print("value_index = ", value_index)
    #         for i in range(len(value_index)):
    #             if value_index[index][i] < -5 * 10 ** (-10):
    #                 min_index = i
    #                 print("min_index = ", min_index)
    #         break
    #     delays_mins.append(round(index[min_index], 3))
    return delays_maxes, delays_mins


if __name__ == '__main__':
    init()
