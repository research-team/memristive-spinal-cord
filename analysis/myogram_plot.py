import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, argrelextrema

fs = 10000.0
lowcut = 100.0
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
        if i < 20:
            meaned.append(a)
        else:
            s = 0
            for j in range(1, 21):
                s += data[i - j]
            meaned.append(s / 20)

    return meaned


if __name__ == "__main__":
    with open("Y3.txt") as f:
        all_lines = f.readlines()

    all_lines = all_lines[400000:410000]

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

    plt.subplot(3, 1, 1)
    plt.plot(left_ankle_angle, label="ankle left")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(left_hip_angle, label="hip left")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(left_ex_ankle, label="ext ankle left")
    plt.legend()

    diff_array = np.diff(left_ex_ankle)
    diff_array = np.append(diff_array, 0)

    cross_idx = np.argwhere(np.diff(np.sign(left_ex_ankle - diff_array))).flatten()

    max_x_arr = []
    max_y_arr = []
    min_x_arr = []
    min_y_arr = []
    for i in range(len(cross_idx) - 1):
        part = left_ex_ankle[cross_idx[i]: cross_idx[i + 1]]
        print(part)
        if all(np.sign(part) == 1):
            print(np.sign(part))
            max_val = max(part)
            max_y_arr.append(max_val)
            max_x_arr.append((np.where(left_ex_ankle == max_val)[0][0]))
        # print(max_val_arr)
        # exit()
        # elif all(np.sign(part)) == -1:
        else:
            print(np.sign(part))
            min_val = min(part)
            min_y_arr.append(min_val)
            min_x_arr.append((np.where(left_ex_ankle == min_val)[0][0]))

    # for num, data in enumerate(left_ankle_angle):
    # 	if num % 100 == 0:
    # 		plt.axvline(x=num, color="red")
    plt.plot(max_x_arr, max_y_arr, color="red")
    plt.plot(min_x_arr, min_y_arr, color="red")
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
