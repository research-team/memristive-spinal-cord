from matplotlib import pyplot as plt
import os
import json
import numpy as np

from python_scripts.ampls import peaks
from python_scripts.genetic_algotithm import Individual

# sudo apt-get install sshpass

password = '"there is should be password"'

path_to_save1 = "/Desktop/ga_v9/logs"

path_to_file_with_bests_pvalue = "/GRAS/multi_gpu_test/files/bests_pvalue.dat"
path_to_file_history = "/GRAS/multi_gpu_test/files/history.dat"
path_to_file_log_of_bests = "/GRAS/multi_gpu_test/files/log_of_bests.dat"


def get_file(path_to_file=path_to_file_with_bests_pvalue, path_to_save=path_to_save1):
    os.system(f'sshpass -p {password} scp openlab@10.160.173.2:~{path_to_file} ~{path_to_save}')


def draw_plot(download=True):
    plt.close()

    if download:
        get_file()

    with open("../logs/bests_pvalue.dat") as file:
        a = file.readlines()

    arr = []

    for i in range(len(a)):
        arr.append(json.loads(a[i], object_hook=Individual.decode))

    print(f"Len = {len(arr)}")

    array_with_ampl_pvalue = []
    array_with_time_pvalue = []
    array_with_2d_pvalue = []
    array_with_dt_pvalue = []
    ff = []

    for i in arr:
        array_with_ampl_pvalue.append(i.pvalue_amplitude)
        array_with_time_pvalue.append(i.pvalue_times)
        array_with_2d_pvalue.append(i.pvalue)
        array_with_dt_pvalue.append(i.pvalue_dt)
        ff.append(i.pvalue_amplitude * i.pvalue * i.pvalue_dt)

    print(f"Max times p-value {max(array_with_time_pvalue)} "
          f"in population {array_with_time_pvalue.index(max(array_with_time_pvalue)) + 1}")
    print(f"Max ampls p-value {max(array_with_ampl_pvalue)} "
          f"in population {array_with_ampl_pvalue.index(max(array_with_ampl_pvalue)) + 1}")
    print(f"Max 2d p-value {max(array_with_2d_pvalue)} "
          f"in population {array_with_2d_pvalue.index(max(array_with_2d_pvalue)) + 1}")

    print(f"Max dt p-value {max(array_with_dt_pvalue)} "
          f"in population {array_with_dt_pvalue.index(max(array_with_dt_pvalue)) + 1}")

    peakD2d, peakI2d = peaks(array_with_2d_pvalue)
    peakDampl, peakIampl = peaks(array_with_ampl_pvalue)
    peakDtime, peakItime = peaks(array_with_time_pvalue)
    peakDff, peakIff = peaks(ff)

    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(8, 8)
    ax[0, 0].plot(array_with_ampl_pvalue)
    ax[0, 1].plot(array_with_time_pvalue)
    ax[1, 0].plot(array_with_2d_pvalue)
    ax[1, 1].plot(ff)

    ax[0, 0].set(xlabel='population', ylabel='p-value ampls')
    ax[0, 1].set(xlabel='population', ylabel='p-value times')
    ax[1, 0].set(xlabel='population', ylabel='p-value 2d')
    ax[1, 1].set(xlabel='population', ylabel='fitness function')

    for i in peakDampl:
        if i:
            ax[0, 0].scatter(x=max(i)[1], y=max(i)[0], color="r")

    for i in peakIampl:
        if i:
            ax[0, 0].scatter(x=min(i)[1], y=min(i)[0], color="b")

    for i in peakDtime:
        if i:
            ax[0, 1].scatter(x=max(i)[1], y=max(i)[0], color="r")

    for i in peakItime:
        if i:
            ax[0, 1].scatter(x=min(i)[1], y=min(i)[0], color="b")

    for i in peakD2d:
        if i:
            ax[1, 0].scatter(x=max(i)[1], y=max(i)[0], color="r")

    for i in peakI2d:
        if i:
            ax[1, 0].scatter(x=min(i)[1], y=min(i)[0], color="b")

    for i in peakDff:
        if i:
            ax[1, 1].scatter(x=max(i)[1], y=max(i)[0], color="r")

    for i in peakIff:
        if i:
            ax[1, 1].scatter(x=min(i)[1], y=min(i)[0], color="b")

    fig.show()

    plt.close()
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(8, 8)
    ax[0, 0].plot(array_with_ampl_pvalue)
    ax[0, 1].plot(array_with_time_pvalue)
    ax[1, 0].plot(array_with_2d_pvalue)
    ax[1, 1].plot(array_with_dt_pvalue)
    ax[0, 0].set(xlabel='population', ylabel='p-value ampls')
    ax[0, 1].set(xlabel='population', ylabel='p-value times')
    ax[1, 0].set(xlabel='population', ylabel='p-value 2d')
    ax[1, 1].set(xlabel='population', ylabel='p-value dt')

    fig.show()

    plt.close(fig)
    plt.plot(array_with_ampl_pvalue, label='ampls p-value')
    plt.plot(array_with_time_pvalue, label='times p-value')
    plt.plot(array_with_2d_pvalue,  label='2d p-value')
    plt.plot(ff,  label='fitness function')

    plt.legend(fontsize='medium')

    plt.show()


def draw_15(download=True):
    plt.close()

    if download:
        get_file(path_to_file_log_of_bests, path_to_save1)

    with open("../logs/log_of_bests.dat") as file:
        all_lines = file.readlines()

    arr_pack15 = []
    for line in all_lines:
        arr_pack15.append(json.loads(line, object_hook=Individual.decode))

    means_pvalue = []
    means_pvalue_ampls = []
    means_pvalue_times = []
    means_pvalue_dt = []
    pvalue = []
    pvalue_ampls = []
    pvalue_times = []
    pvalue_dt = []
    for i in arr_pack15:
        arr_pval = []
        arr_pval_ampls = []
        arr_pval_times = []
        arr_pval_dt = []
        for j in i:
            arr_pval.append(j.pvalue)
            arr_pval_ampls.append(j.pvalue_amplitude)
            arr_pval_times.append(j.pvalue_times)
            arr_pval_dt.append(j.pvalue_dt)
        means_pvalue.append(np.mean(arr_pval))
        means_pvalue_ampls.append(np.mean(arr_pval_ampls))
        means_pvalue_times.append(np.mean(arr_pval_times))
        means_pvalue_dt.append(np.mean(arr_pval_dt))
        pvalue.append(arr_pval)
        pvalue_ampls.append(arr_pval_ampls)
        pvalue_times.append(arr_pval_times)
        pvalue_dt.append(arr_pval_dt)

    fig, ax = plt.subplots(2, 2)
    fig.suptitle('Means p-value')
    fig.set_size_inches(8, 8)
    ax[0, 0].plot(means_pvalue)
    ax[0, 1].plot(means_pvalue_times)
    ax[1, 0].plot(means_pvalue_ampls)
    ax[1, 1].plot(means_pvalue_dt)

    ax[0, 0].set(xlabel='population', ylabel='means p-value 2d')
    ax[0, 1].set(xlabel='population', ylabel='means p-value times')
    ax[1, 0].set(xlabel='population', ylabel='means p-value ampls')
    ax[1, 1].set(xlabel='population', ylabel='means p-value dt')

    fig.show()

    fig, ax = plt.subplots(2, 2)
    fig.suptitle('p-value 25')
    fig.set_size_inches(8, 8)
    ax[0, 0].boxplot(pvalue)
    ax[0, 1].boxplot(pvalue_times)
    ax[1, 0].boxplot(pvalue_ampls)
    ax[1, 1].boxplot(pvalue_dt)

    ax[0, 0].set(xlabel='population', ylabel='p-value 2d')
    ax[0, 1].set(xlabel='population', ylabel='p-value times')
    ax[1, 0].set(xlabel='population', ylabel='p-value ampls')
    ax[1, 1].set(xlabel='population', ylabel='p-value dt')

    fig.show()

    # raise Exception
    #
    # for i in range(len(arr_pack15[0])):
    #     print(arr_pack15[11][i])
    #     print(arr_pack15[14][i])
    #
    # print(arr_pack15[10][0].gen)
    # print(arr_pack15[10][1].gen)


if __name__ == "__main__":

    # draw_plot()

    draw_15(False)