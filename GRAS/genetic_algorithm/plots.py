from matplotlib import pyplot as plt
import os
import time

# sudo apt-get install sshpass

password = '"there is should be password"'

path_to_save1 = "/Desktop/ga_v9/logs"

path_to_file1 = "/GRAS/multi_gpu_test/files/bests_pvalue.dat"
path_to_file2 = "/GRAS/multi_gpu_test/files/history.dat"


def get_file(path_to_file=path_to_file1, path_to_save=path_to_save1):
    os.system(f'sshpass -p {password} scp openlab@10.160.173.2:~{path_to_file} ~{path_to_save}')


def draw_plot():
    get_file()

    array_with_ampl_pvalue = []

    with open("../logs/bests_pvalue.dat") as file:
        while True:
            try:
                file.readline()
                array_with_ampl_pvalue.append(float(file.readline().replace(",", "").split()[2]))
                file.readline()
                file.readline()
                file.readline()
                file.readline()
            except:
                break

    plt.xlabel("population")
    plt.ylabel("p-value ampl")

    plt.plot(array_with_ampl_pvalue)
    plt.show()


if __name__ == "__main__":
    draw_plot()