import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    with open("bio.txt") as file:
        bio_arr = list(map(lambda x: float(x), file.readline().split(", ")))

    print(bio_arr)
    print(f"Len of bio arr = {len(bio_arr)}")

    # plt.plot(bio_arr[0:250])
    # plt.show()

    with open("gras.txt") as file:
        gras_arr = list(map(lambda x: float(x), file.readline().split(", ")))

    print(gras_arr)
    print(f"Len of gras arr = {len(gras_arr)}")

    bio_arr = gras_arr[:250]

    # plt.plot(bio_arr[0:250])
    # plt.show()

    # bio_arr = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 7, 5, 3, 2, 1, 5, 9, 11, 12, 18, 17, 12, 10, 8, 6, 4]
    peakI = [[]]
    peakD = [[]]

    inc = 0
    dec = 0
    flag = False
    for i in range(len(bio_arr) - 1):
        if bio_arr[i] > bio_arr[i + 1]:
            peakD[dec].append((bio_arr[i], i))
            flag = True
        elif flag:
            peakD.append([])
            dec += 1
            flag = False

    flag = False
    for i in range(len(bio_arr) - 1):
        if bio_arr[i] < bio_arr[i + 1]:
            peakI[inc].append((bio_arr[i], i))
            flag = True
        elif flag:
            peakI.append([])
            inc += 1
            flag = False

    print(peakD)
    print(peakI)

    for i in peakD:
        if i:
            plt.scatter(x=max(i)[1], y=max(i)[0], color="r")

    for i in peakI:
        if i:
            plt.scatter(x=min(i)[1], y=min(i)[0], color="b")

    maximums = []

    for peak in peakD:
        if peak:
            maximums.append(max(peak))

    print(len(maximums))

    diffs = []

    for i in range(len(maximums) - 1):
        diffs.append(maximums[i + 1][1] - maximums[i][1])

    print(diffs)

    # plt.plot(bio_arr)
    # plt.show()