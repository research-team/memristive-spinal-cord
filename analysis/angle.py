import matplotlib.pyplot as plt
import numpy as np
import os


# exclude = ('40', '5')


def moving_average(data, weight):
    return np.convolve(data, np.ones(weight), 'valid') / weight


def calc_min(angle_list):
    without_zero = [i for i in angle_list if i != 0]
    return min(without_zero)


def plot(angle_hip, angle_ankle, savepath, filename, show=False):
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(angle_hip, 'k')
    plt.plot(angle_ankle, 'r')
    plt.savefig(f'{savepath}/{filename[:-4]}.png', format='png')
    if show:
        plt.show()


def read_data(datapath, savepath, type):
    filenames = [name for name in os.listdir(f"{datapath}") if name.endswith(".txt")]
    for filename in filenames:
        if '5' not in filename:
            angle_hip, angle_ankle = [], []
            with open(f'{datapath}/{filename}') as file:
                for line in file:
                    angle_data = list(map(int, (line.split())))
                    angle_hip.append(angle_data[-1])
                    angle_ankle.append(angle_data[3])

            # сглаживает данные
            angle_hip_smoothed = moving_average(angle_hip, 150)
            angle_ankle_smoothed = moving_average(angle_ankle, 150)

            # отрисовка
            plot(angle_hip_smoothed, angle_ankle_smoothed, savepath, filename, show=False)

            # расчет углов в зависимоти от эксперимента
            if type == 'free walking':
                angle_hip = round(abs(calc_min(angle_hip_smoothed) - angle_hip_smoothed[0]) / 20, 2)
                angle_ankle = round(abs(max(angle_ankle_smoothed) - angle_ankle_smoothed[0]) / 20, 2)

            if type == 'pattern':
                if filename.startswith('test1'):  # extensor
                    angle_hip = round(abs(max(angle_hip_smoothed) - angle_hip_smoothed[0]) / 20, 2)
                    angle_ankle = round(abs(max(angle_ankle_smoothed) - angle_ankle_smoothed[0]) / 20, 2)

                if filename.startswith('test2'):  # flexor
                    angle_hip = round(abs(calc_min(angle_hip_smoothed) - angle_hip_smoothed[0]) / 20, 2)
                    angle_ankle = round(abs(calc_min(angle_ankle_smoothed) - angle_ankle_smoothed[0]) / 20, 2)

            with open(f'{savepath}/{filename}_result.txt', 'w') as f:
                f.write(f'голеностоп    колено' + '\n')
                f.write(f'{angle_ankle}\t{angle_hip}')

            print(f'голеностоп = {angle_ankle}, колено = {angle_hip}')


def average(path):
    filenames = [name for name in os.listdir(f"{path}") if name.endswith(".txt")]
    for filename in filenames:
        if filename.startswith('test1'):
            if '40' in filename:
                pass


def main():
    # datapath = 'C:/Users/Ann/Desktop/angles'
    # savepath = 'C:/Users/Ann/Desktop/angles/results'
    # type = 'free walking'
    # read_data(datapath, savepath, type=type)

    path = '/home/b-rain/angles/protocol_data'
    foldernames = [name for name in os.listdir(f"{path}")]
    type = 'pattern'
    for foldername in foldernames:
        datapath = os.path.join(path, foldername)
        savepath = f'/home/b-rain/angles/protocol_data/{foldername}/results'
        if os.path.exists(savepath):
            pass
        else:
            os.mkdir(savepath)
        read_data(datapath, savepath, type=type)


if __name__ == '__main__':
    main()
