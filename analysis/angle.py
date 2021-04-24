import matplotlib.pyplot as plt
import numpy as np


def moving_average(data, weight):
	return np.convolve(data, np.ones(weight), 'valid') / weight


def read_data(datapath, filename, savepath):
	angle1, angle2 = [], []
	with open(f'{datapath}/{filename}') as file:
		for line in file:
			angle_data = list(map(int, (line.split())))
			angle1.append(angle_data[3])
			angle2.append(angle_data[-1])

	angle1_smoothed = moving_average(angle1, 15)
	angle2_smoothed = moving_average(angle2, 15)

	plt.plot(angle1_smoothed, 'k')
	plt.plot(angle2_smoothed, 'r')
	plt.savefig(f'{savepath}/{filename[:-4]}.png', format='png')
	plt.show()

	max_angle1 = abs(max(angle1) - angle1_smoothed[0]) / 20
	min_angle2 = abs(min(angle2) - angle2_smoothed[0]) / 20

	with open(f'{savepath}/{filename[:-4]}_result.txt', 'w') as f:
		f.write(f'максимальный 1й угол = {max_angle1}, '
		        f'минимальный 2й угол = {min_angle2}')

	print(
		f'максимальный 1й угол = {max_angle1}, '
		f'минимальный 2й угол = {min_angle2}')


def main():
	datapath = 'C:/Users/exc24/Desktop'
	filename_Max1 = 'Max1.txt'
	filename_Max2 = 'Max2.txt'
	filename_Ann = 'Ann.txt'
	savepath = 'C:/Users/exc24/Desktop/fig'

	read_data(datapath, filename_Max1, savepath)


if __name__ == '__main__':
	main()
