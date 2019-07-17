import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

with open('dT.dat') as file:
	dT = np.array(list(map(float, file.read().split())))

with open('dW.dat') as file:
	dW = np.array(list(map(float, file.read().split())))

