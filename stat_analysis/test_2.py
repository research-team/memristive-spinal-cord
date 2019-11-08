import h5py
import numpy as np
import math as mt
from scipy import stats


#extensor, foot, скорость 13.5
with h5py.File('C:\\Users\\Ангелина\\PycharmProjects\\pop\\HDF5 MAIN\\foot\\3\\bio_E_13.5cms_40Hz_i100_2pedal_no5ht_T_0.1step.hdf5', 'r') as f_foot_3:
	data_foot_3 = [test_values3[:] for test_values3 in f_foot_3.values()]

with h5py.File('C:\\Users\\Ангелина\\PycharmProjects\\pop\\HDF5 MAIN\\foot\\4\\bio_E_13.5cms_40Hz_i100_2pedal_no5ht_T_0.1step.hdf5', 'r') as f_foot_4:
	data_foot_4 = [test_values4[:] for test_values4 in f_foot_4.values()]

#усредняем две крысы
data_foot_3 = np.array(data_foot_3)
data_foot_4 = np.array(data_foot_4)
data_foot = np.vstack((data_foot_4, data_foot_4))
print(data_foot.shape)
#extensor, qpz, скорость
with h5py.File('C:\\Users\\Ангелина\\PycharmProjects\\pop\\HDF5 MAIN\\qpz\\7\\bio_E_13.5cms_40Hz_i100_2pedal_5ht_T.hdf5', 'r') as f_foot_7:
	data_qpz_7 = [test_values7[:] for test_values7 in f_foot_7.values()]

with h5py.File('C:\\Users\\Ангелина\\PycharmProjects\\pop\\HDF5 MAIN\\qpz\\4\\bio_E_13.5cms_40Hz_i100_2pedal_5ht_T.hdf5', 'r') as f_foot_4:
	data_qpz_4 = [test_values4[:] for test_values4 in f_foot_4.values()]

data_qpz_4 = np.array(data_qpz_4)
data_qpz_7 = np.array(data_qpz_7)
data_qpz = np.vstack((data_qpz_4, data_qpz_7))

d=[]
for i in data_foot:
	data_foot


print(data_qpz.shape)

#делаем Колмогорова-Смирнова для air и qpz
t = stats.kstest(data_foot, data_qpz)
print(t)
# z = []
# for i in range(len(data_foot_4_mean)):
# 	z.append(mt.fabs(data_qpz_4_mean[i] - data_foot_4_mean[i]))
# print(max(z))
