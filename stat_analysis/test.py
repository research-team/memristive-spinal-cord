import h5py
import numpy as np
from scipy import stats
import math as mt
#extensor, foot, скорость 13.5
with h5py.File('C:\\Users\\Ангелина\\PycharmProjects\\pop\\HDF5 MAIN\\foot\\3\\bio_E_13.5cms_40Hz_i100_2pedal_no5ht_T_0.1step.hdf5','r') as f_foot_3:
	data_foot_3 = [test_values3[:] for test_values3 in f_foot_3.values()]

with h5py.File('C:\\Users\\Ангелина\\PycharmProjects\\pop\\HDF5 MAIN\\foot\\4\\bio_E_13.5cms_40Hz_i100_2pedal_no5ht_T_0.1step.hdf5','r') as f_foot_4:
	data_foot_4 = [test_values4[:] for test_values4 in f_foot_4.values()]

#усредняем две крысы
data_foot_3 = np.array(data_foot_3)
data_foot_4 = np.array(data_foot_4)
print(data_foot_3.shape)
print(data_foot_4.shape)
data_foot = np.vstack((data_foot_3, data_foot_4))

mean_data_foot = np.mean(data_foot, axis=0)
print(mean_data_foot.shape)


#extensor, qpz, скорость
with h5py.File('C:\\Users\\Ангелина\\PycharmProjects\\pop\\HDF5 MAIN\\qpz\\7\\bio_E_13.5cms_40Hz_i100_2pedal_5ht_T.hdf5','r') as f_foot_7:
	data_qpz_7 = [test_values7[:] for test_values7 in f_foot_7.values()]

with h5py.File('C:\\Users\\Ангелина\\PycharmProjects\\pop\\HDF5 MAIN\\qpz\\4\\bio_E_13.5cms_40Hz_i100_2pedal_5ht_T.hdf5','r') as f_foot_4:
	data_qpz_4 = [test_values4[:] for test_values4 in f_foot_4.values()]

data_qpz_4 = np.array(data_qpz_4)
data_qpz_7 = np.array(data_qpz_7)

data_qpz = np.vstack((data_qpz_4, data_qpz_7))

mean_data_qpz = np.mean(data_qpz, axis=0)
print(mean_data_qpz.shape)

#делаем Колмогорова-Смирнова для air и qpz
# z = []
# for i in range(len(mean_data_foot)):
# 	z.append(mt.fabs(mean_data_foot[i] - mean_data_qpz[i]))
# print(max(z))
# p_value = 1
t = stats.kstest(mean_data_foot, mean_data_qpz)
print(t)
