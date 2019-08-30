from analysis.functions import read_data
from matplotlib import pylab as plt

bio_runs = read_data('/home/anna/Desktop/data/bio/4pedal/bio_F_21cms_40Hz_i100_4pedal_no5ht_T_0.25step.hdf5')

all_bio_slices = []
step = 0.25
for k in range(len(bio_runs)):
	bio_slices= []
	offset= 0
	for i in range(int(len(bio_runs[k]) / 100)):
		bio_slices_tmp = []
		for j in range(offset, offset + 100):
			bio_slices_tmp.append(bio_runs[k][j])
		bio_slices.append(bio_slices_tmp)
		offset += 100
	all_bio_slices.append(bio_slices)   # list [4][16][100]

print(len(all_bio_slices), len(all_bio_slices[0]), len(all_bio_slices[0][0]))