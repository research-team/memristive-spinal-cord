from analysis.article_plotting import process_dataset

# folder where will be saved all PDF
save_all_to = '/home/alex/GitHub/DATA/test1'

# extendable list of short meta info about datafile which can be mapped on all datas (neuron, gras, nest)
# is need for fast data processing
# for sim data set [None] in rat number if there is only one simulation dataset
# otherwise use the correct "rat" number as index of data (save the hdf properly -- '#1' ..'#N'!)
comb = [
	('/home/alex/BIO/bio_E_PLT_13.5cms_40Hz_2pedal_0.1step.hdf5', [1, 3, 4, 8]),
	# ('/home/alex/BIO/bio_E_PLT_21cms_40Hz_2pedal_0.1step.hdf5', [3, 4]),
	('/home/alex/BIO/bio_E_PLT_6cms_40Hz_2pedal_0.1step.hdf5', [1, 8]),
	# ('/home/alex/BIO/bio_E_PLT_21cms_40Hz_4pedal_0.25step.hdf5', [None]),
	# ('/home/alex/BIO/bio_E_PLT_13.5cms_40Hz_4pedal_0.25step.hdf5', [None]),
	# ('/home/alex/BIO/bio_E_QPZ_13.5cms_40Hz_2pedal_0.1step.hdf5', [4, 7, 8]),
	# ('/home/alex/BIO/bio_E_STR_13.5cms_40Hz_2pedal_0.1step.hdf5', [1, 3, 4, 7, 8]),
	# ('/home/alex/BIO/bio_E_TOE_13.5cms_40Hz_2pedal_0.1step.hdf5', [3, 4, 8]),
	# ('/home/alex/BIO/bio_E_AIR_13.5cms_40Hz_2pedal_0.1step.hdf5', [7, 8]),
]

for i in range(len(comb)):
	for j in range(i + 1, len(comb)):
		for rat1 in comb[i][1]:
			for rat2 in comb[j][1]:
				pack = [
					(comb[i][0], rat1),
					(comb[j][0], rat2)
				]

				# what do you need to do with data?
				flags = dict(plot_pca3d=False,
				             plot_correlation=False,
				             plot_slices_flag=False,
				             kde_peak_slice=False,
				             plot_ks2d=False,
				             plot_ks_b=True,
				             ks_analyse='poly_tail', # mono, poly, poly_tail or [t1, t2] in ms
				             plot_peaks_by_intervals=False)
				# be carafull with convert_dstep_to -- very important to compare data at one data step size
				process_dataset(pack, save_all_to, flags, convert_dstep_to=0.1)
