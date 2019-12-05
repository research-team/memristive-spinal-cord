from analysis.article_plotting import process_dataset

# folder where will be saved all PDF
save_all_to = '/home/alex/GitHub/DATA/test'

# extendable list of short meta info about datafile which can be mapped on all datas (neuron, gras, nest)
# is need for fast data processing
comb = [
	("PLT", 21, 2, 0.1),
]

for c in comb:
	# datafiles which will be compared
	compare_pack = [
		'/home/alex/BIO/bio_E_PLT_13.5cms_40Hz_2pedal_0.1step.hdf5',
		'/home/alex/BIO/bio_E_STR_13.5cms_40Hz_2pedal_0.1step.hdf5',
	]
	# what do you need to do with data?
	flags = dict(plot_pca3d=False,
	             plot_correlation=False,
	             plot_slices_flag=False,
	             plot_ks2d=True,
	             ks_analyse='poly', # full mono poly
	             plot_peaks_by_intervals=False)
	# be carafull with convert_dstep_to -- very important to compare data at one data step size
	process_dataset(compare_pack, save_all_to, flags, convert_dstep_to=0.1)
