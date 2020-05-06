from analysis.article_plotting import Analyzer

comb = [
	('bio_E_PLT_13.5cms_40Hz_2pedal_0.1step', [3, 4, 8]), # except 1
	('bio_E_PLT_21cms_40Hz_2pedal_0.1step', [3, 4]),
	('bio_E_PLT_6cms_40Hz_2pedal_0.1step', [8]), # except 1
	# ('/home/alex/BIO/bio_E_PLT_21cms_40Hz_4pedal_0.25step.hdf5', [None]),
	# ('/home/alex/BIO/bio_E_PLT_13.5cms_40Hz_4pedal_0.25step.hdf5', [None]),
	('bio_E_STR_21cms_40Hz_2pedal_0.1step', [1, 3, 4]),
	('bio_E_STR_13.5cms_40Hz_2pedal_0.1step', [1, 3, 4, 8]), # except 7
	('bio_E_STR_6cms_40Hz_2pedal_0.1step', [1, 3, 8]), # except 7
	('bio_E_QPZ_13.5cms_40Hz_2pedal_0.1step', [4, 7, 8]),
	('bio_E_TOE_13.5cms_40Hz_2pedal_0.1step', [3, 4, 8]),
	('bio_E_AIR_13.5cms_40Hz_2pedal_0.1step', [7, 8]),
]

analyzer = Analyzer(pickle_folder='/home/alex/pickle')
# if new data
# analyzer.prepare_data(folder='/home/alex/BIO/', dstep_to=0.1)

for filename, rats in comb:
	analyzer.plot_density_3D(file=filename, rats=rats, show=False)
	analyzer.plot_fMEP_boxplots(file=filename, rats=rats, borders=[3, 8], show=False)
	analyzer.plot_shadow_slices(file=filename, rats=rats, add_kde=True, show=False)
	# print(analyzer.get_latency_volume(file=filename, rats=rats))
