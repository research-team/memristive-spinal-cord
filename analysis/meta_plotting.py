from analysis.article_plotting import Analyzer


analyzer = Analyzer(pickle_folder='/Users/sulgod/Desktop/res/cpg/pickle/')
# if new data
# analyzer.prepare_data(folder='/home/alex/BIO/', dstep_to=0.1, fill_zeros=True, filter_val=0.05)
analyzer.prepare_data(folder='/Users/sulgod/Desktop/res/cpg/hips', dstep_to=0.025, fill_zeros=True, filter_val=0.05)

comb = [
	# ('bio_E_PLT_13.5cms_40Hz_2pedal_0.1step', [4, 8]), # 3???
	# ('bio_E_PLT_21cms_40Hz_2pedal_0.1step', [3, 4]),
	# ('bio_E_PLT_6cms_40Hz_2pedal_0.1step', [8]),
	# ('bio_E_PLT_21cms_40Hz_4pedal_0.25step.hdf5', [None]),
	# ('bio_E_PLT_13.5cms_40Hz_4pedal_0.25step.hdf5', [None]),
	# ('bio_E_STR_21cms_40Hz_2pedal_0.1step', [1, 3, 4]),
	# ('bio_E_STR_13.5cms_40Hz_2pedal_0.1step', [1, 3, 4, 8]),
	# ('bio_E_STR_6cms_40Hz_2pedal_0.1step', [1, 3, 8]),
	# ('bio_E_QPZ_13.5cms_40Hz_2pedal_0.1step', [4, 7]), #8 ?
	# ('bio_E_TOE_13.5cms_40Hz_2pedal_0.1step', [3, 8]),
	# ('bio_E_AIR_13.5cms_40Hz_2pedal_0.1step', [7, 8]),
	('neuron_E_PLT_21cms_40Hz_2pedal_0.025step', [0]),
]
"""
* При ненадобности -- закомментировать
"""
for filename, rats in comb:
	# analyzer.plot_density_3D(source=filename, rats=rats, show=False)
	analyzer.print_metainfo(source=filename, rats=rats)
	# analyzer.plot_fMEP_boxplots(source=filename, rats=rats, borders=[[3, 8], [8, 25]], show=True)
	analyzer.plot_shadow_slices(source=filename, rats=rats, add_kde=True, show=False)
"""
* При ненадобности -- закомментировать
* В анализе 'outside_compare' используется только мыщца extensor
* Сравниваются все крысы указанные в comb по-отдельности и все шаги полным перебором (один ко всем)
* 'outside_compare' используется для построения и теста KDE (пик-слайс и пик-амплитуда)
"""
# analyzer.outside_compare(comb, border='poly_tail', axis=['time', 'ampl'],
#                          per_step=False, plot=True, show=True)

"""
* При ненадобности -- закомментировать
* Кумулятивный график, требует список: какие режимы сравнивать друг с другом и в каком порядке
"""
cmltv = [
	("bio_AIR_13.5", "bio_TOE_13.5", 0.1),
	("bio_TOE_13.5", "bio_PLT_13.5", 0.1),
	("bio_PLT_13.5", "bio_QPZ_13.5", 0.1),
	("bio_PLT_13.5", "bio_STR_13.5", 0.1),
	("bio_PLT_13.5", "bio_PLT_21", 0.1),
	("bio_PLT_21", "bio_STR_21", 0.1),
	("bio_AIR_13.5", "bio_PLT_13.5", 0.1),
]
order = ['bio_AIR_13.5', 'bio_TOE_13.5', 'bio_PLT_13.5', 'bio_QPZ_13.5', 'bio_STR_13.5', 'bio_STR_21', 'bio_PLT_21']
# analyzer.plot_cumulative(cmltv, border='poly_tail', order=order) # borders: 'poly_tail' or list [t1, t2]
