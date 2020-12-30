from analysis.article_plotting import Analyzer
import pylab as plt

analyzer = Analyzer(pickle_folder='/Users/sulgod/Desktop/res/cpg/pickle/')
# if new data
# analyzer.prepare_data(folder='/Users/sulgod/Desktop/res/cpg/4peda/', dstep_to=None, fill_zeros=True, filter_val=0.09)
# analyzer.prepare_data(folder='/Users/sulgod/Desktop/res/cpg/newrats/', dstep_to=0.025, fill_zeros=True, filter_val=0.09)

comb = [
	('bio_E_PLT_13.5cms_40Hz_2pedal_0.1step', [4]), # 3???
	# ('bio_E_PLT_21cms_40Hz_2pedal_0.1step', [3, 4]),
	# ('bio_E_PLT_6cms_40Hz_2pedal_0.1step', [8]),
	# ('bio_E_PLT_21cms_40Hz_4pedal_0.25step', [1]),
	# ('bio_E_PLT_13.5cms_40Hz_4pedal_0.25step', [1]),
	# ('bio_E_STR_21cms_40Hz_2pedal_0.1step', [1, 3, 4]),
	# ('bio_E_STR_13.5cms_40Hz_2pedal_0.1step', [1, 3, 4, 8]),
	# ('bio_E_STR_6cms_40Hz_2pedal_0.1step', [1, 3, 8]),
	# ('bio_E_QPZ_13.5cms_40Hz_2pedal_0.1step', [4, 7]), #8 ?
	# ('bio_E_TOE_13.5cms_40Hz_2pedal_0.1step', [3, 8]),
	# ('bio_E_AIR_13.5cms_40Hz_2pedal_0.1step', [7, 8]),
	# ('neuron2_E_PLT_13.5cms_40Hz_2pedal_0.025step', [0]),
	# ('neuron99_E_STR_13.5cms_40Hz_2pedal_0.025step', [0]),
	# ('neuron7_E_PLT_13.5cms_40Hz_2pedal_0.025step', [0]),
	# ('neuron99_E_PLT_13.5cms_40Hz_4pedal_0.025step', [0]),
	# ('neuron5_E_AIR_13.5cms_40Hz_2pedal_0.025step', [0]),
	# ('neuron99_E_TOE_13.5cms_40Hz_2pedal_0.025step', [0]),
	# ('neuron99_E_QPZ_13.5cms_40Hz_2pedal_0.025step', [0]),
]
"""
* При ненадобности -- закомментировать
"""
# plt.close()

for filename, rats in comb:
	# analyzer.plot_density_3D(source=filename, rats=rats, show=False)
	# analyzer.plot_fMEP_boxplots(source=filename, rats=rats, borders=[0, 25], show=False)

	# analyzer.plot_amp_boxplots(source=filename, rats=rats, borders=[8, 28], show=False)
	analyzer.plot_shadow_slices(source=filename, rats=rats, add_kde=True, show=False)
	analyzer.print_metainfo(source=filename, rats=rats)

# plt.legend()
# plt.show()
"""
* При ненадобности -- закомментировать
* В анализе 'outside_compare' используется только мыщца extensor
* Сравниваются все крысы указанные в comb по-отдельности и все шаги полным перебором (один ко всем)
* 'outside_compare' используется для построения и теста KDE (пик-слайс и пик-амплитуда)
"""
# analyzer.outside_compare(comb, border=[8,28], axis=['ampl', 'slice'], muscletype='E', per_step=False, plot=True, show=False)

# analyzer.outside_compare(comb, border=[0,25], axis=['ampl', 'slice'], muscletype='F', per_step=False, plot=True, show=False)

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
