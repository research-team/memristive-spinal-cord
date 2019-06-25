from GRAS.shadows_boxplot import plot_shadows_boxplot
from analysis.patterns_in_bio_data import bio_data_runs
import numpy as np

bio_data = bio_data_runs()
bio_data_np = np.array(bio_data)
save_folder = '/home/anna/Desktop/lab/boxplot_shadows'
filename = 'quadrupedal_control_15'
plot_shadows_boxplot(bio_data_np, 40, 0.25, save_folder, filename)