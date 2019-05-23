from statsmodels.stats.stattools import jarque_bera
from analysis.patterns_in_bio_data import bio_data_runs
import numpy as np

bio_data = bio_data_runs()
bio_mean = list(map(lambda elements: np.mean(elements), zip(*bio_data.values())))
checking = jarque_bera(bio_mean)
print(checking)