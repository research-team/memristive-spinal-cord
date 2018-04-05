import os
from pkg_resources import resource_filename

results_path = resource_filename('neuron_group_filtration_test', 'results')
raw_data_path = os.path.join(results_path, 'raw_data')
img_path = os.path.join(results_path, 'img')
