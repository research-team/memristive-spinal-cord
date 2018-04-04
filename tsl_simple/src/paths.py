import os
from pkg_resources import resource_filename

results_path = resource_filename('tsl_simple', 'results')
raw_data_path = os.path.join(results_path, 'raw_data')
img_path = os.path.join(results_path, 'img')
