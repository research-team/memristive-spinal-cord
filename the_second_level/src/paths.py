import os
from pkg_resources import resource_filename

results_path = resource_filename('the_second_level', 'results')
raw_data_path = os.path.join(results_path, 'raw_data')
img_path = os.path.join(results_path, 'img')
topologies_path = 'the_second_level.src.topologies'
