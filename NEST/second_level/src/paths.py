import os
from pkg_resources import resource_filename

results_path = resource_filename('second_level', 'results')
raw_data_path = os.path.join(results_path, 'raw_data')
img_path = os.path.join(results_path, 'img')
spiketimes_path = os.path.join(results_path, 'spiketimes.txt')
topologies_path = 'second_level.src.topologies'
