import os
from nest import Create
from the_second_level.src.paths import raw_data_path

def add_spike_detector(name):
	"""
	Args:
		name: neurons group name
    Returns:
		list: list of spikedetector GID
    """
	return Create(
		model='spike_detector',
		n=1,
		params={
			'label': os.path.join(raw_data_path, name),
			'withgid': True,
			'to_file': True,
			'to_memory': True})
