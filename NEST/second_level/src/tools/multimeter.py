import os
from nest import Create
from second_level.src.paths import raw_data_path


def add_multimeter(name, record_from='Extracellular'):
	"""
	Function for creating NEST multimeter node
	Args:
		name (str):
			name of the node to which will be connected the multimeter
		record_from (str):
			Extracellular or V_m (intracelullar) recording variants
	Returns:
		tuple: global NEST ID of the multimeter
	"""
	if record_from not in ['Extracellular', 'V_m']:
		raise Exception("The '{}' parameter is not implemented "
		                "for membrane potential recording".format(record_from))
	return Create(
		model='multimeter',
		n=1,
		params={
			'label': os.path.join(raw_data_path, name),
			'record_from': [record_from],
			'withgid': True,
			'withtime': True,
			'interval': 0.1,
			'to_file': True,
			'to_memory': True})
