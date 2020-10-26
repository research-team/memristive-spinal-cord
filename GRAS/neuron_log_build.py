def offset_lines(f, lines):
	for _ in range(lines):
		f.readline()

import numpy as np

data = []
#
with open('/home/alex/NEURTEST/rawlog') as file:
	offset_lines(file, 87)
	iterattion = []
	while file:
		try:
			iterattion = float(file.readline().replace("=", ''))
			row = [iterattion]
			offset_lines(file, 35)
			row += list(map(float, file.readline().split("\t")))
			offset_lines(file, 34)
			for i in range(3):
				row += list(map(float, file.readline().split("\t")))
			data.append(row)
			offset_lines(file, 3)
		except Exception as e:
			print(iterattion)
			print(e)
			break

	data = np.array(data)
	names = "t cai il ina ik ica Eca v m h n p mc hc " + " ".join(f"A{i} B{i} D{i} RINV{i} Vm{i}" for i in range(3))
	names = names.split()
	assert len(names) == len(data[0])
	vals = len(names)
	template_s = "{}\t" * vals
	template_f = "{}\t" * vals
	with open("/home/alex/NEURTEST/tablelog", 'w') as file:
		file.write(template_s.format(*names) + "\n")
		for i, d in enumerate(data):
			file.write(template_f.format(*d) + "\n")
