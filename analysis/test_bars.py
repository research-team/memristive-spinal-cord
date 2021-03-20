import numpy as np
import matplotlib.pyplot as plt

def plot(indicies, heights, pvalues):
	"""

	"""
	text = {'ns': (5e-2, 1),
	        '*': (1e-2, 5e-2),
	        '**': (1e-3, 1e-2),
	        '***': (1e-4, 1e-3),
	        # '****': (0, 1e-4)
	        }
	sorted_pairs = sorted(pairs, key=lambda pair: pair[1] - pair[0])
	line_upper = max(heights) * 0.08
	serif_size = line_upper / 5
	bar_shift = 1 / 5

	def calc_line_height(pair):
		l_bar, r_bar = pair
		return max(heights[l_bar], heights[r_bar], *heights[l_bar:r_bar]) + line_upper

	def diff(h1, h2):
		return abs(h2 - h1) < line_upper / 2

	line_height = list(map(calc_line_height, sorted_pairs))

	# plot text and lines
	for index in range(len(sorted_pairs) - 1):
		left_bar, right_bar = sorted_pairs[index]
		hline = line_height[index]
		# line
		line_x1, line_x2 = left_bar + bar_shift, right_bar - bar_shift
		# serifs
		serif_x1, serif_x2 = left_bar + bar_shift, right_bar - bar_shift
		serif_y1, serif_y2 = hline - serif_size, hline

		plt.plot([line_x1, line_x2], [hline, hline], color='k')
		plt.plot([serif_x1, serif_x1], [serif_y1, serif_y2], color='k')
		plt.plot([serif_x2, serif_x2], [serif_y1, serif_y2], color='k')

		# check the next lines and move them upper if need
		for i1 in range(index + 1, len(sorted_pairs)):
			if diff(line_height[i1], line_height[index]):
				line_height[i1] += line_upper
		pvalue = pvalues[(left_bar, right_bar)]

		for t, (l, r) in text.items():
			if l < pvalue <= r:
				plt.text((left_bar + right_bar) / 2, hline + line_upper / 5, t, ha='center')

	plt.bar(indicies, heights)

	plt.ylim(0, max(line_height) + line_upper)
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	times = 6
	pairs = []
	for i in range(times - 1):
		for j in range(i + 1, times):
			pairs.append((i, j))
	print(pairs)

	heights = np.random.uniform(3, 7, times)
	indicies = range(len(heights))
	pvalues = {k: np.random.uniform(0.001, 0.1) for k in pairs}

	plot(indicies, heights, pvalues)