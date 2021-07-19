import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

DEBUG = True

# dataset 1
x1 = np.random.normal(3, 0.4, 200)
y1 = np.random.normal(3, 0.5, 200)

# dataset 2
x2 = np.random.normal(3, 0.41, 240)
y2 = np.random.normal(3, 0.51, 240)


def form_1d_kde(X, xmin, xmax):
	xx = np.linspace(xmin, xmax, 1000)
	dx = gaussian_kde(X)(xx)
	return xx, dx


def form_2d_kde(x, y, xmin, xmax, ymin, ymax):
	xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
	# re-present grid in 1D and pair them as (x1, y1 ...)
	positions = np.vstack([xx.ravel(), yy.ravel()])
	values = np.vstack([x, y])
	# use a Gaussian KDE
	a = gaussian_kde(values)(positions).T
	# re-present grid back to 2D
	z = np.reshape(a, xx.shape)

	return xx, yy, z


def overlap_density(d1, d2):
	assert d1.size == d2.size
	overlay = np.sum(d1[d1 <= d2]) + np.sum(d2[d2 < d1])
	area1, area2 = np.sum(d1), np.sum(d2)
	iou = overlay / (area1 + area2 - overlay)
	return iou


def test_iou():
	# 1D
	xx1, dx1 = form_1d_kde(y1, xmin=0, xmax=25)
	xx2, dx2 = form_1d_kde(x2, xmin=0, xmax=25)

	iou = overlap_density(dx1, dx2)
	print(iou)

	# 2D
	X1, Y1, z1 = form_2d_kde(x1, y1, xmin=0, xmax=6, ymin=0, ymax=8)
	X2, Y2, z2 = form_2d_kde(x2, y2, xmin=0, xmax=6, ymin=0, ymax=8)

	iou = overlap_density(z1, z2)
	print(iou)

	if DEBUG:
		# 1D
		plt.figure()
		plt.plot(xx1, dx1)
		plt.plot(xx2, dx2)
		plt.show()

		# 2D
		plt.figure()
		plt.contour(X1, Y1, z1, levels=10, linewidths=1)
		plt.contourf(X1, Y1, z1, levels=10, alpha=0.7, zorder=0)

		plt.contour(X2, Y2, z2, levels=10, linewidths=1)
		plt.contourf(X2, Y2, z2, levels=10, alpha=0.7, zorder=0)

		plt.figure()
		ax = plt.axes(projection='3d')
		ax.plot_wireframe(X1, Y1, z1, rstride=1, cstride=1, alpha=0.5, color='r')
		ax.plot_wireframe(X2, Y2, z2, rstride=1, cstride=1, alpha=0.5, color='b')

		plt.show()


if __name__ == '__main__':
	test_iou()
