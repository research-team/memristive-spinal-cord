import cv2
import scipy.io
import numpy as np
import pylab as plt

methodic = 0
filepath = '/home/alex/Downloads/ios_p8.mat'

dat = scipy.io.loadmat(filepath)['components']
width, height, frames_num, methodics_num = dat.shape

# per frame
for frame in range(frames_num):
	plt.suptitle(f"Frame: {frame + 1}/{frames_num}")
	# get data
	image = dat[:, :, frame, methodic]
	# normalize data from 0 to 255 with dynamic borders (min and max). It mades grayscale cmap
	image = ((image - image.min()) * (1 / (image.max() - image.min()) * 255)).astype('uint8')
	# set the dynamic thresh value to catch the most "deviant" data -- 95%
	thresh_value = np.percentile(image.ravel(), 95)

	# plotting original picture
	plt.subplot(221)
	plt.title("Original image")
	plt.imshow(image, cmap='gray')
	# plotting blured by median with filtered dots
	plt.subplot(222)
	plt.title("Median blured + dots")
	# blur the image
	median_blur = cv2.medianBlur(image, 5)
	# get coordinates of points which are greater than thresh value
	y, x = np.where(median_blur >= thresh_value)
	# plotting blur
	plt.imshow(median_blur, cmap='gray')
	# for better percentile calculating we need more than 10 dots
	if len(y) > 10:
		# calc the quartiles (without medians)
		q1_x, _, q3_x = np.percentile(x, [5, 50, 95])
		q1_y, _, q3_y = np.percentile(y, [5, 50, 95])
		# filter dots which a far away from the main group of dots
		mask = (((x <= q3_x) & (x >= q1_x)) & ((y <= q3_y) & (y >= q1_y)))
		# plot dots which a noisy-like
		plt.plot(x[~mask], y[~mask], '.', color='pink', ms=8)
		# get the dots which form an object
		good_x, good_y = x[mask], y[mask]
		plt.plot(good_x, good_y, '.', color='r', ms=8)

	# plot the classical threshold mask and calculate contour area
	plt.subplot(223)
	plt.title(f"Mask, morph open/close")
	kernel = np.ones((3, 3), np.uint8)
	# get the mask based on dynamic thresh value
	_, thresh = cv2.threshold(median_blur, thresh_value, 255, cv2.THRESH_BINARY)
	# transform morphology of the mask
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	# get the contour of the mask
	im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	# get an area of contour
	area = sum(cv2.contourArea(c) for c in contours)
	# plot the mask of thresh
	plt.imshow(thresh, cmap='gray')

	# optional - plot the approximated poly area
	plt.subplot(224)
	plt.title(f"Approx Poly DP = {area:.2f}")
	plt.imshow(median_blur, cmap='gray')
	# approximate each found contour
	for c in contours:
		if len(c) > 5:
			epsilon = 0.01 * cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, epsilon, True).squeeze()
			approx = np.vstack((approx, approx[0, :]))
			plt.fill(approx[:, 0], approx[:, 1], color='g', alpha=0.5)
			plt.plot(approx[:, 0], approx[:, 1], color='g')

	plt.pause(0.0005)
	plt.draw()
	plt.clf()

plt.close()
