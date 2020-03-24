import os
import sys
import cv2
import time
import h5py
import numpy as np
import scipy.io as sio
import scipy.stats as st
from skimage import measure
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator, QFont
from PyQt5.QtWidgets import QPushButton, QLabel, QRadioButton, QLineEdit, QComboBox
from PyQt5.QtWidgets import QGridLayout, QFileDialog, QApplication, QMainWindow, QWidget, QVBoxLayout, QFrame

original_image = "original"
reversed_image = "reversed"

class AppForm(QMainWindow):
	def __init__(self, parent=None):
		QMainWindow.__init__(self, parent)
		self.setWindowTitle('Computer Vision Analyzer v2.0 (NcN lab product 2020)')
		self.create_main_frame()
		self.create_status_bar()

		self.current_frame = 0
		self.analyse_type = original_image
		self.excl_indices = None

	def open_file(self, path):
		try:
			self.dat = sio.loadmat(path)['components'][:]
			return self.dat.shape
		except NotImplementedError:
			with h5py.File(path, 'r') as file:
				self.dat = file['components'][:]
				return self.dat.shape
		except:
			ValueError('Could not read the file at all...')

	def reshape_data(self):
		# get new shape as tuple
		new_order = tuple(map(int, self.in_data_reshape.text().split()))
		# reshape data to the new shape
		self.dat = np.transpose(self.dat, new_order)
		new_shape = self.dat.shape
		#
		self.image_height = new_shape[0]
		self.image_width = new_shape[1]
		self.total_frames = new_shape[2]
		# disable buttons of reshaping
		self.in_data_reshape.setEnabled(False)
		self.btn_reshape_data.setEnabled(False)
		# init frames in GUI form
		self.in_start_frame.setValidator(QIntValidator(0, self.total_frames - 1))
		self.in_end_frame.setValidator(QIntValidator(0, self.total_frames - 1))
		self.in_end_frame.setText(str(self.total_frames - 1))
		# form a mesh grid
		self.xx, self.yy = np.meshgrid(np.linspace(-10, new_shape[1] + 9, new_shape[1] + 20),
		                               np.linspace(-10, new_shape[0] + 9, new_shape[0] + 20))
		# re-present grid in 1D and pair them as (x1, y1 ...)
		self.xy_meshgrid = np.vstack([self.xx.ravel(), self.yy.ravel()])
		# ToDo add GUI
		self.exclude()
		# update status
		self.status_text.setText(f"Data was reshaped to {new_shape}")

	def file_dialog(self):
		fname = QFileDialog.getOpenFileName(self, 'Open file', "", "MAT file (*.mat)")
		# if exists
		if fname[0]:
			self.dat = None
			self.status_text.setText("Unpack .mat file... Please wait")
			# FixMe it is a bug of QApplication
			QApplication.processEvents()
			QApplication.processEvents()

			data_shape = self.open_file(fname[0])

			self.status_text.setText(f".mat file is unpacked, data shape {data_shape}")

			str_format = len(data_shape) * '{:<5}'
			self.label_fileinfo.setText(f"Shape: {str_format.format(*data_shape)}\n"
			                            f"Index: {str_format.format(*list(range(4)))}")
			self.label_fileinfo.setFont(QFont("Courier New"))

			for obj in [self.btn_save_results, self.btn_loop_draw, self.btn_frame_right,
			            self.btn_frame_left, self.btn_reshape_data, self.in_data_reshape]:
				obj.setEnabled(True)


	def exclude(self):
		methodic = int(self.box_method.currentText())

		diff = np.zeros(shape=self.dat[:, :, 0, methodic].shape)

		for frame in range(self.total_frames):
			diff += self.dat[:, :, frame, methodic]

		# for xi in range(self.image_width):
		# 	for yi in range(self.image_height):
		# 		self.magnitudes[yi, xi] = np.sqrt(self.dat[yi, xi, :, methodic])

		q1, m, q3 = np.percentile(diff, q=(0.5, 50, 99.5))
		row, col = np.indices(diff.shape)
		diff = diff.ravel()
		self.excl_indices = np.stack((col.ravel(), row.ravel()), axis=1)[(diff >= q3) | (diff <= q1)]

	@staticmethod
	def eigsorted(cov):
		vals, vecs = np.linalg.eigh(cov)
		order = vals.argsort()[::-1]
		return vals[order], vecs[:, order]

	@staticmethod
	def array_xor(A, B):
		aset = set(map(tuple, A))
		bset = set(map(tuple, B))
		return np.array(list(map(tuple, aset.difference(bset))))

	def update_draws(self, frame, percentile, save_result=False):
		# toDO
		max_contour_number  = 15
		# get the current methodic
		methodic = int(self.box_method.currentText())
		# get data
		original = self.dat[:, :, frame, methodic]
		# normalize data from 0 to 255 with dynamic borders (min and max). It mades grayscale cmap
		image = ((original - original.min()) * (1 / (original.max() - original.min()) * 255)).astype('uint8')
		# reverse colors if epilepsy radio button checked
		if self.radio_reversed.isChecked():
			image = 255 - image

		image = cv2.medianBlur(image, 5)
		# set the dynamic thresh value to catch the most "deviant" data -- 95%
		thresh_value = np.percentile(image.ravel(), percentile)
		# get coordinates of points which are greater than thresh value
		found_points = np.stack(np.where(image >= thresh_value), axis=1)[:, [1, 0]]
		found_points = self.array_xor(found_points, self.excl_indices)
		x, y = found_points[:, 0], found_points[:, 1]

		#### ELLIPSE ####
		nstd = 2.5
		ell_center = found_points.mean(axis=0)
		cov = np.cov(found_points, rowvar=False)
		vals, vecs = self.eigsorted(cov)
		theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
		# Width and height are "full" widths, not radius
		width, height = 2 * nstd * np.sqrt(vals)

		cos_angle = np.cos(np.radians(180 - theta))
		sin_angle = np.sin(np.radians(180 - theta))

		xc = x - ell_center[0]
		yc = y - ell_center[1]

		xct = xc * cos_angle - yc * sin_angle
		yct = xc * sin_angle + yc * cos_angle

		rad_cc = (xct ** 2 / (width / 2) ** 2) + (yct ** 2 / (height / 2) ** 2)

		# calc raw CV contours to decide -- search contour or not
		mask = np.zeros(shape=image.shape)
		mask[y, x] = 255
		mask = mask.astype('uint8')

		kernel = np.ones((3, 3), np.uint8)
		_, thresh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
		# transform morphology of the mask
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
		# get the contour of the mask
		*im2, CVcontours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		med_dist = np.median(np.sqrt((x - ell_center[0]) ** 2 + (y - ell_center[1]) ** 2))

		contours = None
		core_contour_x = None
		core_contour_y = None

		if len(CVcontours) < max_contour_number:
			# get a KDE function values based on found points with XX YY meshgrid
			a = st.gaussian_kde(np.stack(found_points, axis=1))(self.xy_meshgrid).T
			# re-present grid back to 2D
			z = np.reshape(a, self.xx.shape)
			# form a step and levels
			step = (np.amax(a) - np.amin(a)) / 3
			# Find contours at a constant value of 0.8
			contours = measure.find_contours(z, step)
			# set the borders and alignment
			core_contour_x = contours[0][:, 1] - 10
			core_contour_x[core_contour_x >= self.image_width - 0.5] = self.image_width - 1
			core_contour_x[core_contour_x <= 0.5] = 0
			core_contour_y = contours[0][:, 0] - 10
			core_contour_y[core_contour_y >= self.image_height - 0.5] = self.image_height - 1
			core_contour_y[core_contour_y <= 0.5] = 0

		if save_result:
			raise NotImplemented
			cont = []
			for c in core_contour:
				if len(c) > 5:
					epsilon = 0.01 * cv2.arcLength(c, True)
					approx = cv2.approxPolyDP(c, epsilon, True).squeeze()
					approx = np.vstack((approx, approx[0, :]))
					cont.append(approx)
			return cont
		else:
			self.current_frame = frame
			self.fig.suptitle(f"Frame {frame}")
			self.ax1.clear()
			self.ax2.clear()

			# plotting original picture with excluded dots (red)
			self.ax1.imshow(original, cmap='gray')
			self.ax1.plot(self.excl_indices[:, 0], self.excl_indices[:, 1], '.', color='r')
			# debug plot
			self.ax2.imshow(image, cmap='gray')
			self.ax2.plot(x, y, '.', color='yellow', ms=1)
			# plot inside ellipse dots
			self.ax2.plot(x[rad_cc <= 1], y[rad_cc <= 1], '.', color='green', ms=1)
			ellip = Ellipse(xy=ell_center, width=width, height=height, angle=theta, lw=3, fill=False, edgecolor='w')
			self.ax2.add_artist(ellip)
			self.ax2.plot(ell_center[0], ell_center[1], 'x', ms=20, color='w')

			if len(CVcontours) < max_contour_number:
				for c in contours:
					# set the borders and alignment
					xx = c[:, 1] - 10
					xx[xx >= self.image_width - 0.5] = self.image_width - 1
					xx[xx <= 0.5] = 0
					yy = c[:, 0] - 10
					yy[yy >= self.image_height - 0.5] = self.image_height - 1
					yy[yy <= 0.5] = 0
					self.ax1.plot(xx, yy)
				self.ax1.plot(core_contour_x, core_contour_y, color='g', linewidth=3)

			self.ax2.set_title(f"Num: {len(CVcontours)} // dist {med_dist:.2f}")
			'''
			if False:
				p = path.Path(core_contour)
				x, y = np.meshgrid(np.arange(130), np.arange(174))  # make a canvas with coordinates
				x, y = x.flatten(), y.flatten()
				points = np.vstack((x, y)).T

				grid = p.contains_points(points)
				inside = points[grid]

				ios = np.mean(original[inside[:, 0], inside[:, 1]])
				self.ax2.plot(inside[:, 0], inside[:, 1], '.', color='pink', ms=3)
			'''
			self.canvas.draw()
			# waiting to see changes
			time.sleep(0.1)
			# flush the changes to the screen
			self.canvas.flush_events()

	def check_percentile(self):
		"""

		Returns:

		"""
		try:
			percentile = float(self.in_threshold_percentile.text())
			if 0.1 <= percentile <= 100:
				return percentile
		except Exception as e:
			print(e)
		self.status_text.setText("Percentile must be a number from 0.1 to 100")

	def save_contour(self):
		"""

		Returns:

		"""
		percentile = self.check_percentile()

		if percentile:
			self.status_text.setText("Saving results.... please wait")
			matframes = np.zeros((self.total_frames, ), dtype=np.object)

			for frame in range(0, self.total_frames - 1):
				conts = self.update_draws(frame, percentile, save_result=True)
				matframes[frame] = np.zeros((len(conts),), dtype=np.object)
				for cont_index, coords in enumerate(conts):
					matframes[frame][cont_index] = np.array(coords, dtype=np.int32)
			sio.savemat(f"{os.getcwd()}/output.mat", {'frames': matframes})

			self.status_text.setText(f"Successfully saved into {os.getcwd()}")

	def on_loop_draw(self):
		"""

		Returns:

		"""
		start = int(self.in_start_frame.text())
		end = int(self.in_end_frame.text())
		step = int(self.in_frame_stepsize.text())

		percentile = self.check_percentile()

		if percentile and 0 <= start < end < self.total_frames and step > 0:
			self.flag_loop_draw_stop = False
			self.btn_loop_draw_stop.setEnabled(True)

			for frame in range(start, end, step):
				if self.flag_loop_draw_stop:
					break
				self.in_start_frame.setText(str(frame))
				self.current_frame = frame
				self.update_draws(frame, percentile)
		else:
			self.status_text.setText("Check the START, END and STEPS values!")

	def stop_loop(self):
		self.flag_loop_draw_stop = True

	def on_hand_draw(self, step, sign=1):
		self.current_frame += sign * step

		percentile = self.check_percentile()

		if self.current_frame < 0:
			self.current_frame = 0
		if self.current_frame >= self.total_frames:
			self.current_frame = self.total_frames - 1

		if percentile:
			self.in_start_frame.setText(str(self.current_frame))
			self.update_draws(self.current_frame, percentile)

	def create_main_frame(self):
		# create the main plot
		self.main_frame = QWidget()
		# 1 set up the canvas
		self.fig = Figure(figsize=(10, 10), dpi=100)
		self.canvas = FigureCanvas(self.fig)
		self.canvas.setParent(self.main_frame)
		# add axes for plotting
		self.ax1 = self.fig.add_subplot(121)
		self.ax2 = self.fig.add_subplot(122)

		self.ax1.set_title("Original image")
		self.ax2.set_title(f"Normalized Gray + Blured")
		# self.ax3 = self.fig.add_subplot(223, sharex=self.ax1, sharey=self.ax1)
		# self.ax4 = self.fig.add_subplot(224, sharex=self.ax1, sharey=self.ax1)

		# 2 Create the navigation toolbar, tied to the canvas
		self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

		# 3 Layout with panel
		btn_panel_grid = QGridLayout()
		btn_panel_grid.setContentsMargins(0, 0, 0, 0)
		# GRID
		# PREPARE BLOCK
		self.btn_file = QPushButton("Open file")
		self.btn_file.clicked.connect(self.file_dialog)
		btn_panel_grid.addWidget(self.btn_file, 1, 0, 1, 2)

		self.label_fileinfo = QLabel("File info")
		self.label_fileinfo.setAlignment(Qt.AlignLeft | Qt.AlignTop)
		btn_panel_grid.addWidget(self.label_fileinfo, 2, 0, 1, 2)

		self.lbl_data_reshape = QLabel("Reshape data\n"
		                               "Height Width Frame Method")
		btn_panel_grid.addWidget(self.lbl_data_reshape, 3, 0, 1, 1)

		self.in_data_reshape = QLineEdit()
		self.in_data_reshape.setPlaceholderText("0 1 2 3")
		self.in_data_reshape.setEnabled(False)
		btn_panel_grid.addWidget(self.in_data_reshape, 4, 0, 1, 1)

		self.btn_reshape_data = QPushButton("Reshape")
		self.btn_reshape_data.clicked.connect(lambda x: self.reshape_data())
		self.btn_reshape_data.setEnabled(False)
		btn_panel_grid.addWidget(self.btn_reshape_data, 4, 1, 1, 1)

		self.label_object = QLabel("Image color")
		self.label_object.setAlignment(Qt.AlignLeading | Qt.AlignLeft | Qt.AlignVCenter)
		btn_panel_grid.addWidget(self.label_object, 5, 0, 1, 1)

		self.radio_original = QRadioButton(original_image)
		self.radio_original.setChecked(True)
		btn_panel_grid.addWidget(self.radio_original, 6, 0, 1, 1)

		self.radio_reversed = QRadioButton(reversed_image)
		btn_panel_grid.addWidget(self.radio_reversed, 7, 0, 1, 1)

		self.label_threshold_percentile = QLabel("Thr percentile")
		self.label_threshold_percentile.setAlignment(Qt.AlignLeading | Qt.AlignLeft | Qt.AlignVCenter)
		btn_panel_grid.addWidget(self.label_threshold_percentile, 8, 0, 1, 1)

		self.in_threshold_percentile = QLineEdit("95")
		btn_panel_grid.addWidget(self.in_threshold_percentile, 8, 1, 1, 1)

		self.label_method = QLabel("Method")
		self.label_method.setAlignment(Qt.AlignLeading | Qt.AlignLeft | Qt.AlignVCenter)
		btn_panel_grid.addWidget(self.label_method, 5, 1, 1, 1)

		self.box_method = QComboBox(self)
		for i in range(4):
			self.box_method.addItem(str(i))
		btn_panel_grid.addWidget(self.box_method, 6, 1, 1, 1)

		self.line = QFrame()
		self.line.setFrameShape(QFrame.VLine)
		self.line.setFrameShadow(QFrame.Sunken)
		btn_panel_grid.addWidget(self.line, 1, 2, 7, 1)
		""" END PREPARE BLOCK"""

		# AUTO BLOCK
		self.label_automatic = QLabel("Automatic view", alignment=Qt.AlignCenter)
		btn_panel_grid.addWidget(self.label_automatic, 1, 3, 1, 2)

		self.label_start_frame = QLabel("Start frame", alignment=Qt.AlignLeading | Qt.AlignLeft | Qt.AlignVCenter)
		btn_panel_grid.addWidget(self.label_start_frame, 2, 3, 1, 1)

		self.in_start_frame = QLineEdit("0")
		btn_panel_grid.addWidget(self.in_start_frame, 2, 4, 1, 1)

		self.label_end_frame = QLabel("End frame", alignment=Qt.AlignLeading | Qt.AlignLeft | Qt.AlignVCenter)
		btn_panel_grid.addWidget(self.label_end_frame, 3, 3, 1, 1)

		self.in_end_frame = QLineEdit("0")
		btn_panel_grid.addWidget(self.in_end_frame, 3, 4, 1, 1)

		self.label_stepsize_frame = QLabel("Step size frame", alignment=Qt.AlignLeading | Qt.AlignLeft | Qt.AlignVCenter)
		btn_panel_grid.addWidget(self.label_stepsize_frame, 4, 3, 1, 1)

		self.in_frame_stepsize = QLineEdit("1")
		self.in_frame_stepsize.setValidator(QIntValidator(0, 100))
		btn_panel_grid.addWidget(self.in_frame_stepsize, 4, 4, 1, 1)

		self.btn_loop_draw = QPushButton("Start loop draw")
		self.btn_loop_draw.clicked.connect(lambda x: self.on_loop_draw())
		self.btn_loop_draw.setEnabled(False)
		btn_panel_grid.addWidget(self.btn_loop_draw, 5, 3, 1, 2)

		self.btn_loop_draw_stop = QPushButton("Stop loop draw")
		self.btn_loop_draw_stop.clicked.connect(lambda x: self.stop_loop())
		self.btn_loop_draw_stop.setEnabled(False)
		btn_panel_grid.addWidget(self.btn_loop_draw_stop, 6, 3, 1, 2)

		self.btn_save_results = QPushButton("Save results")
		self.btn_save_results.clicked.connect(lambda x: self.save_contour())
		self.btn_save_results.setEnabled(False)
		btn_panel_grid.addWidget(self.btn_save_results, 7, 3, 1, 2)

		self.line = QFrame()
		self.line.setFrameShape(QFrame.VLine)
		self.line.setFrameShadow(QFrame.Sunken)
		btn_panel_grid.addWidget(self.line, 1, 5, 7, 1)
		""" END AUTO BLOCK """

		# MANUAL BLOCK
		self.lbl_manual = QLabel("Manual view", alignment=Qt.AlignCenter)
		btn_panel_grid.addWidget(self.lbl_manual, 1, 6, 1, 3)

		self.lbl_framestep = QLabel("Frame step", alignment=Qt.AlignCenter)
		btn_panel_grid.addWidget(self.lbl_framestep, 2, 6, 1, 3)

		in_frame_step = QLineEdit("1", alignment=Qt.AlignCenter)
		in_frame_step.setValidator(QIntValidator(1, 100))
		btn_panel_grid.addWidget(in_frame_step, 3, 7, 1, 1)

		left_step = lambda x: self.on_hand_draw(int(in_frame_step.text()), sign=-1)
		right_step = lambda x: self.on_hand_draw(int(in_frame_step.text()))

		self.btn_frame_left = QPushButton("<<")
		self.btn_frame_left.clicked.connect(left_step)
		self.btn_frame_left.setEnabled(False)
		btn_panel_grid.addWidget(self.btn_frame_left, 3, 6, 1, 1)

		self.btn_frame_right = QPushButton(">>")
		self.btn_frame_right.clicked.connect(right_step)
		self.btn_frame_right.setEnabled(False)
		btn_panel_grid.addWidget(self.btn_frame_right, 3, 8, 1, 1)

		# 4 combne all in the structure
		vbox = QVBoxLayout()
		vbox.addWidget(self.canvas)
		vbox.addWidget(self.mpl_toolbar)
		vbox.addLayout(btn_panel_grid)

		self.main_frame.setLayout(vbox)
		self.setCentralWidget(self.main_frame)

	def create_status_bar(self):
		self.status_text = QLabel("Waiting a file...")
		self.statusBar().addWidget(self.status_text, stretch=1)


def main():
	app = QApplication(sys.argv)
	form = AppForm()
	form.resize(900, 900)

	form.show()
	app.exec_()


if __name__ == "__main__":
	main()
