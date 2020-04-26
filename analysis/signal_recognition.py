import os
import sys
import cv2
import time
import h5py
import numpy as np
import scipy.io as sio
from fastkde import fastKDE
from matplotlib.figure import Figure
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
		self.setWindowTitle('Computer Vision Analyzer v2.5 (NcN lab product 2020)')
		self.create_main_frame()
		self.create_status_bar()

		self.dat = None
		self.current_frame = 0
		self.excl_x = None
		self.excl_y = None
		self.maxDIFF = None
		self.minDIFF = None
		self.analyse_type = original_image

	def open_file(self, path):
		"""
		Try to read data from files with different types
		Args:
			path (str): file path
		Returns:
			tuple : shape of the data
		"""
		self.variables = {}

		try:
			for k, v in sio.loadmat(path).items():
				# get only 4-dim data
				if len(np.shape(v)) == 4:
					self.variables[k] = v[:]
		except NotImplementedError:
			with h5py.File(path, 'r') as file:
				for k, v in file.items():
					# get only 4-dim data
					if len(np.shape(v)) == 4:
						self.variables[k] = v[:]
		except:
			ValueError('Could not read the file at all...')

	def reshape_data(self):
		"""
		Transpose the multi-dimensional matrix if need
		"""
		# get new shape as tuple
		new_order = tuple(map(int, self.in_data_reshape.text().split()))
		# reshape data to the new shape
		self.dat = np.transpose(self.dat, new_order)
		self.im_height, self.im_width, self.total_frames, methods_num = self.dat.shape
		# disable buttons of reshaping
		self.in_data_reshape.setEnabled(False)
		self.btn_reshape_data.setEnabled(False)
		# init frames in GUI form
		self.in_start_frame.setValidator(QIntValidator(0, self.total_frames - 1))
		self.in_end_frame.setValidator(QIntValidator(0, self.total_frames - 1))
		self.in_end_frame.setText(str(self.total_frames - 1))
		# update status
		self.status_text.setText(f"Data was reshaped to {self.dat.shape}")

		self.box_method.clear()
		for i in range(methods_num):
			self.box_method.addItem(str(i))

	def file_dialog(self):
		"""
		Invoke PyQT file dialog with unblocking buttons
		"""
		fname = QFileDialog.getOpenFileName(self, 'Open file', "", "MAT file (*.mat)")
		# if exists
		if fname[0]:
			self.box_variable.clear()
			# delete old data if exists
			if self.dat is not None:
				del self.dat
			self.dat = None

			# prepare the data
			self.status_text.setText("Unpack .mat file... Please wait")
			QApplication.processEvents()
			QApplication.processEvents()

			self.filepath = fname[0]
			self.open_file(fname[0])
			self.status_text.setText(f".mat file is unpacked ({self.filepath})")

			# based on data set the possible variables
			self.box_variable.setEnabled(True)
			for k in self.variables.keys():
				self.box_variable.addItem(str(k))

	def choose_variable(self):
		""" Invoked if text in QComboBox is changed """
		# get the user's choose
		var = self.box_variable.currentText()
		# get the data by name
		self.dat = self.variables[var]
		# meta info
		data_shape = self.dat.shape
		str_format = len(data_shape) * '{:<5}'
		self.label_fileinfo.setText(f"Shape: {str_format.format(*data_shape)}\n"
		                            f"Index: {str_format.format(*list(range(4)))}")
		self.label_fileinfo.setFont(QFont("Courier New"))

		self.box_method.clear()
		# unblock buttons
		for obj in [self.btn_save_results, self.btn_loop_draw, self.btn_frame_right,
		            self.btn_frame_left, self.btn_reshape_data, self.in_data_reshape,
		            self.box_method, self.in_data_filter, self.btn_filter_data]:
			obj.setEnabled(True)
		self.status_text.setText(f"{var} is chosen {data_shape}")

	def filter_exclude(self):
		"""
		Get a coords of points that have very small changes till the all recordings
		"""
		methodic = int(self.box_method.currentText())
		input_q3 = self.check_input(self.in_data_filter, borders=[50, 100])
		# find the mean value of all frames per pixel
		diff = np.mean(self.dat[:, :, :, methodic], axis=2)
		# find the most unchanged data (they will have the highest values after mean)
		q1, m, q3 = np.percentile(diff, q=(100 - input_q3, 50, input_q3))
		# take coordinates of the points where their values are fliers
		row, col = np.indices(diff.shape)
		diff = diff.ravel()
		# form a coordinates array of exluded points
		excl_indices = np.stack((col.ravel(), row.ravel()), axis=1)[(diff >= q3) | (diff <= q1)]
		self.excl_x, self.excl_y = excl_indices[:, 0], excl_indices[:, 1]
		# make excluded points values as a median
		for frame in range(self.total_frames):
			self.dat[self.excl_y, self.excl_x, frame, methodic] = np.percentile(self.dat[:, :, frame, methodic], input_q3)

		self.maxDIFF = q3
		self.minDIFF = q1

		# plot the filtering result
		self.ax1.clear()
		self.ax1.imshow(self.dat[:, :, 0, methodic], cmap='gray')
		self.ax1.plot(self.excl_x, self.excl_y, '.', color='r', ms=1)
		self.canvas.draw()
		self.canvas.flush_events()

	@staticmethod
	def polygon_area(coords):
		x, y = coords[:, 0], coords[:, 1]
		return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

	@staticmethod
	def align_coord(coords, border):
		coords[coords >= border - 0.5] = border - 1
		coords[coords <= 0.5] = 0
		return coords

	def update_draws(self, frame, threshold_percentile, save_result=False):
		"""
		Calculate contours and draw them on the axis
		Args:
			frame (int): index of the frame
			threshold_percentile (float): Q3 percentile for finding non-normal values
			save_result (bool): flag for skipping drawing if we want just save results
		Returns:
			tuple : x and y coords of the contour if 'save_result' is true
		"""
		a = time.time()

		max_contour = None
		# get the current methodic
		methodic = int(self.box_method.currentText())
		max_fragmentation = int(self.check_input(self.in_max_fragmentation, borders=[3, 80]))

		# get an original data
		original = self.dat[:, :, frame, methodic]
		# normalize data from 0 to 255 with dynamic borders (min and max). It mades grayscale cmap
		image = ((original - original.min()) * (1 / (original.max() - original.min()) * 255)).astype('uint8')
		# reverse colors if epilepsy radio button checked
		if self.radio_reversed.isChecked():
			image = 255 - image
		# blur the image to smooth very light pixels
		image = cv2.medianBlur(image, 5)
		# set the dynamic thresh value
		zdiff = np.mean(self.dat[:, :, frame:frame + 10, methodic], axis=2)
		mid = np.median(zdiff)
		if mid >= 0:
			coef = abs(mid / self.maxDIFF)
		else:
			coef = -abs(mid / self.minDIFF)
		# self.magnitudes = np.zeros(shape=(self.image_height, self.image_width))
		# for xi in range(self.image_width):
		# 	for yi in range(self.image_height):
		# 		self.magnitudes[yi, xi] = np.linalg.norm(self.dat[yi, xi, :600, methodic])
		new_threshold = threshold_percentile + threshold_percentile * 1.2 * coef
		if new_threshold > 99:
			new_threshold = 99
		if new_threshold < 50:
			new_threshold = 50

		thresh_value = np.percentile(image.ravel(), new_threshold)
		# get coordinates of points which are greater than thresh value
		y, x = np.where(image >= thresh_value)
		# calc raw CV contours to decide -- search contour or not
		mask = np.zeros(shape=image.shape)
		mask[y, x] = 255
		mask = mask.astype('uint8')
		#
		_, thresh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
		# transform morphology of the mask
		morph_kernel = np.ones((3, 3), np.uint8)
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel)
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, morph_kernel)
		# get the contour of the mask
		*im2, CVcontours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# only if number of CV not so big (more fragmentation -- more confidence that there are no epilepsy contour)

		if len(CVcontours) <= max_fragmentation:
			# get a KDE function values based on found points above XY meshgrid
			PDF, (vx, vy) = fastKDE.pdf(x, y)
			# find the contour by maximal area
			max_cont = max(self.ax1.contour(vx, vy, PDF, levels=1).allsegs[1], key=self.polygon_area)
			# limit coordinates within the border
			max_contour = (self.align_coord(max_cont[:, 0], self.im_width),
			               self.align_coord(max_cont[:, 1], self.im_height))

		b = time.time()
		# print(f"{b - a:.3f}s")
		if save_result:
			return max_contour
		else:
			self.current_frame = frame
			self.fig.suptitle(f"Frame {frame}")
			self.ax1.clear()
			self.ax2.clear()
			self.ax3.clear()
			self.ax4.clear()
			# plotting original picture with excluded dots (red) and contour
			self.ax1.set_title(f"Original image + contour")
			self.ax1.imshow(original, cmap='gray')
			self.ax1.plot(self.excl_x, self.excl_y, '.', color='r', ms=1)
			if len(CVcontours) <= max_fragmentation:
				self.ax1.plot(max_contour[0], max_contour[1], color='g', lw=3)
			# found points
			self.ax2.set_title(f"Fragmented contours: {len(CVcontours)}")
			self.ax2.imshow(image, cmap='gray')
			# all points plotting, but after recoloring they will mean the points outside of ellipse
			self.ax2.plot(x, y, '.', color='#00AA00', ms=1)
			# demonstrate forecast (mean) of 10 frames
			self.ax3.set_title(f"Forecast")
			self.ax3.imshow(zdiff)
			# save axis plot
			# extent = self.ax3.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
			# self.fig.savefig(f'/home/alex/example/{frame}.jpg', format='jpg')
			self.ax4.set_title(f"Debug info")
			self.ax4.text(0, 0, f"Median frame value: {mid:.1f}\n"
			                    f"Max global: {self.maxDIFF:.1f}\n"
			                    f"Min global: {self.minDIFF:.1f}\n"
			                    f"Coef: {1 + coef:.3f} x {threshold_percentile:.1f}\n"
			                    f"New threshold: {new_threshold:.1f}")
			self.fig.tight_layout()
			self.canvas.draw()
			# waiting to see changes
			time.sleep(0.01)
			# flush the changes to the screen
			self.canvas.flush_events()

	def check_input(self, input_value, borders=(-np.inf, np.inf)):
		"""
		Checking the input value on validity
		Returns:
			float : converted from string a value
		Raises:
			ValueError : value not in borders
			Exception : cannot convert from string
		"""
		try:
			value = float(input_value.text())
			if borders[0] <= value <= borders[1]:
				return value
			else:
				self.status_text.setText(f"Value ({value}) not in borders")
				raise ValueError(f"Value ({value}) not in borders")
		except Exception as e:
			print(e)
		self.status_text.setText(f"Value must be a number from {borders[0]} to {borders[1]}")

	def save_contour(self):
		"""
		Converting numpy arrays of contours to a mat file
		"""
		start = int(self.check_input(self.in_start_frame))
		end = int(self.check_input(self.in_end_frame))
		step = int(self.check_input(self.in_frame_stepsize))
		threshold_percentile = self.check_input(self.in_threshold_percentile, borders=[0.1, 100])
		# check if value is correct
		if threshold_percentile and 0 <= start < end < self.total_frames and step > 0:
			self.status_text.setText("Saving results.... please wait")
			# prepare array of objects per frame
			matframes = np.zeros((self.total_frames, ), dtype=np.object)
			# init by void arrays
			for frame in range(self.total_frames):
				matframes[frame] = np.array([], dtype=np.int32)
			# get data per frame and fill the 'matframes'
			for index, frame in enumerate(range(start, end, step)):
				contour = self.update_draws(frame, threshold_percentile, save_result=True)
				if contour is not None:
					matframes[frame] = np.array(contour, dtype=np.int32)
				QApplication.processEvents()
				QApplication.processEvents()
				self.status_text.setText(f"Processed {index / len(range(start, end, step)) * 100:.2f} %")
			# save data into mat format
			filepath = os.path.dirname(self.filepath)
			filename = os.path.basename(self.filepath)

			sio.savemat(f"{filepath}/prepared_{filename}", {'frames': matframes})
			# you are beautiful :3
			self.status_text.setText(f"Successfully saved into {filepath}/prepared_{filename}")

	def on_loop_draw(self):
		"""
		Automatic drawing data in loop by user panel settings
		"""
		start = int(self.check_input(self.in_start_frame))
		end = int(self.check_input(self.in_end_frame))
		step = int(self.check_input(self.in_frame_stepsize))
		threshold_percentile = self.check_input(self.in_threshold_percentile, borders=[0.1, 100])

		if threshold_percentile and 0 <= start < end < self.total_frames and step > 0:
			self.flag_loop_draw_stop = False
			self.btn_loop_draw_stop.setEnabled(True)

			for frame in range(start, end, step):
				if self.flag_loop_draw_stop:
					break
				self.in_start_frame.setText(str(frame))
				self.current_frame = frame
				self.update_draws(frame, threshold_percentile)
		else:
			self.status_text.setText("Check the START, END and STEPS values!")

	def stop_loop(self):
		self.flag_loop_draw_stop = True

	def on_hand_draw(self, step, sign=1):
		"""
		Manual drawing frames
		Args:
			step (int): stepsize of left/right moving
			sign (int): -1 or 1 show the side moving (-1 is left, 1 is right)
		"""
		self.current_frame += sign * step

		threshold_percentile = self.check_input(self.in_threshold_percentile, borders=[0.1, 100])

		if self.current_frame < 0:
			self.current_frame = 0
		if self.current_frame >= self.total_frames:
			self.current_frame = self.total_frames - 1

		if threshold_percentile:
			self.in_start_frame.setText(str(self.current_frame))
			self.update_draws(self.current_frame, threshold_percentile)

	def create_main_frame(self):
		# create the main plot
		self.main_frame = QWidget()
		# 1 set up the canvas
		self.fig = Figure(figsize=(10, 15), dpi=100)
		self.canvas = FigureCanvas(self.fig)
		self.canvas.setParent(self.main_frame)
		# add axes for plotting
		self.ax1 = self.fig.add_subplot(221)
		self.ax2 = self.fig.add_subplot(222, sharex=self.ax1, sharey=self.ax1)
		self.ax3 = self.fig.add_subplot(223, sharex=self.ax1, sharey=self.ax1)
		self.ax4 = self.fig.add_subplot(224)

		self.ax1.set_title(f"Original image + contour")
		self.ax2.set_title(f"Fragmented contours")
		self.ax3.set_title(f"Forecast")
		self.ax4.set_title(f"Debug info")

		# 2 Create the navigation toolbar, tied to the canvas
		self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

		# 3 Layout with panel
		btn_panel_grid = QGridLayout()
		btn_panel_grid.setContentsMargins(0, 0, 0, 0)

		''' PREPARE BLOCK '''
		# FILE
		self.btn_file = QPushButton("Open file")
		self.btn_file.clicked.connect(self.file_dialog)
		btn_panel_grid.addWidget(self.btn_file, 1, 0, 1, 1)

		# VARIABLE
		self.box_variable = QComboBox(self)
		btn_panel_grid.addWidget(self.box_variable, 1, 1, 1, 1)
		self.box_variable.currentTextChanged.connect(lambda x: self.choose_variable())
		self.box_variable.setEnabled(False)

		self.label_fileinfo = QLabel("File info")
		self.label_fileinfo.setAlignment(Qt.AlignLeft | Qt.AlignTop)
		btn_panel_grid.addWidget(self.label_fileinfo, 2, 0, 1, 2)

		# RESHAPE
		self.lbl_data_reshape = QLabel("Reshape data\nHeight Width Frame Method")
		btn_panel_grid.addWidget(self.lbl_data_reshape, 3, 0, 1, 1)

		self.in_data_reshape = QLineEdit()
		self.in_data_reshape.setText("0 1 2 3")
		btn_panel_grid.addWidget(self.in_data_reshape, 4, 0, 1, 1)
		self.in_data_reshape.setEnabled(False)

		self.btn_reshape_data = QPushButton("Reshape")
		self.btn_reshape_data.clicked.connect(lambda x: self.reshape_data())
		btn_panel_grid.addWidget(self.btn_reshape_data, 4, 1, 1, 1)
		self.btn_reshape_data.setEnabled(False)

		# METHOD
		self.box_method = QComboBox(self)
		btn_panel_grid.addWidget(self.box_method, 5, 0, 1, 1)
		self.box_method.setEnabled(False)

		self.label_method = QLabel("Method")
		self.label_method.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
		btn_panel_grid.addWidget(self.label_method, 5, 1, 1, 1)

		# FILTER
		self.in_data_filter = QLineEdit("99.5")
		self.in_data_filter.setPlaceholderText("Filter percentile 50-100")
		btn_panel_grid.addWidget(self.in_data_filter, 6, 0, 1, 1)
		self.in_data_filter.setEnabled(False)

		self.btn_filter_data = QPushButton("Filter")
		self.btn_filter_data.clicked.connect(lambda x: self.filter_exclude())
		btn_panel_grid.addWidget(self.btn_filter_data, 6, 1, 1, 1)
		self.btn_filter_data.setEnabled(False)

		# IMAGE COLOR
		self.label_object = QLabel("Image color")
		self.label_object.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
		btn_panel_grid.addWidget(self.label_object, 7, 0, 1, 1)

		self.radio_original = QRadioButton(original_image)
		self.radio_reversed = QRadioButton(reversed_image)
		self.radio_original.setChecked(True)
		btn_panel_grid.addWidget(self.radio_original, 7, 1, 1, 1)
		btn_panel_grid.addWidget(self.radio_reversed, 8, 1, 1, 1)

		# THRESHOLD
		self.label_threshold_percentile = QLabel("Threshold percentile")
		self.label_threshold_percentile.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
		btn_panel_grid.addWidget(self.label_threshold_percentile, 9, 0, 1, 1)

		self.in_threshold_percentile = QLineEdit("95")
		self.in_threshold_percentile.setPlaceholderText("50-100")
		btn_panel_grid.addWidget(self.in_threshold_percentile, 9, 1, 1, 1)

		# FRAGMENTATION
		self.label_max_fragmentation = QLabel("Max fragmentation")
		self.label_max_fragmentation.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
		btn_panel_grid.addWidget(self.label_max_fragmentation, 10, 0, 1, 1)

		self.in_max_fragmentation = QLineEdit("15")
		self.in_max_fragmentation.setPlaceholderText("3-50")
		btn_panel_grid.addWidget(self.in_max_fragmentation, 10, 1, 1, 1)

		self.line = QFrame()
		self.line.setFrameShape(QFrame.VLine)
		self.line.setFrameShadow(QFrame.Sunken)
		btn_panel_grid.addWidget(self.line, 1, 2, 10, 1)
		''' END PREPARE BLOCK '''

		''' BEGIN AUTO BLOCK '''
		self.label_automatic = QLabel("Automatic view")
		self.label_automatic.setAlignment(Qt.AlignCenter)

		btn_panel_grid.addWidget(self.label_automatic, 1, 3, 1, 2)

		self.label_start_frame = QLabel("Start frame")
		self.label_start_frame.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
		btn_panel_grid.addWidget(self.label_start_frame, 2, 3, 1, 1)

		self.in_start_frame = QLineEdit("0")
		btn_panel_grid.addWidget(self.in_start_frame, 2, 4, 1, 1)

		self.label_end_frame = QLabel("End frame")
		self.label_end_frame.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
		btn_panel_grid.addWidget(self.label_end_frame, 3, 3, 1, 1)

		self.in_end_frame = QLineEdit("0")
		btn_panel_grid.addWidget(self.in_end_frame, 3, 4, 1, 1)

		self.label_stepsize_frame = QLabel("Step size frame")
		self.label_stepsize_frame.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
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

		''' MANUAL BLOCK '''
		self.lbl_manual = QLabel("Manual view")
		self.lbl_manual.setAlignment(Qt.AlignCenter)
		btn_panel_grid.addWidget(self.lbl_manual, 1, 6, 1, 3)

		self.lbl_framestep = QLabel("Frame step")
		self.lbl_framestep.setAlignment(Qt.AlignCenter)
		btn_panel_grid.addWidget(self.lbl_framestep, 2, 6, 1, 3)

		in_frame_step = QLineEdit("1")
		in_frame_step.setAlignment(Qt.AlignCenter)
		in_frame_step.setValidator(QIntValidator(1, 100))
		btn_panel_grid.addWidget(in_frame_step, 3, 7, 1, 1)

		left_step = lambda x: self.on_hand_draw(int(self.check_input(in_frame_step)), sign=-1)
		right_step = lambda x: self.on_hand_draw(int(self.check_input(in_frame_step)))

		self.btn_frame_left = QPushButton("<<")
		self.btn_frame_left.clicked.connect(left_step)
		self.btn_frame_left.setEnabled(False)
		btn_panel_grid.addWidget(self.btn_frame_left, 3, 6, 1, 1)

		self.btn_frame_right = QPushButton(">>")
		self.btn_frame_right.clicked.connect(right_step)
		self.btn_frame_right.setEnabled(False)
		btn_panel_grid.addWidget(self.btn_frame_right, 3, 8, 1, 1)
		''' END MANUAL BLOCK '''

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
