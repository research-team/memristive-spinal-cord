import os
import sys
import cv2
import time
import h5py
import numpy as np
import scipy.io as sio
from fastkde import fastKDE
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator, QFont
from PyQt5.QtWidgets import QPushButton, QLabel, QRadioButton, QLineEdit, QComboBox, QMessageBox
from PyQt5.QtWidgets import QGridLayout, QFileDialog, QApplication, QMainWindow, QWidget, QVBoxLayout, QFrame

original_image = "original"
reversed_image = "reversed"

class PlotWindow(QMainWindow):
	def __init__(self, parent=None):
		super(PlotWindow, self).__init__(parent)
		self.setWindowTitle("Visualization window")
		self.interactive_dist = None
		self._polygon_points = None
		# 1 set up the canvas
		self.fig = Figure(figsize=(8, 8), dpi=100)
		self.canvas = FigureCanvas(self.fig)
		self.canvas.setParent(self)
		# add axes for plotting
		self.ax1 = self.fig.add_subplot(221)
		self.ax2 = self.fig.add_subplot(222, sharex=self.ax1, sharey=self.ax1)
		self.ax3 = self.fig.add_subplot(223, sharex=self.ax1, sharey=self.ax1)
		#
		self.ax1.set_title("Interactive area")
		self.ax2.set_title("Threshold: void\n"
		                   "Fragments: void")
		self.ax3.set_title("Result output")
		#
		self.init_interactive()

		# 2 Create the navigation toolbar, tied to the canvas
		self.mpl_toolbar = NavigationToolbar(self.canvas, self)

		layout = QVBoxLayout()
		layout.addWidget(self.canvas)
		layout.addWidget(self.mpl_toolbar)
		widget = QWidget()
		widget.setLayout(layout)
		self.setCentralWidget(widget)

		self.resize(900, 900)
		self.fig.tight_layout()
		self.show()

	def init_interactive(self):
		self._polygon_points = []
		self._line, self._dragging_point = None, None
		self.canvas.mpl_connect('button_press_event', self.on_click)
		self.canvas.mpl_connect('button_release_event', self.on_release)
		self.canvas.mpl_connect('motion_notify_event', self.on_motion)

	def update_plot(self):
		if not self._polygon_points:
			self._line.set_data([], [])
		else:
			x, y = zip(*self._polygon_points)
			# Add new plot
			if not self._line:
				(self._line, ) = self.ax1.plot(x, y, "r", marker="o", markersize=8, zorder=10)
			# Update current plot
			else:
				x, y = list(x), list(y)
				self._line.set_data(x + [x[0]], y + [y[0]])
		self.fig.tight_layout()
		self.canvas.draw()

	def add_point(self, event):
		self._polygon_points.append((event.xdata, event.ydata))

	def remove_point(self, point):
		if point in self._polygon_points:
			self._polygon_points.remove(point)

	def find_neighbor_point(self, event):
		""" Find point around mouse position
		:rtype: ((int, int)|None)
		:return: (x, y) if there are any point around mouse else None
		"""
		if self._polygon_points:
			nx, ny = min(self._polygon_points, key=lambda p: np.hypot(event.xdata - p[0], event.ydata - p[1]))
			if np.hypot(event.xdata - nx, event.ydata - ny) < self.interactive_dist:
				return nx, ny
		return None

	@staticmethod
	def isBetween(pA, pB, p0):
		p = pB
		p0A = np.hypot(p0[0] - pA[0], p0[1] - pA[1])
		p0B = np.hypot(p0[0] - pB[0], p0[1] - pB[1])
		if p0A < p0B:
			p = pA

		dotproduct = (p0[0] - pA[0]) * (pB[0] - pA[0]) + (p0[1] - pA[1]) * (pB[1] - pA[1])
		if dotproduct < 0:
			return None, None

		squaredlengthba = (pB[0] - pA[0]) * (pB[0] - pA[0]) + (pB[1] - pA[1]) * (pB[1] - pA[1])
		if dotproduct > squaredlengthba:
			return None, None

		return p0B + p0A, p

	def on_click(self, event):
		""" callback method for mouse click event
		:type event: MouseEvent
		"""
		# left click
		if event.inaxes in [self.ax1] and event.button == 1:
			point = self.find_neighbor_point(event)
			p_next = None
			p0 = (event.xdata, event.ydata)
			mind = np.inf

			if len(self._polygon_points) >= 3:
				a = self._polygon_points + [self._polygon_points[0]]
				for p1, p2 in zip(a, a[1:]):
					d, p = self.isBetween(p1, p2, p0)
					if d and d < mind:
						mind = d
						p_next = p2
			if point:
				self._dragging_point = point
			elif p_next:
				self._polygon_points.insert(self._polygon_points.index(p_next), p0)
			else:
				self.add_point(event)
			self.update_plot()
		# mid click
		elif event.inaxes in [self.ax1] and event.button == 2:
			self._polygon_points = []
			self.update_plot()
		elif event.inaxes in [self.ax1] and event.button == 3:
			point = self.find_neighbor_point(event)
			if point:
				self.remove_point(point)
				self.update_plot()

	def on_release(self, event):
		""" callback method for mouse release event
		:type event: MouseEvent
		"""
		if event.inaxes in [self.ax1] and event.button == 1 and self._dragging_point:
			self._dragging_point = None
			self.update_plot()

	def on_motion(self, event):
		""" callback method for mouse motion event
		:type event: MouseEvent
		"""
		if not self._dragging_point:
			return
		if event.xdata is None or event.ydata is None:
			return
		# get index of the previous dragged point
		index = self._polygon_points.index(self._dragging_point)
		# set new point
		self._dragging_point = (event.xdata, event.ydata)
		# update previous point
		self._polygon_points[index] = self._dragging_point
		self.update_plot()


class Application(QMainWindow):
	def __init__(self, parent=None):
		QMainWindow.__init__(self, parent)
		self.setWindowTitle('Computer Vision Analyzer v2.7 (NcN lab product 2020)')
		self.create_main_frame()
		self.create_status_bar()

		self.dat = None
		self.current_frame = 0
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
			QMessageBox.about(self, f"Error", 'Could not read the file at all...')

	def reshape_data(self):
		"""
		Transpose the multi-dimensional matrix if need
		"""
		# get new shape as tuple
		new_order = tuple(map(int, self.in_data_reshape.text().split()))
		# reshape data to the new shape
		self.dat = np.transpose(self.dat, new_order)
		self.im_height, self.im_width, self.total_frames, methods_num = self.dat.shape
		self.im_shape = (self.im_height, self.im_width)
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

		self.plot_frame.interactive_dist = max(self.im_width, self.im_height) * 0.1

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
			self.open_file(self.filepath)
			self.status_text.setText(f".mat file is unpacked ({self.filepath})")

			# based on data set the possible variables
			self.box_variable.setEnabled(True)
			for k in self.variables.keys():
				self.box_variable.addItem(str(k))

	def choose_variable(self):
		""" Invoked if text in QComboBox is changed """
		# get the user's choose
		var = self.box_variable.currentText()
		if var != '':
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
			            self.box_method, self.btn_filter_data]:
				obj.setEnabled(True)
			self.status_text.setText(f"{var} is chosen {data_shape}")

	def filter_exclude(self):
		"""

		"""
		polygon = Polygon(self.plot_frame._polygon_points)
		row, col = np.indices(self.im_shape)
		grid = zip(col.ravel(), row.ravel())
		# form a coordinates array of exluded points
		self.mask_inside = np.array(list(map(polygon.contains, map(Point, grid)))).reshape(self.im_shape)
		self.true_y, self.true_x = np.where(self.mask_inside)

	def method_onchange(self):
		text = self.box_method.currentText()
		if text != '':
			methodic = int(text)

			# self.ax1.clear()
			self.plot_frame.ax1.imshow(np.mean(self.dat[:, :, :, methodic], axis=2), zorder=-10, cmap='gray')
			if self.plot_frame._line:
				self.plot_frame.update_plot()
			else:
				self.plot_frame.canvas.draw()

	@staticmethod
	def polygon_area(coords):
		x, y = coords[:, 0], coords[:, 1]
		return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

	@staticmethod
	def align_coord(coords, border):
		coords[coords >= border - 0.5] = border - 1
		coords[coords <= 0.5] = 0
		return coords

	def update_draws(self, frame, methodic, max_fragmentation, save_result=False):
		"""
		Calculate contours and draw them on the axis
		Args:
			frame (int): index of the frame
			save_result (bool): flag for skipping drawing if we want just save results
		Returns:
			tuple : x and y coords of the contour if 'save_result' is true
		"""
		# get an original data
		original = self.dat[:, :, frame, methodic]
		data_inside = original[self.mask_inside]
		min_inside = np.min(data_inside)
		max_inside = np.max(data_inside)
		# normalize data from 0 to 255 with dynamic borders (min and max). It mades grayscale cmap
		image = ((original - min_inside) / (max_inside - min_inside) * 255).astype('uint8')
		# reverse colors if epilepsy radio button checked
		if self.radio_reversed.isChecked():
			image = 255 - image
			original = 255 - original
		image[~self.mask_inside] = 0
		# blur the image to smooth very light pixels
		image = cv2.medianBlur(image, 5)
		# set the dynamic thresh value
		in_mask_image = image[self.mask_inside]

		# first, rude iteration
		new_threshold = 99
		morph_kernel = np.ones((3, 3), np.uint8)
		mask = np.zeros(shape=image.shape, dtype='uint8')
		while True:
			thresh_value = np.percentile(in_mask_image, new_threshold)
			# get coordinates of points which are greater than thresh value
			y, x = np.where(image >= thresh_value)
			# calc raw CV contours to decide -- search contour or not
			tmpmask = np.array(mask, copy=True)
			tmpmask[y, x] = 255
			#
			_, thresh = cv2.threshold(tmpmask, 200, 255, cv2.THRESH_BINARY)
			# transform morphology of the mask
			thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel)
			thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, morph_kernel)
			# get the contour of the mask
			*im2, CVcontours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			# only if number of CV not so big (more fragmentation -- more confidence that there are no epilepsy contour)
			if len(CVcontours) > max_fragmentation:
				break
			new_threshold -= 2
			if new_threshold < 50:
				new_threshold += 2
				break
		# second, more preciescly iteration
		while True:
			new_threshold += 0.2
			if new_threshold > 99:
				new_threshold -= 0.2
				break
			thresh_value = np.percentile(in_mask_image, new_threshold)
			# get coordinates of points which are greater than thresh value
			y, x = np.where(image >= thresh_value)
			# calc raw CV contours to decide -- search contour or not
			tmpmask = np.array(mask, copy=True)
			tmpmask[y, x] = 255
			#
			_, thresh = cv2.threshold(tmpmask, 200, 255, cv2.THRESH_BINARY)
			# transform morphology of the mask
			thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel)
			thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, morph_kernel)
			# get the contour of the mask
			*im2, CVcontours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			# only if number of CV not so big (more fragmentation -- more confidence that there are no epilepsy contour)
			if len(CVcontours) <= max_fragmentation:
				break
		#
		max_contour = None
		if len(x) > 10:
			# get a KDE function values based on found points above XY meshgrid
			PDF, (vx, vy) = fastKDE.pdf(x, y)
			# find the contour by maximal area
			max_cont = max(self.plot_frame.ax3.contour(vx, vy, PDF, levels=1, alpha=0).allsegs[1], key=self.polygon_area)
			# limit coordinates within the border
			max_contour = (self.align_coord(max_cont[:, 0], self.im_width),
			               self.align_coord(max_cont[:, 1], self.im_height))
		#
		if save_result:
			return max_contour
		else:
			self.current_frame = frame
			self.plot_frame.fig.suptitle(f"Frame {frame}")
			self.plot_frame.ax2.clear()
			self.plot_frame.ax3.clear()

			self.plot_frame.ax1.set_title("Interactive area")
			self.plot_frame.ax2.set_title(f"Threshold: {new_threshold:.1f}\n"
			                              f"Fragments: {len(CVcontours)}")
			self.plot_frame.ax3.set_title("Result output")

			# plotting original picture with excluded dots (red) and contour
			self.plot_frame.ax2.imshow(image, cmap='gray')
			self.plot_frame.ax2.plot(x, y, '.', color='#00AA00', ms=1)
			if max_contour:
				self.plot_frame.ax2.plot(max_contour[0], max_contour[1], color='g', lw=3)
			self.plot_frame.ax3.imshow(original, cmap='gray')
			if max_contour:
				self.plot_frame.ax3.plot(max_contour[0], max_contour[1], color='r', lw=3)
			# save axis plot
			# extent = self.ax3.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
			# self.fig.savefig(f'/home/alex/example/{frame}.jpg', format='jpg')
			# self.plot_frame.ax4.text(0, 0, f"New threshold: {new_threshold:.1f}\n"
			#                     f"Fragments: {len(CVcontours)}")
			self.plot_frame.fig.tight_layout()
			self.plot_frame.canvas.draw()
			# waiting to see changes
			time.sleep(0.01)
			# flush the changes to the screen
			self.plot_frame.canvas.flush_events()

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
				QMessageBox.about(self, f"Error value '{value}'", f"Value must be a number from {borders[0]} to {borders[1]}")
				return None
		except Exception:
			QMessageBox.about(self, f"Error value '{input_value.text()}'", f"Value must be a number from {borders[0]} to {borders[1]}")
		return None

	def save_contour(self):
		"""
		Converting numpy arrays of contours to a mat file
		"""
		start = int(self.check_input(self.in_start_frame))
		end = int(self.check_input(self.in_end_frame))
		step = int(self.check_input(self.in_frame_stepsize))

		methodic = int(self.box_method.currentText())
		max_fragmentation = int(self.check_input(self.in_max_fragmentation, borders=[3, 80]))

		# check if value is correct
		if 0 <= start < end < self.total_frames and step > 0:
			self.status_text.setText("Saving results.... please wait")
			# prepare array of objects per frame
			matframes = np.zeros((self.total_frames, ), dtype=np.object)
			# init by void arrays
			for frame in range(self.total_frames):
				matframes[frame] = np.array([], dtype=np.int32)
			# get data per frame and fill the 'matframes'
			for index, frame in enumerate(range(start, end, step)):
				contour = self.update_draws(frame, methodic, max_fragmentation, save_result=True)
				if contour is not None:
					matframes[frame] = np.array(contour, dtype=np.int32)
				QApplication.processEvents()
				QApplication.processEvents()
				self.status_text.setText(f"Processed {index / len(range(start, end, step)) * 100:.2f} %")
			# save data into mat format
			filepath = os.path.dirname(self.filepath)
			filename = os.path.basename(self.filepath)[:-4]

			newpath = f"{filepath}/{filename}_{self.box_variable.currentText()}.mat"
			sio.savemat(newpath, {'frames': matframes})
			# you are beautiful :3
			self.status_text.setText(f"Successfully saved into {newpath}")

	def on_loop_draw(self):
		"""
		Automatic drawing data in loop by user panel settings
		"""
		start = int(self.check_input(self.in_start_frame))
		end = int(self.check_input(self.in_end_frame))
		step = int(self.check_input(self.in_frame_stepsize))

		methodic = int(self.box_method.currentText())
		max_fragmentation = self.check_input(self.in_max_fragmentation, borders=[3, 80])

		if any(d is None for d in [start, end, step, max_fragmentation]):
			return

		if 0 <= start < end < self.total_frames and step > 0:
			self.flag_loop_draw_stop = False
			self.btn_loop_draw_stop.setEnabled(True)

			for frame in range(start, end, step):
				if self.flag_loop_draw_stop:
					break
				self.in_start_frame.setText(str(frame))
				self.current_frame = frame
				self.update_draws(frame, methodic, int(max_fragmentation))
		else:
			QMessageBox.about(self, "Error", "Check the START, END and STEPS values!")

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

		methodic = int(self.box_method.currentText())
		max_fragmentation = int(self.check_input(self.in_max_fragmentation, borders=[3, 80]))

		if self.current_frame < 0:
			self.current_frame = 0
		if self.current_frame >= self.total_frames:
			self.current_frame = self.total_frames - 1

		self.in_start_frame.setText(str(self.current_frame))
		self.update_draws(self.current_frame, methodic, max_fragmentation)

	def create_main_frame(self):
		# create the main plot
		self.main_frame = QWidget()
		self.plot_frame = PlotWindow(self)
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
		self.box_method.currentTextChanged.connect(self.method_onchange)  # changed!

		self.label_method = QLabel("Method")
		self.label_method.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
		btn_panel_grid.addWidget(self.label_method, 5, 1, 1, 1)

		# POLYGON FILTER
		self.btn_filter_data = QPushButton("Apply polygon")
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
		vbox.addLayout(btn_panel_grid)

		self.main_frame.setLayout(vbox)
		self.setCentralWidget(self.main_frame)

	def create_status_bar(self):
		self.status_text = QLabel("Waiting a file...")
		self.statusBar().addWidget(self.status_text, stretch=1)


def main():
	app = QApplication(sys.argv)
	form = Application()
	form.resize(800, 350)

	form.show()
	app.exec_()

if __name__ == "__main__":
	main()
