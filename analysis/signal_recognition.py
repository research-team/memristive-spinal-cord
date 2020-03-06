import os
import sys
import cv2
import time
import h5py
import numpy as np
import hdf5storage
import scipy.io as sio
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

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
		new_order = tuple(map(int, self.in_data_reshape.text().split()))
		self.dat = np.transpose(self.dat, new_order)

		self.in_data_reshape.setEnabled(False)
		self.btn_reshape_data.setEnabled(False)

		new_shape = self.dat.shape
		self.total_frames = new_shape[2]

		self.in_start_frame.setValidator(QIntValidator(0, self.total_frames - 1))
		self.in_end_frame.setValidator(QIntValidator(0, self.total_frames - 1))
		self.in_end_frame.setText(str(self.total_frames - 1))

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

	def update_draws(self, frame, percentile, save_result=False):
		# get the current methodic
		methodic = int(self.box_method.currentText())
		# get data
		image = self.dat[:, :, frame, methodic]  # image = np.flip(np.rot90(dat[methodic, frame, :, :], -1), 1)[:100, :]
		# absolute minimum and maximum
		# print(np.max(self.dat[methodic, :, :, :]))
		# print(np.min(self.dat[methodic, :, :, :]))
		# normalize data from 0 to 255 with dynamic borders (min and max). It mades grayscale cmap
		image = ((image - image.min()) * (1 / (image.max() - image.min()) * 255)).astype('uint8')
		# reverse colors if epilepsy radio button checked
		if self.radio_reversed.isChecked():
			image = 255 - image
		# set the dynamic thresh value to catch the most "deviant" data -- 95%
		thresh_value = np.percentile(image.ravel(), percentile)
		# blur the image
		median_blur = cv2.medianBlur(image.copy(), 5)
		# get coordinates of points which are greater than thresh value
		y, x = np.where(median_blur >= thresh_value)
		kernel = np.ones((3, 3), np.uint8)
		# get the mask based on dynamic thresh value
		_, thresh = cv2.threshold(median_blur, thresh_value, 255, cv2.THRESH_BINARY)
		# transform morphology of the mask
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
		# get the contour of the mask
		*im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		# get an area of contour
		area = sum(cv2.contourArea(c) for c in contours)

		if save_result:
			cont = []
			for c in contours:
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
			self.ax3.clear()
			self.ax4.clear()
			# plotting original picture
			self.ax1.set_title("Original image")
			self.ax1.imshow(image, cmap='gray')
			# plotting blured by median with filtered dots
			self.ax2.set_title("Median blured + dots")
			self.ax2.imshow(median_blur, cmap='gray')
			self.ax2.plot(x, y, '.', color='pink', ms=8)
			self.ax3.set_title(f"Mask, morph open/close")
			# plot the mask of thresh
			self.ax3.imshow(thresh, cmap='gray')
			# plot the approximated poly area
			self.ax4.set_title(f"Approx Poly DP = {area:.2f}")
			self.ax4.imshow(median_blur, cmap='gray')
			# approximate each found contour
			for c in contours:
				if len(c) > 3:
					epsilon = 0.01 * cv2.arcLength(c, True)
					approx = cv2.approxPolyDP(c, epsilon, True).squeeze()
					approx = np.vstack((approx, approx[0, :]))
					self.ax4.fill(approx[:, 0], approx[:, 1], color='g', alpha=0.5)
					self.ax4.plot(approx[:, 0], approx[:, 1], color='g')
			# redraw canvas by new objects
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
		self.ax1 = self.fig.add_subplot(221)
		self.ax2 = self.fig.add_subplot(222, sharex=self.ax1, sharey=self.ax1)
		self.ax3 = self.fig.add_subplot(223, sharex=self.ax1, sharey=self.ax1)
		self.ax4 = self.fig.add_subplot(224, sharex=self.ax1, sharey=self.ax1)

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
