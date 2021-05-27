import cv2

class VideoAnalyzer:
	def __init__(self, video_path):
		self.capture = cv2.VideoCapture(video_path)
		self.fps = self.capture.get(cv2.CAP_PROP_FPS)
		self.window_name = 'Video analyzer'
		self.frame_in_milliseconds = int(1 / self.fps * 1000)
		self.video_length = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
		self.start, self.end = 0, self.video_length
		self.frame_index = self.start

	def start_on_change(self, trackbar_frame):
		self.plot(trackbar_frame)
		self.start = trackbar_frame

	def end_on_change(self, trackbar_frame):
		self.plot(trackbar_frame)
		self.end = trackbar_frame

	def plot(self, index=None):
		if index:
			self.frame_index = index
		self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
		_, img = self.capture.read()
		frame_time = self.capture.get(cv2.CAP_PROP_POS_MSEC)
		cv2.putText(img=img,
		            text=f'Frame: {self.frame_index}; Time: {frame_time / 1000:.3f}s; FPS: {self.fps}',
		            org=(50, 50),
		            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
		            color=(0, 255, 255), thickness=2)
		cv2.imshow(self.window_name, img)
		self.frame_index += 1
		return cv2.waitKey(self.frame_in_milliseconds)

	def run(self):
		cv2.namedWindow(self.window_name)
		cv2.createTrackbar('start', self.window_name, 0, self.video_length, self.start_on_change)
		cv2.createTrackbar('end', self.window_name, self.video_length, self.video_length, self.end_on_change)

		self.start_on_change(0)

		self.start = cv2.getTrackbarPos('start', self.window_name)
		self.end = cv2.getTrackbarPos('end', self.window_name)

		self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.start)
		self.frame_index = self.start

		while True:
			if self.frame_index >= self.end:
				self.frame_index = self.start
			key = self.plot()
			if key == ord('p'):
				cv2.waitKey(-1)  # wait until any key is pressed
			if key == ord('s'):
				self.frame_index = self.start
				self.plot()
				cv2.waitKey(-1)  # wait until any key is pressed

video = '/home/alex/Videos/IMG_0128.MOV'
v = VideoAnalyzer(video)
v.run()
