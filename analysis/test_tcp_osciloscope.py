import pyvisa
import numpy as np
from matplotlib import pyplot as plt
from time import sleep


class Oscilloscope:
	def __init__(self, address):
		self.manager = pyvisa.ResourceManager()
		self.inst = self.manager.open_resource(address)

	@staticmethod
	def printt(text, value):
		if isinstance(value, float):
			print(f"{text:<30} {value}")
		else:
			print(f"{text:<30} {value.strip()}")

	def write(self, command):
		print(f"{command:<30}", end=' ')
		self.inst.write(command)
		print(self.inst.last_status.name)

	def run(self):
		self.inst.timeout = 1000
		self.inst.InputBufferSize = 2048

		read, query, write, printt = self.inst.read, self.inst.query, self.write, self.printt

		printt("Oscilloscope", query('*IDN?'))
		# write('*RST')
		# sleep(10)
		write(':ACQUIRE:MDEPTH 1200000')

		write(':WAVEFORM:SOURCE CHAN1')
		write(':WAVEFORM:MODE RAW')
		write(':WAVEFORM:FORMAT BYTE')

		memorydepth = int(query(':ACQUIRE:MDEPTH?'))  # 300kpts
		step = 250000 #int(query(':WAVeform:POINTS?'))
		print(step, memorydepth)
		xinc = float(query(':WAVeform:XINCrement?'))

		plt.ion()
		for begin in range(1, memorydepth + 1, step):
			end = begin + step - 1
			if end > memorydepth:
				end = memorydepth
			self.inst.write(f':WAV:START {begin}')
			self.inst.write(f':WAV:STOP {end}')
			
			data = np.fromiter(query(':WAV:DATA?')[11:].split(","), dtype=np.float)
			new_len = data.size
			plt.plot(np.arange(begin, begin + new_len) * xinc, data)
			plt.draw()
			plt.pause(0.001)
			# plt.clf()

		plt.show(block=True)


		exit()
		#
		# write(":TIM:MAIN:SCALE 10")  # in seconds, 1E-3 is 1 ms
		# write(':ACQUIRE:MDEPTH 12000000')
		# write(':CHAN1:SCALE 1')
		# write(':CHANnel1:BWLimit 20M')
		#
		# # write(':MEASURE:COUNTER:SOURCE CHAN1')
		# # printt("Hz", query(':MEASure:COUNter:VALue?'))
		#
		# printt("Scale", query(":TIMebase:MAIN:SCALE?"))
		# printt("Memory depth", query(':ACQUIRE:MDEPTH?'))
		# printt("WAVE:MODE", query(':WAVEFORM:MODE?'))
		# printt("Rate", query(':ACQUIRE:SRATE?'))
		# printt("Reading channel", query(':WAVEFORM:SOURCE?'))
		# printt("Points", query(':WAVeform:POINTS?'))
		#
		# xinc = float(query(':WAVeform:XINCrement?'))
		#
		#
		# preamble_head = ["format", "type", "points", "count", "xincrement", "xorigin", "xreference", "yincrement",
		#				  "yorigin", "yreference"]
		# preamble_data = query(':WAVeform:PRE?').split(",")
		# for (k, t) in zip(preamble_head, preamble_data):
		#	 printt(k, t)
		#
		# # values = inst.query_ascii_values('CURV?', container=np.array)

		print("= " * 10)
		print("PAUSE 2 seconds")
		sleep(2)
		plt.ion()
		memorydepth = int(query(':ACQUIRE:MDEPTH?'))  # 300kpts
		step = 125000

		for begin in range(1, memorydepth + 1, step):
			end = begin + step - 1
			if end > memorydepth:
				end = memorydepth
			write(f':WAV:STAR {begin}')
			write(f':WAV:STOP {end}')
			d = query(':WAV:DATA?')


		start = 0
		for i in range(5):
			d = query(':WAVEFORM:DATA?')
			data = np.fromiter(d[11:].split(","), dtype=np.float)
			new_len = data.size
			plt.plot(np.arange(start, start + new_len) * xinc, data)
			plt.draw()
			plt.pause(0.001)
			# plt.clf()
			start += new_len

			# (0x8E - YORigin - YREFerence) × YINCrement.

		plt.show(block=True)

		# write(":OUTPut 0")

	def __del__(self):
		self.manager.close()

if __name__ == '__main__':
	osc = Oscilloscope("TCPIP0::192.168.1.33::INSTR")
	# osc = Oscilloscope("USB0::0x1ab1::0x04ce::DS1ZB205000622")
	# :RIGOL TECHNOLOGIES,DS1074Z,DS1ZB205000622,00.04.04.SP3
	osc.run()
