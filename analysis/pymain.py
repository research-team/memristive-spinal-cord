import time

from pynput import keyboard
from threading import Thread

channels_message = {
	'1': " ",
	'2': " ",
	'3': " ",
	'4': " ",
	'5': " ",
	'6': " ",
	'7': " ",
	'8': " ",
}

F1_F4_message = {"f1": " ", "f2": " ", "f3": " ", "f4": " "}


def header(message):
	return f"\n\n  {message}\n\n"


def zero_line(count):
	return "\n" * (count - 1)


def text_line(number, message, color):
	str_num = str(number)
	if color == "white":
		color = "[30;47m"
	if color == "original":
		color = "[0m"

	t_line = "\033" + color + "  " + str_num + "  " + message + " \033[0m\n"
	return t_line


# def final_line(number, message_arr, color):
#
# 	str_num = str(number).upper()
# 	if color == "white":
# 		color = "[30;47m"
# 	if color == "original":
# 		color = "[0m"
#
# 	all_line= ' '
# 	for i in range(len(message_arr)-1):
# 		t_line = "\033" + color + "  " + str_num + "  " + f1 + " \033[0m\n"
# 	return t_line


from pynput.keyboard import Key, Listener


def on_press(key):
	pass


def on_release(key):
	if key == Key.esc:
		# Stop listener
		return False
	menu(str(key).replace("'", ''))


def SHAPE_OF_SIGNAL(selected_channel):
	print(header("SETUP"))
	print("  SHAPE OF SIGNAL")
	channels_message["1"] = "SQUARE PULSES (ALL)"
	channels_message["2"] = "PATTERN (ALL)"
	channels_message["3"] = "MIXED"

	F1_F4_message["f1"] = "BACK"

	# bug - double print of 3 MIXED
	for channel_num, channel_text in channels_message.items():
		if channel_text != " ":
			print(text_line(number=channel_num, message=channel_text, color="original"), end="\r")

	if selected_channel == "f1":
		SETUP_scr(selected_channel=1)


def SETUP_scr(selected_channel):
	channels_message["1"] = "MAXIMUM CURRENT    20 mA"
	channels_message["2"] = "STIMULATION TIME    1 min"
	channels_message["3"] = "ACCELERATION TIME    5 sec"
	channels_message["4"] = "SHAPE OF SIGNAL"
	channels_message["5"] = "CURRENT LIMITS"

	F1_F4_message["f1"] = "SAVE & BACK"
	F1_F4_message["f2"] = "RESET"

	print(header("SETUP"))

	for channel_num, channel_text in channels_message.items():
		if channel_text != " ":
			if selected_channel == "4":
				for k in channels_message.keys():
					channels_message[k] = " "
				for k in F1_F4_message.keys():
					F1_F4_message[k] = " "

				SHAPE_OF_SIGNAL(selected_channel=selected_channel)
			if selected_channel == channel_num:
				color = "white"
			else:
				color = "original"
			print(text_line(number=channel_num, message=channel_text, color=color), end="\r")

	print(zero_line(12))

	all_line = ' '
	for F_num, F_message in F1_F4_message.items():
		if F_message != " ":
			all_line += F_num.upper() + ' ' + F_message + "\t"
	print(f"  {all_line}")


def menu(selected_channel='1'):
	SETUP_scr(selected_channel=selected_channel)


if __name__ == "__main__":
	main_thread = Thread(target=menu)
	main_thread.start()

	# Collect events until released
	with Listener(on_press=on_press, on_release=on_release) as listener:
		listener.join()
