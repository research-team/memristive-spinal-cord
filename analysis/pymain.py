from pynput import keyboard
from threading import Thread

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

from pynput.keyboard import Key, Listener

def on_press(key):
	pass

def on_release(key):
	if key == Key.esc:
		# Stop listener
		return False
	menu(str(key).replace("'", ''))


def menu(selected_channel='1'):
	channels_message = {
		'1': "MAXIMUM CURRENT    20 mA",
		'2': "STIMULATION TIME    1 min",
		'3': "ACCELERATION TIME    5 sec",
		'4': "SHAPE OF SIGNAL",
		'5': " ",
		'6': " ",
		'7': " ",
		'8': " ",
	}

	print(header("SETUP"))

	for channel_num, channel_text in channels_message.items():
		if channel_text != " ":
			if selected_channel == channel_num:
				color = "white"
			else:
				color = "original"
			print(text_line(number=channel_num, message=channel_text, color=color), end="\r")

	print(zero_line(12))
	print(header("  F1 SAVE & BACK    F2 RESET"))


if __name__ == "__main__":
	main_thread = Thread(target=menu)
	main_thread.start()

	# Collect events until released
	with Listener(on_press=on_press,on_release=on_release) as listener:
		listener.join()



