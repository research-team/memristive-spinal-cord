#include <iostream>   // output/input streams definitions
#include <unistd.h>   // UNIX standard definitions
#include <fcntl.h>    // file control definitions
#include <termios.h>  // POSIX terminal control definitions
#include <cstring>    // string function definitions

#define COLOR_RESET "\x1b[0m"
#define COLOR_RED "\x1b[1;31m"
#define COLOR_GREEN "\x1b[1;32m"

/**
for more information see here: https://www.cmrr.umn.edu/~strupp/serial.html
run the program with sudo or execute the command: sudo adduser $USER dialout
*/

using namespace std;

int serial_port;
constexpr const char* const SERIAL_PORT_USB = "/dev/ttyUSB0";

void errorchk(bool condition, string text) {
	cout << text << " ... ";
	if (condition) {
		cout << COLOR_RED "ERROR" COLOR_RESET << endl;
		cout << COLOR_RED "FAILED!" COLOR_RESET << endl;
		close(serial_port);
		exit(0);
	} else {
		cout << COLOR_GREEN "OK" COLOR_RESET << endl;
	}
}

int main(int argc, char **argv) {
	// structure to store the port settings in
	struct termios port_settings;
	// struct for terminal app, since terminal also connects through a virtual system serial port
	// "/dev/ttyUSB0" so it is necessary to redirect input and output, however this wouldn't be necessary
	// if we want to send data streams from the program directly and also if we don't need to show the
	// raw output to the user.
	struct termios stdio;
	// after our work is over we can reset the terminal input and output to the os instead of to and from the serial port
	struct termios old_port_settings;

	unsigned char data = 'D';

	// get the current options of the STDOUT_FILENO
	tcgetattr(STDOUT_FILENO, &old_port_settings);

	printf("Start with %s \n", SERIAL_PORT_USB);

	// O_RDWR - read/write access to serial port
	// O_NONBLOCK - when possible, the file is opened in nonblocking mode
	serial_port = open(SERIAL_PORT_USB, O_RDWR | O_NONBLOCK);
	errorchk(serial_port < 0, "open serial port");

	// populate the structures with the memory size of the structure by zeroes
	memset(&stdio, 0, sizeof(stdio));
	memset(&port_settings, 0, sizeof(port_settings));

	// output parameters
	stdio.c_iflag = 0;
	stdio.c_oflag = 0;
	stdio.c_cflag = 0;
	stdio.c_lflag = 0;
	stdio.c_cc[VMIN] = 1;
	stdio.c_cc[VTIME] = 0;

	// set the new options for the port
	// TCSANOW - make changes now without waiting for data to complete
	// TCSAFLUSH - flush input and output buffers and make the change
	tcsetattr(STDOUT_FILENO, TCSANOW, &stdio);
	tcsetattr(STDOUT_FILENO, TCSAFLUSH, &stdio);

	// make the reads non-blocking
	fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);

	// input mode flags:
	port_settings.c_iflag = 0;
	// output mode flags: 0 - raw output (no output processing)
	port_settings.c_oflag = 0;
	// local mode flags:
	port_settings.c_lflag = 0;
	// control mode flags: 8 bits, no parity, 1 stop bit | enable receiver | ignore modem control lines
	port_settings.c_cflag |= (CS8 | CREAD | CLOCAL);
	// minimum number of characters to read => 0 - read doesn't block
	port_settings.c_cc[VMIN] = 0;
	// time to wait for data (tenths of seconds) => 0 seconds read timeout
	port_settings.c_cc[VTIME] = 0;

	cfsetispeed(&port_settings, B115200);  // set read speed as 115 200 (bps)
	cfsetospeed(&port_settings, B115200);  // set write speed as 115 200 (bps)

	// set the new options for the serial port
	tcsetattr(serial_port, TCSANOW, &port_settings);

	// main body of sending data
	for (int i = 0; i < 10; i++) {
		// send data to the port
		write(serial_port, &data, sizeof(data));
		// wait
		usleep(250000);
		// read data from the port
		write(STDOUT_FILENO, &data, sizeof(data));
	}

	// close the serial port
	close(serial_port);

	// restore old port settings
	tcsetattr(STDOUT_FILENO, TCSANOW, &old_port_settings);

	return EXIT_SUCCESS;
}