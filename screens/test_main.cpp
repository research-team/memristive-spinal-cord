#include <iostream>
#include "printing_functions.h"
#include "active_test.h"
#include "all_sensors.h"
//#include "current_limits.h"
#include "electrodes_cont.h"
#include "main_scr.h"
#include "channels_signals_setup.h"
#include "passive_test.h"
#include "pattern.h"
#include "self_test.h"
#include "setup.h"
//#include "shape_of_signal.h"
#include "square_pulses.h"
#include "stimulation.h"
#include "end_of_stim.h"

using namespace std;

int main()
{		//SELF_TEST("1");
	MAIN("1", true);
	SETUP("1");
	CHANNELS_SIGNALS_SETUP("1");
	CHANNELS_SIGNALS_SETUP("2");
	SELF_TEST("1");
	ALL_SENSORS("1");
	ACTIVE_TEST("1", true);
	PASSIVE_TEST("1", true);
	ELECTRODES_CONTACT("1");
	STIMULATION("1");
	END_OF_STIM("1");
	
//	SHAPE_OF_SIGNAL("1");

    return 0;
}
