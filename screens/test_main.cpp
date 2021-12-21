#include <iostream>
#include "printing_functions.h"
#include "active_test.h"
#include "all_sensors.h"
#include "current_limits.h"
#include "electrodes_cont.h"
#include "main_scr.h"
#include "mixed.h"
#include "passive_test.h"
#include "pattern.h"
#include "self_test.h"
#include "setup.h"
#include "shape_of_signal.h"
#include "square_pulses.h"
#include "stimulation.h"

using namespace std;

int main()
{
	STIMULATION("1");
	ACTIVE_TEST("1", false);
	ACTIVE_TEST("1", true);
	PASSIVE_TEST("1", false);
	PASSIVE_TEST("1", true);
	ALL_SENSORS("1");
	CURRENT_LIMITS("1");
	ELECTRODES_CONTACT("1");
	MAIN("1", false);
	MAIN("1", true);
	MIXED("1");
	PATTERN("1");
	SELF_TEST("1");
	SETUP("1");
	SHAPE_OF_SIGNAL("1");
	SQUARE_PULSES("1");
	STIMULATION("1");

    return 0;
}
