#include <iostream>
#include "main_scr.h"
//#include "shape_of_signal.h"
#include "channels_signals_setup.h"
//#include "current_limits.h"
#include "printing_functions.h"

using namespace std;

int SETUP(string selected_channel = "1")
{
    //fill with text
    string channels_message[10] = {};

    channels_message[0] = "MAXIMUM OF CURRENT (FOR ALL CHANNELS)    20 mA";
    channels_message[1] = "STIMULATION TIME    1 min";
    channels_message[2] = "ACCELERATION TIME    5 sec";
    channels_message[3] = "CHANNELS SIGNALS SETUP";

    string F1_F4_message = "  F1 SAVE & BACK  F2 RESET";

    //to switch between screens
    /*if (selected_channel == "4")
    {
        SHAPE_OF_SIGNAL(selected_channel);
    }*/
    if (selected_channel == "5")
    {
        CHANNELS_SIGNALS_SETUP(selected_channel);
    }

    if (selected_channel == "F1")
    {
        MAIN(selected_channel, false);
    }
    /*if (selected_channel == "F2")
    {
        RESET??
    }*/

    //main lines
    cout << header("SETUP") << endl;

    int first_channel = 0;
    int last_channel = 10;
    int lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);

    // 23 - max hight, 4 - header + ending line
    int lines_without_text = 23 - 4 - lines_with_text;
    cout << empty_line(lines_without_text) << endl;

    cout << F1_F4_message << endl;

    std::cin.ignore();

    return 0;
}
