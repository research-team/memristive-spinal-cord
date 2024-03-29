#include <iostream>
#include "square_pulses.h"
#include "pattern.h"
#include "mixed.h"
#include "setup.h"
#include "printing_functions.h"

using namespace std;

int CHANNELS_SIGNALS_SETUP(string selected_channel = "1")
{
    //fill with text
    string channels_message[10] = {};

    channels_message[0] = "SQUARE PULSES (ON ALL CHANNELS)";
    channels_message[1] = "PATTERN (ON ALL CHANNELS)";
    channels_message[2] = "MIXED (SQUARE PULSES OR PATTERN ON EACH CHANNEL)";

    string F1_F4_message = "  F1  BACK";

    //to switch between screens
    if (selected_channel == "1")
    {
        SQUARE_PULSES(selected_channel);
    }
    if (selected_channel == "2")
    {
        PATTERN(selected_channel);
    }
    if (selected_channel == "3")
    {
        MIXED(selected_channel);
    }
    if (selected_channel == "F1")
    {
        SETUP(selected_channel);
    }

    //main lines
    cout << header("SETUP") << endl;
    cout << "  SHAPE OF SIGNAL " << "\n" << endl;

    int first_channel = 0;
    int last_channel = 10;
    int lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);

    // 26 - max hight, 4 - header + ending line
    int lines_without_text = 26 - 4 - lines_with_text-2;
    cout << empty_line(lines_without_text) << endl;
    cout << F1_F4_message << endl;

    std::cin.ignore();

    return 0;
}
