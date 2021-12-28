#include <iostream>
//#include "shape_of_signal.h"
#include "channels_signals_setup.h"
#include "self_test.h"
#include "printing_functions.h"

using namespace std;

int SQUARE_PULSES(string selected_channel = "1")
{
    //fill with text
    string channels_message[10] = {};

    channels_message[0] = "LIMIT OF CURRENT FOR CHANNEL    20 mA";
    channels_message[1] = "PHASE    MONOPHASIC";
    channels_message[2] = "PULSE WIDTH    250 us";
    channels_message[3] = "FREQUENCY    40 Hz";
    channels_message[4] = "BURST DURATION    10 ms";
    channels_message[5] = "INTERBURST INTERVAL    100ms";
    channels_message[6] = "INTERNAL MODULATION FREQUENCY    5000 Hz";

    string F1_F4_message = "  F1 SAVE & BACK  F2 SAVE & STIM";

    //to switch between screens
    if (selected_channel == "F1")
    {
        CHANNELS_SIGNALS_SETUP(selected_channel);
    }
    if (selected_channel == "F2")
    {
        SELF_TEST(selected_channel);
    }

    //main lines
    cout << header("SETUP") << endl;
    cout << "  SQUARE PULSES \n" << endl;

    int first_channel = 0;
    int last_channel = 10;
    int lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);

    // 23 - max hight, 4 - header + ending line, 2 - other printed lines
    int lines_without_text = 23 - 4 - lines_with_text - 2;
    cout << empty_line(lines_without_text) << endl;

    cout << F1_F4_message << endl;

    std::cin.ignore();

    return 0;
}
