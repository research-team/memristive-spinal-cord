#include <iostream>
//#include "shape_of_signal.h"
#include "channels_signals_setup.h"
#include "self_test.h"
#include "printing_functions.h"

using namespace std;

int PATTERN(string selected_channel = "1")
{
    //fill with text
    string channels_message[10] = {};

    channels_message[0] = "LIMIT OF CURRENT ON CHANNEL    20 mA";
    channels_message[1] = "MODULATION    ON";
    channels_message[2] = "INTERNAL MODULATION FREQUENCY    5000 Hz";

    //channels_message[1] = "9 CHANNEL    250 us, 40 Hz";
    //channels_message[2] = "10 CHANNEL    250 us, 40 Hz";

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
    cout << "  PATTERN \n" << endl;

    int first_channel = 0;
    int last_channel = 3;
    int first_lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);
    cout << "\n" << endl;

    //cout << "  SCS: " << endl;
    //first_channel = 1;
    //last_channel = 3;
    //int second_lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);

    // 23 - max hight, 4 - header + ending line, 2 - other printed lines
    int lines_without_text = 23 - 4 - first_lines_with_text  - 4; //- second_lines_with_text
    cout << empty_line(lines_without_text) << endl;

    cout << F1_F4_message << endl;

    std::cin.ignore();

    return 0;
}
