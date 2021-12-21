#include <iostream>
#include "electrodes_cont.h"
#include "all_sensors.h"
#include "main_scr.h"
#include "printing_functions.h"

using namespace std;

int SELF_TEST(string selected_channel = "1")
{
    //fill with text
    string channels_message[10] = {};

    channels_message[0] = "WITHOUT SENSORS";
    channels_message[1] = "ALL SENSORS";
    channels_message[2] = "FLEXIMETERS ONLY";
    channels_message[3] = "PRESSURE ONLY";

    string F1_F4_message = "F1  BACK";

    //to switch between screens
    if (selected_channel == "1")
    {
        ELECTRODES_CONTACT(selected_channel);
    }
    if (selected_channel == "2")
    {
        ALL_SENSORS(selected_channel);
    }
    if (selected_channel == "3")
    {
        ALL_SENSORS(selected_channel);
    }
    if (selected_channel == "4")
    {
        ALL_SENSORS(selected_channel);
    }
    if (selected_channel == "F1")
    {
        MAIN(selected_channel, false);
    }

    //main lines
    cout << header("SELF TEST") << endl;
    cout << "  SENSORS CONFIGURATION:" << "\n" << endl;

    int first_channel = 0;
    int last_channel = 10;
    int lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);

    // 26 - max hight, 4 - header + ending line, 2 - other
    int lines_without_text = 26 - 4 - lines_with_text - 2;
    cout << empty_line(lines_without_text) << endl;

    cout << F1_F4_message << endl;

    std::cin.ignore();

    return 0;
}
