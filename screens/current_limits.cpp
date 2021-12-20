#include <iostream>
#include "shape_of_signal.h"
#include "self_test.h"
#include "printing_functions.h"

using namespace std;

int CURRENT_LIMITS(string selected_channel = "1")
{
    //fill with text
    string channels_message[8] = {};
    channels_message[0] = "xx mA";
    channels_message[1] = "xx mA";
    channels_message[2] = "xx mA";
    channels_message[3] = "xx mA";
    channels_message[4] = "xx mA";
    channels_message[5] = "xx mA";
    channels_message[6] = "xx mA";
    channels_message[7] = "xx mA";

    string F1_F4_message = "F1  BACK";

    //to switch between screens
    if (selected_channel == "F1")
    {
        SHAPE_OF_SIGNAL(selected_channel);
    }
    if (selected_channel == "F2")
    {
        SELF_TEST(selected_channel);
    }

    //main lines
    cout << header("SETUP") << endl;
    cout << "  CURRENT LIMITS " << "\n" << endl;
    cout << "  LEFT LEG " << endl;

    int first_channel = 0;
    int last_channel = 4;
    int first_blok_lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);

    cout << "  RIGHT LEG " << endl;
    first_channel = 4;
    last_channel = 8;
    int second_blok_lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);

    // 26 - max hight, 4 - header + ending line, 4 - other printed lines
    int lines_without_text = 26 - 4 - first_blok_lines_with_text - second_blok_lines_with_text - 4;
    cout << empty_line(lines_without_text) << endl;

    cout << F1_F4_message << endl;

    std::cin.ignore();

    return 0;
}
