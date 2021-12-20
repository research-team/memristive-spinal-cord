#include <iostream>
#include "shape_of_signal.h"
#include "self_test.h"
#include "printing_functions.h"

using namespace std;

int PATTERN(string selected_channel = "1")
{
    //fill with text
    string channels_message[8] = {};

    channels_message[0] = "PACK DURATION    150 ms";

    string F1_F4_message = "F1  SAVE & BACK    F2  SAVE & STIM";

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
    cout << "  PATTERN \n" << endl;

    int first_channel = 0;
    int last_channel = 8;
    int lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);

    // 26 - max hight, 4 - header + ending line, 2 - other printed lines
    int lines_without_text = 26 - 4 - lines_with_text - 2;
    cout << empty_line(lines_without_text) << endl;

    cout << F1_F4_message << endl;

    std::cin.ignore();

    return 0;
}
