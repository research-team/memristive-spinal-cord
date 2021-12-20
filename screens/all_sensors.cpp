#include <iostream>
#include "active_test.h"
#include "passive_test.h"
#include "self_test.h"
#include "printing_functions.h"

using namespace std;

int ALL_SENSORS(string selected_channel = "1")
{
    //fill with text
    string channels_message[8] = {};

    channels_message[0] = "ACTIVE";
    channels_message[1] = "PASSIVE";

    string F1_F4_message = "F1  BACK";

    //to switch between screens
    if (selected_channel == "1")
    {
        ACTIVE_TEST(selected_channel, false);
    }
    if (selected_channel == "2")
    {
        PASSIVE_TEST(selected_channel, false);
    }
    if (selected_channel == "F1")
    {
        SELF_TEST(selected_channel);
    }

    //main lines
    cout << header("SELF TEST") << endl;
    cout << "  ALL SENSORS" << "\n" << endl;
    cout << "\n" << "  Checking the correct operation of the sensors." << "\n" << endl;
    cout << "  For patients who can perform voluntary movements:" << "\n" << endl;

    int first_channel = 0;
    int last_channel = 1;
    int first_blok_lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);

    cout << "  For patients who need nursing help:" << "\n" << endl;

    first_channel = 1;
    last_channel = 2;
    int second_blok_lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);

    // 26 - max hight, 4 - header + ending line, 2 - other
    int lines_without_text = 26 - 4 - first_blok_lines_with_text -second_blok_lines_with_text - 9;
    cout << empty_line(lines_without_text) << endl;

    cout << F1_F4_message << endl;

    std::cin.ignore();

    return 0;
}
