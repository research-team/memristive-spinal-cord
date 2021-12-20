#include <iostream>
#include "setup.h"
#include "self_test.h"
#include "printing_functions.h"

using namespace std;

int MAIN(string selected_channel = "1", bool error = false)
{
    //fill with text
    string channels_message[8] = {};

    channels_message[0] = "PATIENT ID: ----";

    string F1_F4_message = "F1  CURSOR LEFT    F2  CURSOR RIGHT    F3  SETUP    F4  TEST & STIM";

    //to switch between screens
    if (selected_channel == "F3")
    {
        SETUP(selected_channel);
    }
    if (selected_channel == "F4")
    {
        SELF_TEST(selected_channel);
    }

    //main lines
    cout << header("MAIN") << endl;
    cout << "  Please select patient in the web application or" << endl;
    cout << "  Enter user ID:" << "\n" << endl;
    cout << "  - Use the encoder to select a digit" << endl;
    cout << "  - Use F1 and F2 to change the position of the cursor" << "\n" << endl;

    int first_channel = 0;
    int last_channel = 2;
    int lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);
    cout << "  DATE: YYYY-MM-DD" << "\n" << endl;
    cout << "  PATIENT NAME: ----" << "\n" << endl;
    cout << "  BIRTH DATE: YYYY-MM-DD" << "\n" << endl;

    if (error == true)
    {
        cout << error_line("Select patient in the web application or enter patient ID") << endl;
        // 26 - max hight, 4 - header + ending line, 14 - other, 4 - for error
        int lines_without_text = 26 - 4 - 14 - 4;
        cout << empty_line(lines_without_text) << endl;
    }
    else
    {
        // 26 - max hight, 4 - header + ending line, 14 - other
        int lines_without_text = 26 - 4 - 14;
        cout << empty_line(lines_without_text) << endl;
    }

    cout << "  F1  CURSOR LEFT    F2  CURSOR RIGHT    F3  SETUP    F4  TEST & STIM" << endl;

    return 0;
}
