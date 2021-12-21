#include <iostream>
#include "electrodes_cont.h"
#include "all_sensors.h"
#include "printing_functions.h"

using namespace std;

int PASSIVE_TEST(string selected_channel = "1", bool error = false)
{
    //fill with text

    string F1_F4_message = "F1  BACK    F2 NEXT";

        //to switch between screens
    if (selected_channel == "F1")
    {
        ALL_SENSORS(selected_channel);
    }
    if (selected_channel == "F2")
    {
        ELECTRODES_CONTACT(selected_channel);
    }

    //main lines
    cout << header("PASSIVE") << endl;
    cout << "  Please bend and straighten the ankle, knee and hip\n  to the maximum possible angles 5 times in series" << "\n" << endl;
    cout << "\n" << endl;

    cout << "        step 1 step 2 step 3 step 4 step 5" << "\n" << endl;
    cout << "  ankle ██████ ▒▒▒▒▒▒ ▒▒▒▒▒▒ ▒▒▒▒▒▒ ▒▒▒▒▒▒" << "\n" << endl;
    cout << "  knee  ██████ ▒▒▒▒▒▒ ▒▒▒▒▒▒ ▒▒▒▒▒▒ ▒▒▒▒▒▒" << "\n" << endl;
    cout << "  hip   ██████ ▒▒▒▒▒▒ ▒▒▒▒▒▒ ▒▒▒▒▒▒ ▒▒▒▒▒▒" << "\n" << endl;
    cout << "\n  After finishing, press F2" << "\n" << endl;

    if (error == true)
    {
        cout << error_line( "Hip sensors (1, 2)  malfunction", "fatal") << endl;
        // 26 - max hight, 4 - header + ending line, 10 - other, 4 - error
        int lines_without_text = 26 - 4 - 16 - 4;
        cout << empty_line(lines_without_text) << endl;
    }
    else
    {
        // 26 - max hight, 4 - header + ending line, 10 - other
        int lines_without_text = 26 - 4 - 16;
        cout << empty_line(lines_without_text) << endl;
    }

    cout << F1_F4_message << endl;

    std::cin.ignore();

    return 0;
}
