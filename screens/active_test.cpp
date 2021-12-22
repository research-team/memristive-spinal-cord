#include <iostream>
#include "electrodes_cont.h"
#include "all_sensors.h"
#include "printing_functions.h"
using namespace std;

int ACTIVE_TEST(string selected_channel = "1", bool error = false)
{
    //fill with text

    string F1_F4_message = "  F1  BACK    F2 NEXT";

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
    cout << header("ACTIVE") << endl;
    cout << "  Please take 5 steps on the treadmill" << "\n"<< endl;
    cout << " " << endl;

    cout << "  step 1 step 2 step 3 step 4 step 5"  << "\n"<< endl;
    cout << "  ██████ ▒▒▒▒▒▒ ▒▒▒▒▒▒ ▒▒▒▒▒▒ ▒▒▒▒▒▒"  << "\n"<< endl;
    cout << "  After finishing, press F2"  << endl;

    if (error == true)
    {
        cout << error_line("Hip sensor (1)  malfunction", "normal") << endl;
        // 26 - max hight, 4 - header + ending line, 10 - other, 4 - error
        int lines_without_text = 26 - 4 - 8 - 4;
        cout << empty_line(lines_without_text) << endl;
    }
    else
    {
        // 26 - max hight, 4 - header + ending line, 10 - other
        int lines_without_text = 26 - 4 - 8;
        cout << empty_line(lines_without_text) << endl;
    }

    cout << F1_F4_message << endl;

    std::cin.ignore();

    return 0;
}
