#include <iostream>
#include "main_scr.h"
#include "printing_functions.h"

using namespace std;

int END_OF_STIM(string selected_channel = "1")
{
    //fill with text
    string F1_F4_message = "  F1 CONTINUE  F2 NEW";

    //to switch between screens
    if (selected_channel == "F1")
    {
        MAIN(selected_channel);
    }
    if (selected_channel == "F2")
    {
        MAIN(selected_channel);
    }

    //main lines
    cout << header(" ") << endl;
    cout << "  The stimulation time is over." << "\n" << endl;
    cout << "\n" << endl;

    cout << "  Would you like to continue with this patient or choose\n  a new one?" << "\n" << endl;
    cout << "\n" << endl;

    cout << "\033[30;47m  Please use the web application. \033[0m" << "\n" << endl;

    int lines_without_text = 23 - 4 - 11;
    cout << empty_line(lines_without_text) << endl;

    cout << F1_F4_message << endl;

    std::cin.ignore();

    return 0;
}
