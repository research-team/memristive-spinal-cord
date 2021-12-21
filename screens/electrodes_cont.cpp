#include <iostream>
#include "main_scr.h"
#include "stimulation.h"
#include "printing_functions.h"

using namespace std;

int ELECTRODES_CONTACT(string selected_channel = "1")
{
    //fill with text
    string channels_message[10] = {};

    channels_message[0] = "connected";
    channels_message[1] = "connected";
    channels_message[2] = "disconnected";
    channels_message[3] = "connected";
    channels_message[4] = "connected";
    channels_message[5] = "connected";
    channels_message[6] = "disconnected";
    channels_message[7] = "connected";
    channels_message[8] = "connected";
    channels_message[9] = "disconnected";

    string F1_F4_message = "F1  BACK TO MAIN    F2 START    F3 REPEAT    F4 STIMULATION";

    //to switch between screens
    if (selected_channel == "F1")
    {
        MAIN(selected_channel, false);
    }
    /*if (selected_channel == "F2")
    {
        START TESTING ???;
    }
    if (selected_channel == "F3")
    {
        REPEAT TESTING ???;
    }*/
    if (selected_channel == "F3")
    {
        STIMULATION(selected_channel);
    }

    //main lines
    cout << header("ELECTRODES' CONTACT TEST") << endl;
    cout << "  Please connect used channels and press F2" << "\n" << endl;

    int first_channel = 0;
    int last_channel = 10;

    int lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);

    cout << "\n  If used channels are disconnected, check the connection\n  and press F3 to re-check" << "\n" << endl;

    // 26 - max hight, 4 - header + ending line, 4 - other
    int lines_without_text = 26 - 4 - lines_with_text - 6;
    cout << empty_line(lines_without_text) << endl;

    cout << F1_F4_message << endl;

    std::cin.ignore();

    return 0;
}

