#include <iostream>
#include "electrodes_cont.h"
#include "setup.h"
#include "main_scr.h"
#include "printing_functions.h"

using namespace std;

int STIMULATION(string selected_channel = "1")
{
    //fill with text
    string channels_message[10] = {};
    channels_message[0] = "0.0 19 mA  Ankle Extensor (G)";
    channels_message[1] = "0.0 17 mA  Ankle Flexor (G)";
    channels_message[2] = "0.0 12 mA  Hip Extensor (G)";
    channels_message[3] = "0.0 20 mA  Hip Flexot (G)";
    channels_message[4] = "0.0 18 mA  Ankle Extensor (G)";
    channels_message[5] = "0.0 12 mA  Monophasic   250 us, 40 Hz";
    channels_message[6] = "0.0 20 mA  Biphasic    200 us, 40 Hz";
    channels_message[7] = "0.0 16 mA  Hip Flexor (G)";
    channels_message[8] = "0.0 12 mA  Monophasic   250 us, 40 Hz";
    channels_message[9] = "0.0 20 mA  Biphasic    200 us, 40 Hz";

    string F1_F4_message = "F1  STOP    F1  STOP & BACK    F1  STOP & SETUP    F1  STOP & MAIN";

    //to switch between screens
    /*if (selected_channel == "F1")
    {
        STOP;
    }*/
    if (selected_channel == "F2")
    {
        ELECTRODES_CONTACT(selected_channel);
    }
    if (selected_channel == "F3")
    {
        SETUP(selected_channel);
    }
    if (selected_channel == "F4")
    {
        MAIN(selected_channel, false);
    }

    //main lines
    cout << header("ACCELERATION TIME: 00:05") << endl;
    cout << "  SIMULATION TIME: 00:00" << "\n" << endl;
    cout << "  LEFT LEG " << endl;

    int first_channel = 0;
    int last_channel = 4;
    int first_blok_lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);
    cout << "  \n " << endl;


    cout << "  RIGHT LEG " << endl;
    first_channel = 4;
    last_channel = 8;
    int second_blok_lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);
    cout << "  \n " << endl;

    cout << "  SCS: " << endl;
    first_channel = 8;
    last_channel = 10;
    int third_blok_lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);


    // 26 - max hight, 4 - header + ending line, 4 - other printed lines
    int lines_without_text = 26 - 4 - first_blok_lines_with_text - second_blok_lines_with_text -third_blok_lines_with_text- 9;
    cout << empty_line(lines_without_text) << endl;

    cout << F1_F4_message << endl;

    std::cin.ignore();

    return 0;
}


