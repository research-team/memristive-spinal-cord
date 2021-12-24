#include <iostream>
#include "setup.h"
#include "self_test.h"
#include "square_pulses.h"
#include "pattern.h"
#include "printing_functions.h"

using namespace std;

int CHANNELS_SIGNALS_SETUP(string selected_channel = "1")
{
    //fill with text
    string channels_message[10] = {};

    channels_message[0] = "ANKLE EXTENSOR (G) >>    2000 Hz, L 20 mA ";
    channels_message[1] = "ANKLE FLEXOR (G) >>    2000 Hz, L 20 mA";
    channels_message[2] = "HIP EXTENSOR (G) >>    2000 Hz, L 20 mA";
    channels_message[3] = "HIP FLEXOR (G) >>    2000 Hz, L 20 mA";
    channels_message[4] = "ANKLE EXTENSOR (G) >>    2000 Hz, L 20 mA";
    channels_message[5] = "MONOPHASIC >>    250 us, 40 Hz, L 20 mA";
    channels_message[6] = "BIPHASIC >>    250 us, 40 Hz, L 20 mA";
    channels_message[7] = "HIP FLEXOR (G) >>    2000 Hz, L 20 mA";
    channels_message[8] = "MONOPHASIC >>    250 us, 40 Hz, L 150 mA";
    channels_message[9] = "BIPHASIC >>   250 us, 40 Hz, L 150 mA";

    string F1_F4_message = "  F1 SAVE & BACK  F2 SELECT  F3 SAVE & STIM";

    //to switch between screens
    if (selected_channel == "1")
    {
        SQUARE_PULSES(selected_channel);
    }
    if (selected_channel == "2")
    {
        PATTERN(selected_channel);
    }
    if (selected_channel == "F1")
    {
        SETUP(selected_channel);
    }
    if (selected_channel == "F2")
    {
        SELF_TEST(selected_channel);
    }

    //main lines
    cout << header("SETUP") << endl;
    cout << "  CHANNELS SIGNALS SETUP " << endl;
    cout << "  Press the encoder or F2 for detaled channel setup" << endl;
    cout << " " << endl;
    cout << "  LEFT LEG " << endl;

    int first_channel = 0;
    int last_channel = 4;
    int first_blok_lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);
    cout << " " << endl;

    cout << "  RIGHT LEG " << endl;
    first_channel = 4;
    last_channel = 8;
    int second_blok_lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);
    cout << " " << endl;

    cout << "  SCS " << endl;
    first_channel = 8;
    last_channel = 10;
    int third_blok_lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);

    // 23 - max hight, 4 - header + ending line, 8 - other printed lines
    int lines_without_text = 23 - 4 - first_blok_lines_with_text - second_blok_lines_with_text - third_blok_lines_with_text - 8;
    cout << " " << endl;
    
    cout << F1_F4_message << endl;

    std::cin.ignore();

    return 0;
}
