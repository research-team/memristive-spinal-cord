#include <map>
#include <iostream>

using namespace std;

/*
int width = 73;
int height = 26;
*/

string error_line(string message)
{
    // 4 lines
    string err_line = "\033[37;41m\n  ERROR \n  " + message + " \033[0m" + "\n";
	return err_line;
}


string header(string message)
{
    string head_line = "";
	head_line = "\n";
	head_line += "  " + message + "\n";
	return head_line;
}


string empty_line(int num)
{
    string e_line = "";
    for (int i = 1; i < num; i++)
    {
        e_line += "\n";
    }
	return e_line;
}


string text_line( string number, string message, string color)
{
    string t_line = "";
    if (color == "white")
    {
        color = "[30;47m";
    }
	if (color == "original")
	{
        color = "[0m";
    }
	t_line = "\033" + color + "  " + number + "  " + message + " \033[0m" + "\n";
	return t_line;
}


string connect_disconnect( string number, string message)
{
    string c_d_line = "";
    string color = "";
    if (message == "disconnected")
    {
        color = "[37;41m";
    }
	if (message == "connected")
	{
        color = "[0m";
    }
	c_d_line = "\033" + color + "  " + number + "  " + message + " \033[0m" + "\n";
	return c_d_line;
}


int print_main_lines(string channels_message[8], string selected_channel, int first_channel, int last_channel, string mode = "basic")
{
    string channels_keys[8] = {"1", "2", "3", "4", "5", "6", "7", "8"};
    string color = "";
    int lines_with_text = 0;
    for (int i = first_channel; i < last_channel; i++)
    {
        if (mode == "basic")
            string ch_key = channels_keys[i];
            string ch_message = channels_message[i];

            if (selected_channel == ch_key)
            {
                color = "white";
            }
            else
            {
                color = "original";
            }

            if (ch_message != "")
            {
                cout << text_line(ch_key, ch_message, color) << endl;
                lines_with_text += 1;
            }
            else;

    }

    // * 2 becuase + \n
    return lines_with_text * 2;
}


int SQUARE_PULSES(string selected_channel = "1")
{
    //fill with text
    string channels_message[8] = {};

    channels_message[0] = "PHASE    MONOPHASIC";
    channels_message[1] = "PULSE DURATION    250 us";
    channels_message[2] = "FREQUENCY    40 Hz";
    channels_message[3] = "BURST DURATION    10 ms";
    channels_message[4] = "INTERBURST INTERVAL    100ms";
    channels_message[5] = "INTERNAL MODULATION PULSES    5000 Hz";

    string F1_F4_message = "F1  SAVE & BACK    F2  SAVE & STIM";

    //main lines
    cout << header("SETUP") << endl;
    cout << "  SQUARE PULSES \n" << endl;

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


int PATTERN(string selected_channel = "1")
{
    //fill with text
    string channels_message[8] = {};

    channels_message[0] = "PACK DURATION    150 ms";

    string F1_F4_message = "F1  SAVE & BACK    F2  SAVE & STIM";

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


int MIXED(string selected_channel = "1")
{
    //fill with text
    string channels_message[8] = {};

    channels_message[0] = "ANKLE EXTENSOR (G)    150 ms ";
    channels_message[1] = "ANKLE FLEXOR (G)    150 ms";
    channels_message[2] = "HIP EXTENSOR (G)    150 ms";
    channels_message[3] = "HIP FLEXOR (G)    150 ms";
    channels_message[4] = "ANKLE EXTENSOR (G)    150 ms";
    channels_message[5] = "MONOPHASIC    250 us, 40 Hz";
    channels_message[6] = "BIPHASIC    250 us, 40 Hz";
    channels_message[7] = "HIP FLEXOR (G)    150 ms";

    string F1_F4_message = "F1  SAVE & BACK    F2  SELECT   F3  SAVE & STIM";

    //main lines
    cout << header("SETUP") << endl;
    cout << "  MIXED "  << "\n" << endl;
    cout << "  Press the encoder or F2 for detaled channel setup" << "\n" << endl;
    cout << "  LEFT LEG " << endl;

    int first_channel = 0;
    int last_channel = 4;
    int first_blok_lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);
    //cout << first_blok_lines_with_text << endl;

    cout << "  RIGHT LEG " << endl;
    first_channel = 4;
    last_channel = 8;
    int second_blok_lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);
    //cout << second_blok_lines_with_text << endl;

    // 26 - max hight, 4 - header + ending line, 6 - other printed lines
    int lines_without_text = 26 - 4 - first_blok_lines_with_text - second_blok_lines_with_text - 6;
    //cout << empty_line(lines_without_text) << endl;

    cout << F1_F4_message << endl;

    std::cin.ignore();

    return 0;
}


int SHAPE_OF_SIGNAL(string selected_channel = "1")
{
    //fill with text
    string channels_message[8] = {};

    channels_message[0] = "SQUARE PULSES (ALL)";
    channels_message[1] = "PATTERN (ALL)";
    channels_message[2] = "MIXED";

    string F1_F4_message = "  F1  BACK";

    //to switch between screens
    if (selected_channel == "1")
    {
        SQUARE_PULSES(selected_channel);
    }
    if (selected_channel == "2")
    {
        PATTERN(selected_channel);
    }
    if (selected_channel == "3")
    {
        MIXED(selected_channel);
    }

    //main lines
    cout << header("SETUP") << endl;
    cout << "  SHAPE OF SIGNAL " << "\n" << endl;

    int first_channel = 0;
    int last_channel = 8;
    int lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);

    // 26 - max hight, 4 - header + ending line
    int lines_without_text = 26 - 4 - lines_with_text;
    cout << empty_line(lines_without_text) << endl;
    cout << F1_F4_message << endl;

    std::cin.ignore();

    return 0;
}


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


int ALL_SENSORS(string selected_channel = "1")
{
    //fill with text
    string channels_message[8] = {};

    channels_message[0] = "ACTIVE";
    channels_message[1] = "PASSIVE";

    string F1_F4_message = "F1  BACK";

    //to switch between screens
    /*if (selected_channel == "1")
    {
        SHAPE_OF_SIGNAL(selected_channel);
    }
    if (selected_channel == "2")
    {
        SHAPE_OF_SIGNAL(selected_channel);
    }*/

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


int STIMULATION(string selected_channel = "1")
{
    //fill with text
    string channels_message[8] = {};
    channels_message[0] = "0.0 19 mA  Ankle Extensor (G)";
    channels_message[1] = "0.0 17 mA  Ankle Flexor (G)";
    channels_message[2] = "0.0 12 mA  Hip Extensor (G)";
    channels_message[3] = "0.0 20 mA  Hip Flexot (G)";
    channels_message[4] = "0.0 18 mA  Ankle Extensor (G)";
    channels_message[5] = "0.0 12 mA  Monophasic   250 us, 40 Hz";
    channels_message[6] = "0.0 20 mA  Biphasic    200 us, 40 Hz";
    channels_message[7] = "0.0 16 mA  Hip Flexor (G)";

    string F1_F4_message = "F1  STOP    F1  STOP & BACK    F1  STOP & SETUP    F1  STOP & MAIN";

    //main lines
    cout << header("ACCELERATION TIME: 00:05") << endl;
    cout << "  SIMULATION TIME: 00:00" << "\n" << endl;
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


int ELECTRODES_CONTACT(string selected_channel = "1")
{
    //fill with text
    string channels_message[8] = {};

    channels_message[0] = "connected";
    channels_message[1] = "connected";
    channels_message[2] = "disconnected";
    channels_message[3] = "connected";
    channels_message[4] = "connected";
    channels_message[5] = "connected";
    channels_message[6] = "disconnected";
    channels_message[7] = "connected";

    string F1_F4_message = "F1  BACK TO MAIN    F2 START    F3 REPEAT    F4 STIMULATION";

    //main lines
    cout << header("ELECTRODES' CONTACT TEST") << endl;
    cout << "  Please connect used channels and press F2" << "\n" << endl;

    int first_channel = 0;
    int last_channel = 8;
    int lines_with_text = connect_disconnect(channels_message, selected_channel, first_channel, last_channel);

    cout << "  If used channels are disconnected, check the connection\n  and press F3 to re-check" << "\n" << endl;

    // 26 - max hight, 4 - header + ending line, 4 - other
    int lines_without_text = 26 - 4 - lines_with_text - 5;
    cout << empty_line(lines_without_text) << endl;

    cout << F1_F4_message << endl;

    std::cin.ignore();

    return 0;
}


int SELF_TEST(string selected_channel = "1")
{
    //fill with text
    string channels_message[8] = {};

    channels_message[0] = "WITHOUT SENSORS";
    channels_message[1] = "ALL SENSORS";
    channels_message[2] = "FLEXIMETERS ONLY";
    channels_message[3] = "PRESSURE ONLY";

    string F1_F4_message = "F1  BACK";

    //to switch between screens
    /*if (selected_channel == "1")
    {
        SHAPE_OF_SIGNAL(selected_channel);
    }
    if (selected_channel == "2")
    {
        SHAPE_OF_SIGNAL(selected_channel);
    }
    if (selected_channel == "3")
    {
        SHAPE_OF_SIGNAL(selected_channel);
    }
    if (selected_channel == "4")
    {
        SHAPE_OF_SIGNAL(selected_channel);
    }*/

    //main lines
    cout << header("SELF TEST") << endl;
    cout << "  SENSORS CONFIGURATION:" << "\n" << endl;

    int first_channel = 0;
    int last_channel = 8;
    int lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);

    // 26 - max hight, 4 - header + ending line, 2 - other
    int lines_without_text = 26 - 4 - lines_with_text - 2;
    cout << empty_line(lines_without_text) << endl;

    cout << F1_F4_message << endl;

    std::cin.ignore();

    return 0;
}


int SETUP(string selected_channel = "1")
{
    //fill with text
    string channels_message[8] = {};

    channels_message[0] = "MAXIMUM CURRENT    20 mA";
    channels_message[1] = "STIMULATION TIME    1 min";
    channels_message[2] = "ACCELERATION TIME    5 sec";
    channels_message[3] = "SHAPE OF SIGNAL";
    channels_message[4] = "CURRENT LIMITS";

    string F1_F4_message = "F1  SAVE & BACK    F2  RESET";

    //to switch between screens
    if (selected_channel == "4")
    {
        SHAPE_OF_SIGNAL(selected_channel);
    }

    if (selected_channel == "5")
    {
        CURRENT_LIMITS(selected_channel);
    }

    //main lines
    cout << header("SETUP") << endl;

    int first_channel = 0;
    int last_channel = 8;
    int lines_with_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);

    // 26 - max hight, 4 - header + ending line
    int lines_without_text = 26 - 4 - lines_with_text;
    cout << empty_line(lines_without_text) << endl;

    cout << F1_F4_message << endl;

    std::cin.ignore();

    return 0;
}


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

int main()
{
    ELECTRODES_CONTACT("1");
    STIMULATION("1");
    ALL_SENSORS("1");
    //MAIN("F4", false);
    /*SHAPE_OF_SIGNAL("3");
    MAIN("1", true);
    SETUP("4");
    SETUP("5");*/

    return 0;
}
