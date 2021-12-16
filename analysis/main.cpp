#include <map>
#include <iostream>

using namespace std;

/*
int width = 73;
int height = 26;
*/

string channels_keys[8] = {"1", "2", "3", "4", "5", "6", "7", "8"};



string header_and_ending(string message)
{
    string head_line = "";
	head_line = "\n";
	head_line += "  " + message;
	//head_line += "  " + message;
	//head_line += "\n";
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
	t_line = "\033" + color + "  " + number + "  " + message + " \033[0m";
	return t_line;
}


int print_main_lines(string channels_message[8], string selected_channel, int first_channel, int last_channel)
{
    string channels_keys[8] = {"1", "2", "3", "4", "5", "6", "7", "8"};
    //int length_of_channels_keys = sizeof(channels_keys)/sizeof(channels_keys[0]);

    string color = "";
    int lines_with_text = 0;
    for (int i = first_channel; i < last_channel; i++)
    {
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

    // 24 - height of the screen, 6 - lines for the header and ending, 3 - ??
    int lines_without_text = 24 - 6 - 3 - lines_with_text;
    //cout << empty_line(lines_without_text) << endl;

    return lines_without_text;
}


int SHAPE_OF_SIGNAL(string selected_channel = "1")
{
    //fill with text
    string channels_message[8] = {};

    channels_message[0] = "SQUARE PULSES (ALL)";
    channels_message[1] = "PATTERN (ALL)";
    channels_message[2] = "MIXED";

    string F1_F4_message = "F1  BACK";

    //main lines
    cout << header_and_ending("SETUP") << endl;
    cout << "  SHAPE OF SIGNAL " << "\n" << endl;

    int first_channel = 0;
    int last_channel = 8;
    int lines_without_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);

    cout << empty_line(lines_without_text) << endl;
    cout << header_and_ending(F1_F4_message) << endl;

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
    cout << header_and_ending("SETUP") << endl;
    cout << "  CURRENT LIMITS " << "\n" << endl;
    cout << "  LEFT LEG " << endl;

    int first_channel = 0;
    int last_channel = 4;
    int usless_lines_without_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);

    cout << "  RIGHT LEG " << endl;
    first_channel = 4;
    last_channel = 8;
    int lines_without_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);
    lines_without_text = lines_without_text - (24 - usless_lines_without_text);
    cout << empty_line(lines_without_text) << endl;

    cout << header_and_ending(F1_F4_message) << endl;

    std::cin.ignore();

    return 0;
}


int SETUP_scr(string selected_channel = "1")
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
    cout << header_and_ending("SETUP") << endl;

    int first_channel = 0;
    int last_channel = 8;
    int lines_without_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);
    cout << empty_line(lines_without_text) << endl;

    cout << header_and_ending(F1_F4_message) << endl;

    std::cin.ignore();

    return 0;
}


int MAIN_scr(string selected_channel = "1")
{
    //fill with text
    string channels_message[8] = {};

    channels_message[0] = "PATIENT ID: ----";

    string F1_F4_message = "F1  CURSOR LEFT    F2  CURSOR RIGHT    F3  SETUP    F4  TEST & STIM";

    //to switch between screens
    if (selected_channel == "F3")
    {
        SETUP_scr(selected_channel);
    }

    /*if (selected_channel == "F4")
    {
        CURRENT_LIMITS(selected_channel);
    }*/

    //main lines
    cout << header_and_ending("MAIN") << endl;
    cout << "  Please select patient in the web application or" << "\n" << endl;
    //cout << "  or" << "\n" << endl;
    cout << "  Enter user ID:" << "\n" << endl;
    cout << "  - Use the encoder to select a digit" << "\n" << endl;
    cout << "  - Use F1 and F2 to change the position of the cursor" << "\n" << endl;
    //cout << "\n" << endl;

    int first_channel = 0;
    int last_channel = 2;
    int lines_without_text = print_main_lines(channels_message, selected_channel, first_channel, last_channel);
    cout << "  DATE: YYYY-MM-DD" << "\n" << endl;
    cout << "  PATIENT NAME: ----" << "\n" << endl;
    cout << "  BIRTH DATE: YYYY-MM-DD" << "\n" << endl;

    //lines_without_text = 24 - 7 - 9 - 5 - 3 ;
    cout << empty_line(4) << endl;

    cout << header_and_ending(F1_F4_message) << endl;

    return 0;
}

int main()
{
    cout << "111111111111111111111111111111111111111111111111111111111111111111111111111111111" << endl;
    //MAIN_scr("4");
    //SETUP_scr("5");

    return 0;
}
