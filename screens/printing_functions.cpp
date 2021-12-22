#include <iostream>

using namespace std;

string error_line(string message, string mode = "normal")
{
    // 4 lines
    string err = "";
    string err_line = "";
    if (mode == "fatal")
    {
        err_line = "\033[37;41m\n  FATAL ERROR \n  " + message + " \033[0m" + "\n";

    }
    else
    {
        err_line = "\033[37;41m\n  ERROR \n  " + message + " \033[0m" + "\n";

    }

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
    if (color == "red")
	{
        color = "[37;41m";
    }
	t_line = "\033" + color + "  " + number + "  " + message + " \033[0m";
	return t_line;
}


int print_main_lines(string channels_message[8], string selected_channel, int first_channel, int last_channel)
{
    string channels_keys[10] = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"};
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

        if (channels_message[i] == "disconnected")
        {
            color = "red";
        }
        if (channels_message[i] == "connected")
        {
            color = "original";
        }
        if (ch_key == "10")
        {
            ch_key = "0";
        }

        if (ch_message != "")
        {
            cout << text_line(ch_key, ch_message, color) << endl;
            lines_with_text += 1;
        }
        else;

    }

    // * 2 becuase + \n
    return lines_with_text ;
}
