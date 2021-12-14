#include <map>
#include <iostream>

using namespace std;

/*
int width = 64;
int height = 24;
*/

string channels_keys[8] = {"1", "2", "3", "4", "5", "6", "7", "8"};
string F1_F4_keys[4] = {"F1", "F2", "F3", "F4"};

map<string, string> channels_message = {
    { "1", "" },
    { "2", "" },
    { "3", "" },
    { "4", "" },
    { "5", "" },
    { "6", "" },
    { "7", "" },
    { "8", "" },
};

map<string, string> F1_F4_message = {
    { "F1", "" },
    { "F2", "" },
    { "F3", "" },
    { "F4", "" },
};

string header(string message){
    string head_line = "";
	head_line = "\n";
	head_line += "  " + message + "\n";
	head_line += "\n";
	return head_line;
}

string zero_line(int num){
    string z_line = "";
    for (int i = 1; i <= num; i++){z_line += "\n";}
	return z_line;
}

string text_line( string number, string message, string color){
    string t_line = "";
    if (color == "white"){color = "[30;47m";}
	if (color == "original"){color = "[0m";}
	t_line = "\033" + color + "  " + number + "  " + message + " \033[0m\n";
	return t_line;
}

string SETUP_scr(string selected_channel){
    channels_message["1"] = "MAXIMUM CURRENT    20 mA";
    channels_message["2"] = "STIMULATION TIME    1 min";
    channels_message["3"] = "ACCELERATION TIME    5 sec";
    channels_message["4"] = "SHAPE OF SIGNAL";
    channels_message["5"] = "CURRENT LIMITS";

    F1_F4_message["f1"] = "SAVE & BACK";
    F1_F4_message["f2"] = "RESET";

    string color = "";
    for (char i : channels_keys){
        if (selected_channel == channels_keys[i]){color = "white";}
        else {color = "original"};

        cout << text_line(number=channel_num, message=channel_text, color=color) << endl;
        }:
    }


    return "0";
}


int main()
{
std::string kek[8] = {"1", "2", "3", "4", "5", "6", "7", "8"};
    SETUP_scr("1");
/*
    {
        printf("\n");
        printf("  SETUP \n");
        printf("\n");
        printf("\n");
        printf("\033[30;47m  1  MAXIMUM CURRENT    20 mA \033[0m\n");
        printf("  2  STIMULATION TIME    1 min \n");
        printf("  3  ACCELERATION TIME \t 5 sec \n");
        printf("  4  SHAPE OF SIGNAL \n");
        printf("  5  CURRENT LIMITS \n");
        printf("\n");
        printf("\n");
        printf("\n");
        printf("\n");
        printf("\n");
        printf("\n");
        printf("\n");
        printf("\n");
        printf("\n");
        printf("\n");
        printf("\n");
        printf("\n");
        printf("\n");
        printf("  F1 SAVE & BACK    F2 RESET\n");
       // printf("\n");
        return 0;
    }
*/
}
