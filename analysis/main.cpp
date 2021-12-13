#include <map>
using namespace std;

/*
int width = 64;
int height = 24;
*/

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
    for (int i = 1; i <= num; i++){
        z_line += "\n";
    }
	return z_line;
}

string text_line( string number, string message, string color){
    string t_line = "";
    if (color == "white")
		{color = "[30;47m";}
	if (color == "original")
		{color = "[0m";}
	t_line = "\033" + color + "  " + number + "  " + message + " \033[0m\n";
	return t_line;
}


int main()
{
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
}
