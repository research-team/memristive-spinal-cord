#ifndef PRINTING_FUNCTION_H
#define PRINTING_FUNCTION_H

using namespace std;

std::string error_line(std::string message, std::string mode = "normal");
std::string header(std::string message);
std::string empty_line(int num);
std::string text_line(std::string number, std::string message, std::string color);
int print_main_lines(std::string channels_message[8], std::string selected_channel, int first_channel, int last_channel);

#endif
