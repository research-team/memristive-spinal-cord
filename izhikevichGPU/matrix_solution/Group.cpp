#include <string>

using namespace std;

class Group {
public:
	Group() = default;

	string group_name;
	int id_start{};
	int id_end{};
	int group_size{};
};