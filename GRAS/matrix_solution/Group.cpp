#include <string>

using namespace std;

class Group {
public:
	Group() = default;

	string group_name;
	unsigned int id_start{};
	unsigned int id_end{};
	unsigned int group_size{};
};