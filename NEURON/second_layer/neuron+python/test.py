import os 
import io


def test():
	return os.system("mpiexec -n 2 nrniv -mpi -python network.py")

def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

print("1")

string = str("        path=str('./res/vMN%dr%dv%d'%(j, rank, 25))\n")
replace_line('network.py', 27,  'speed = 25\n')
replace_line('network.py', 273,  string)
#test()

string = str("        path=str('./res/vMN%dr%dv%d'%(j, rank, 50))\n")
replace_line('network.py', 27,  'speed = 50\n')
replace_line('network.py', 273,  string)
#test()

string = str("        path=str('./res/vMN%dr%dv%d'%(j, rank, 125))\n")
replace_line('network.py', 27,  'speed = 125\n')
replace_line('network.py', 273,  string)
test()

print("2")