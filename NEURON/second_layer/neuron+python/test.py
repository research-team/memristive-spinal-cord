import os 
import io


def test():
	return os.system("mpiexec -n 2 nrniv -mpi -python cpg.py")

def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

print("1")

for i in range(2):
	string = str('version = %d \n'%(i))
	replace_line('cpg.py', 13, string)
	test()

print("2")