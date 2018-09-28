import os 
import io


#def test():
#	return os.system("nrniv 2ndlayer.hoc")

def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

print("1")
p=1
p1=9
while p < p1:
	string = str("sprint($s1, \"%%s%%dv%d.%%s\", $s2, $5, $s3) \n"%(p))
	replace_line('recording.hoc', 5,  string)
	ppp = str("mpiexec -n %d nrniv -mpi 2ndlayer.hoc"%(p))
	os.system(ppp)
	#test()
	p+=1

print("2")