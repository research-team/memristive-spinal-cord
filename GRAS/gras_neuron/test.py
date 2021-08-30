from Interface import *
from Interface import py_add_gen, py_connect_ind, py_connect_gen


x = Py_Group('a')
a = Py_Group('gen', 1, 'g')
py_add_gen(a, 0, 1, 0.025)

py_connect_gen(a, x, 0.025, 0.3, 50)


