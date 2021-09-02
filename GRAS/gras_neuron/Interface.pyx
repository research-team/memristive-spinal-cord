# distutils: language = c++
# distutils: sources = core.cu

from Interface cimport *

# def py_form_group(name_group, nrns_in_grp=50, mode='i', seg=1):
#     cdef string name = bytes(name_group, 'utf-8')
#     cdef int nrns = nrns_in_grp
#     cdef char model = ord(mode)
#     cdef int segs = seg
#     form_group(name, nrns, model, segs)

cdef class Py_Group():
    cdef Group cpp_group
    def __init__(self, name, nrns_in_group=50, model='i', segs=1):
        self.cpp_group = form_group(bytes(name, 'utf-8'), nrns_in_group, ord(model), segs)

# cdef class Py_generator

def py_connect_ind(pre_group : Py_Group, post_group: Py_Group, py_delay: float, py_weight: float, py_indegree=50, py_high=0):
    connect_fixed_indegree(pre_group.cpp_group, post_group.cpp_group, py_delay, py_weight, py_indegree, py_high)

def py_add_gen(group : Py_Group, py_start: float, py_end:float, py_freq: float):
    add_generator(group.cpp_group, py_start, py_end, py_freq)


def py_connect_gen(generator: Py_Group, group: Py_Group, py_delay: float, py_weight: float, py_indegree=50):
    conn_generator(generator.cpp_group, group.cpp_group, py_delay, py_weight, py_indegree)


def py_simulate():
    simulate()