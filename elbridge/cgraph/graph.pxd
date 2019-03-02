# distutils: language=c++
# http://docs.cython.org/en/latest/src/userguide/wrapping_CPlusPlus.html#add-public-attributes
from libcpp cimport bool 
from libcpp.vector cimport vector

cdef extern from "graph.cpp":
    pass

cdef extern from "graph.h" namespace "graph":
    cdef cppclass Graph:
        Graph() except +
        void add(int, vector[int])
        bool contiguous(vector[int])
        vector[int] validate(vector[int])
        vector[int] border_vtds(vector[int])
        vector[int] unallocated_on_border(int)
        void allocate(vector[int], bool)
        void reset(int)
