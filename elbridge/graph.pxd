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
        bool shared_border(vector[int], bool)
        bool contiguous(vector[int])
        vector[int] validate(vector[int])
        void allocate(vector[int], bool)
        void print_neighbors(int) # for debugging
