# distutils: language = c++
from graph cimport Graph
from libcpp.vector cimport vector
from libcpp cimport bool
from cpython cimport array
from cython.operator cimport dereference
import numpy as np


cdef class CGraph:
    cdef Graph c_graph

    def __cinit__(self, adj):
        self.c_graph = Graph()
        
        cdef vector[int] *neighbors = new vector[int]()
        for idx in range(adj.n):
            for k in adj[idx]:
                neighbors.push_back(k)
            self.c_graph.add(idx, dereference(neighbors))
            neighbors.clear()

    def shared_border(self, unallocated, prev_district):
        cdef vector[int] *v = new vector[int]()
        for idx in unallocated:
            v.push_back(idx)
        return self.c_graph.shared_border(dereference(v), prev_district)

    def border_vtds(self, unallocated):
        cdef vector[int] *v = new vector[int]()
        for idx in unallocated:
            v.push_back(idx)
        return self.c_graph.border_vtds(dereference(v))

    def unallocated_on_border(self, district):
        return self.c_graph.unallocated_on_border(district)

    def contiguous(self, unallocated):
        cdef vector[int] *v = new vector[int]()
        for idx in unallocated:
            v.push_back(idx)
        return self.c_graph.contiguous(dereference(v))
        
    def validate(self, unallocated):
        cdef vector[int] v = unallocated
        return list(self.c_graph.validate(v))

    def allocate(self, unallocated, next_district):
        cdef vector[int] v = unallocated
        self.c_graph.allocate(v, next_district)

    def reset_district(self):
        self.c_graph.reset_district()

    def print_neighbors(self, id):
        self.c_graph.print_neighbors(id)
