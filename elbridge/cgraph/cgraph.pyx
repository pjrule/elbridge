# distutils: language = c++
from graph cimport Graph
from libcpp.vector cimport vector
from libcpp cimport bool
from cpython cimport array
from cython.operator cimport dereference
import numpy as np
cimport numpy as np

cdef class CGraph:
    cdef Graph c_graph
    cdef dict _adj

    # NOTE: to make pickling work, this class does not contain a __cinit__().
    # Its nodes and edges must be explicitly initialized with init().
    def init(self, adj):
        self._adj = adj
        self.c_graph = Graph()
        self._build_graph(adj)

    def __getstate__(self):
        # TODO: pickling is only partially supported--the adjacency
        # matrix can only be used to reconstruct the initial state of the Graph.
        # This is probably OK for now (state is preserved in the higher-level
        # elbridge.graph.Graph object), but may be worth fixing eventually.
        return self._adj

    def __setstate__(self, state):
        self.init(state)

    def _build_graph(self, adj):
       cdef vector[int] *neighbors = new vector[int]()
       for idx in adj:
           for k in adj[idx]:
               neighbors.push_back(k)
           self.c_graph.add(idx, dereference(neighbors))
           neighbors.clear()

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

    def reset(self, max_district):
        self.c_graph.reset(max_district)

