#ifndef GRAPH_H
#define GRAPH_H

#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace graph {
    class VTD {
        int id;
        int district;
        public:
            VTD(int id);
            void add_neighbor(VTD* vtd);
            bool is_neighbor(int id);
            int get_id();
            int get_district();
            void set_district(int _district);
            std::vector<VTD*> neighbors;
    };

    class Graph {
        std::unordered_map<int, VTD> vtds;
        std::unordered_map<int, VTD> prev_vtds;
        std::unordered_map<int, VTD> curr_vtds;
        int curr_district;
        int ws_left;
        void next_district();
        public:
            Graph();
            void              add(int id, std::vector<int> neighbors);
            bool              shared_border(std::vector<int> unallocated, bool prev);
            bool              contiguous(std::vector<int> unallocated);
            std::vector<int>  validate(std::vector<int> unallocated); 
            void              allocate(std::vector<int> unallocated, bool next);
            void              print_neighbors(int id);
    };   
}

#endif