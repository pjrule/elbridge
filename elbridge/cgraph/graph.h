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
        std::vector<int> curr_vtds;
        std::vector<int> prev_vtds;
        int curr_district;
        int ws_left;
        void next_district();
        public:
            Graph();
            void              add(int id, std::vector<int> neighbors);
            bool              shared_border(std::vector<int> unallocated, bool prev);
            bool              contiguous(std::vector<int> unallocated);
            std::vector<int>  validate(std::vector<int> unallocated); 
            std::vector<int>  border_vtds(std::vector<int> unallocated);
            std::vector<int>  unallocated_on_border(int district);
            void              allocate(std::vector<int> unallocated, bool next);
            void              reset_district();
            void              print_neighbors(int id);
            void              print_curr_district(); 
    };   
}

#endif