/* Minimal graph computations for district allocation. */

#include <vector>
#include <stack>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include "graph.h"
#define NO_DISTRICT 0

namespace graph {
    VTD::VTD(int _id){
        id = _id;
        district = 0;
    }

    void VTD::add_neighbor(VTD* vtd){
        neighbors.push_back(vtd);
    }

    bool VTD::is_neighbor(int id){
        for(auto vtd: neighbors){
            if(vtd->id == id){
                return true;
            }
        }
        return false;
    }

    int VTD::get_id(){
        return id;
    }
    int VTD::get_district(){
        return district;
    }
    void VTD::set_district(int _district){
        district = _district;
    }

    Graph::Graph(){
        curr_district = 0;
        ws_left = 0;
    }

    void Graph::add(int id, std::vector<int> neighbors){
        /*
        Add a VTD to the Graph and link it to its neighbors.
        If its neighbors don't exist, create them.
        If the VTD already exists, do nothing. 
        */
       VTD* parent;
        if(vtds.find(id) != vtds.end()){
            parent = &vtds.at(id);
        } else {
            parent = new VTD(id);
            vtds.insert({id, *parent});
            ws_left++;
        }
        // Add nonexistent neighbors
        for(int neighbor: neighbors){
            if(vtds.find(neighbor) == vtds.end()){
                vtds.insert({neighbor, *(new VTD(neighbor))});
                ws_left++;
            } 
            vtds.at(id).add_neighbor(&(vtds.at(neighbor)));
        }
    }

    bool Graph::shared_border(std::vector<int> unallocated, bool any){
        /* Given an array of VTD IDs, determine whether at least one of the VTDs is contiguous to the current (or prev) district. */
        std::vector<int> neighbor_vtds = curr_vtds;
        if(any){
            for(int unalloc_vtd: unallocated){
                for(auto vtd: vtds){
                    if(vtd.second.get_district() != NO_DISTRICT){
                        bool neighboring = vtd.second.is_neighbor(unalloc_vtd);
                        if(neighboring){
                            return true;
                        }
                    }
                }
            }
        } else {
            for(int unalloc_vtd: unallocated){
                for(int vtd: neighbor_vtds){
                    bool neighboring = vtds.at(vtd).is_neighbor(unalloc_vtd);
                    if(neighboring){
                        return true;
                    }
                }
            }
        }
        return false;
    }

    std::vector<int> Graph::border_vtds(std::vector<int> unallocated){
        /* 
         * Given an array of IDs of unallocated VTDs, determine which of the VTDs are on the border of the allocation.
         * A VTD is said to be on the border of an allocation if, after the allocation, it will have at least one unallocated neighbor.
         */
        std::sort(unallocated.begin(), unallocated.end());
        std::vector<bool> on_border(vtds.size());
        for(int vtd_idx: unallocated){
            auto vtd = vtds.at(vtd_idx);
            int district = vtd.get_district();
            if(!on_border[district]){
                for(auto neighbor_vtd: vtd.neighbors){
                    bool in_unallocated = std::binary_search(unallocated.begin(), unallocated.end(), neighbor_vtd->get_id());
                    if(neighbor_vtd->get_district() == NO_DISTRICT && !in_unallocated){
                        on_border[vtd.get_id()] = true;
                        break;
                    }
                }
            }
        }
        std::vector<int> on_border_ids;
        for(int idx = 0; idx < on_border.size(); idx++){
            if(on_border[idx]){
                on_border_ids.push_back(idx);
            }            
        }
        return on_border_ids;
    }

    std::vector<int> Graph::unallocated_on_border(int district){
        /* Returns the IDs of the unallocated VTDs on the border of a given district. */
        std::vector<bool> on_border(vtds.size());
        for(auto vtd: vtds){
            if(vtd.second.get_district() == district){
                for(auto neighbor_vtd: vtd.second.neighbors){
                    if(neighbor_vtd->get_district() == NO_DISTRICT){
                        on_border[neighbor_vtd->get_id()] = true;
                    }
                }
            }
        }
        std::vector<int> on_border_ids;
        for(int idx = 0; idx < on_border.size(); idx++){
            if(on_border[idx]){
                on_border_ids.push_back(idx);
            }            
        }
        return on_border_ids;
    }


    bool Graph::contiguous(std::vector<int> unallocated){
        /*
        Given an array of VTD IDs, determine whether:
            1. the VTDs are contiguous w.r.t. each other
            2. either:
                (a) at least one of the VTDs is contiguous to the current district
                (b) at least one of the VTDs if contiguous to the previous district (if current district has no VTDs)
                (c) no districts have been allocated yet
        */

        // 1. the VTDs are contiguous w.r.t. each other
        // DFS basically copied from validate()
        if(unallocated.size() > 1){
            std::stack<int> to_visit;
            std::vector<int> visited(vtds.size());

            std::sort(unallocated.begin(), unallocated.end());
            to_visit.push(unallocated[0]);
            while(!to_visit.empty()){
                int vtd = to_visit.top();
                to_visit.pop();
                //visited[vtd] = true;
                for(auto n: vtds.at(vtd).neighbors){
                    bool vtd_in_unallocated = std::binary_search(unallocated.begin(), unallocated.end(), n->get_id());
                    if(n->get_district() == NO_DISTRICT && !visited[n->get_id()] && vtd_in_unallocated){
                        to_visit.push(n->get_id());
                        visited[n->get_id()] = true;
                    }
                }
            }
            int contiguous = 0;
            for(bool v: visited){
                if(v){ contiguous += 1; }
            }
            if(contiguous < unallocated.size()){
                return false;
            }
        }
        // 2c: no districts have been allocated yet
        if(curr_district == NO_DISTRICT) { return true; }
        // 2a: at least one of the VTDs is contiguous to the current district
        // 2b: at least one of the VTDs if contiguous to another district        
        if(curr_vtds.size() > 0){
            return shared_border(unallocated, false);
        } 
        return shared_border(unallocated, true);
    }

    std::vector<int> Graph::validate(std::vector<int> unallocated){
        /* Ensure that a a new allocation doesn't create two or more pockets of whitespace. */
        // DFS based on BFS (https://en.wikipedia.org/wiki/Breadth-first_search#Pseudocode)
        std::stack<int> to_visit;
        std::vector<bool> discovered(vtds.size());

        // initialize queue
        std::sort(unallocated.begin(), unallocated.end());
        for(auto v: vtds){
            bool vtd_in_unallocated = std::binary_search(unallocated.begin(), unallocated.end(), v.second.get_id());
            if(v.second.get_district() == NO_DISTRICT && !vtd_in_unallocated && v.second.neighbors.size() > 0){
                to_visit.push(v.second.get_id());
                discovered[v.second.get_id()] = true;
                break;
            }
        }
        
        int ws_visited = 0;
        while(!to_visit.empty()){
            int id = to_visit.top();
            to_visit.pop();
            for(auto n: vtds.at(id).neighbors){
                bool vtd_in_unallocated = std::binary_search(unallocated.begin(), unallocated.end(), n->get_id());
                if(n->get_district() == NO_DISTRICT && !discovered[n->get_id()] && !vtd_in_unallocated){
                    to_visit.push(n->get_id());
                    discovered[n->get_id()] = true;
                }
            }
            ws_visited++;
        }

        // whitespace not divided--no VTDs surrounded
        if(ws_visited == ws_left - unallocated.size()){
            return std::vector<int>(0);
        }

        std::vector<int> surrounded;
        if(ws_visited > (ws_left - ws_visited - unallocated.size())){
            // find and return the section of whitespace with the lower number of VTDs
            for(auto a: vtds){
                if(a.second.get_district() == NO_DISTRICT && !discovered[a.second.get_id()]){
                    bool vtd_in_unallocated = std::binary_search(unallocated.begin(), unallocated.end(), a.second.get_id());
                    if(!vtd_in_unallocated){
                        surrounded.push_back(a.second.get_id());
                    }
                }
            }
        } else {
            // visited section is smallest
            for(int idx = 0; idx < discovered.size(); idx++){
                if(discovered[idx]){
                    surrounded.push_back(idx);
                }
            }
        } 
        // TODO: return both large and small parts
        return surrounded;
    }

    void Graph::allocate(std::vector<int> unallocated, bool next){
        /*
        Allocate an array of VTDs to the current district.
        Assumes that the allocation has already between validated.
        */
        if(next || (curr_district == 0 && unallocated.size() > 0)){
            next_district();
        }
        for(int vtd: unallocated){
            if(vtds.find(vtd) != vtds.end()){
                vtds.at(vtd).set_district(curr_district);
                curr_vtds.push_back(vtd);
            }
        }
        ws_left -= unallocated.size();
    }

    void Graph::reset_district(){
        /* Return all VTDs in the current district to an unallocated state. */
        for(int idx = 0; idx < vtds.size(); idx++){
            if(vtds.at(idx).get_district() == curr_district){
                vtds.at(idx).set_district(NO_DISTRICT);
                ws_left++;
            }   
        }
        curr_vtds.clear();
    }

    void Graph::next_district(){
        curr_district++;
        std::vector<int> _prev;
        _prev = curr_vtds;
        curr_vtds.clear();
        prev_vtds = _prev;
    }

    void Graph::print_neighbors(int id){
        std::cout << "id: " << id << "\tdistrict: " << vtds.at(id).get_district() << "\tneighbors: " << vtds.at(id).neighbors.size() << std::endl;
        for(auto a: vtds.at(id).neighbors){
            std::cout << "\tneighbor id: " << a->get_id() << "\tdistrict: " << a->get_district() << "\tneighbors: " << a->neighbors.size() << std::endl; 
        }
    }

    void Graph::print_curr_district(){
        std::cout << curr_district << std::endl;
    }
}