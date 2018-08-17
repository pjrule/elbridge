import geopandas as gpd
import numpy as np
import rtree
from shapely.geometry import Polygon
from geopandas.geoseries import GeoSeries, Point
from shapely.prepared import prep
from collections import defaultdict
import pysal
from copy import copy, deepcopy
from random import choice
from numba import jit

# districts' populations can differ by TOLERANCE%.
# Right now, this is implemented such that, if a district goes over quota by 0.1%, then the next district must go under quota by 0.1% when possible.
# This prevents "tolerance debt" from accumulating.
# For example, consider10 districts going over quota by 1%, which would require an 11th district to go over quota 10%.
TOLERANCE = 0.05  
N = 100

class Map:
    """
    Re-rendering a high-quality version of a district map at every allocation step is obviously not too fast.
    However, we can break a state's map up into a grid of squares of equal area in a rectangular bounding box.
    Then, we can precompute just once how much a particular precinct's allocation contributes to the map as a whole.
    With these precomptued matrices, we can update the map frame-by-frame extremely quickly with arbitrary resolution.   
    """
    def __init__(self, resolution, decay):
        """ Initializes the fixed-resolution representation of the GeoDataFrame. """
        if not hasattr(self, 'df'):
            self.df = gpd.GeoDataFrame() # this should never be true for specific states
            self.n_districts = 1

        """
        The U.S. Census Bureau tends to use the Albers Equal Area projection.
        See:
        https://lehd.ces.census.gov/doc/help/onthemap/OnTheMapImportTools.pdf
        https://TOLERANCEg.io/102003
        http://spatialreference.org/ref/esri/usa-contiguous-albers-equal-area-conic/
        """
        self.df = self.df.to_crs({'init': 'esri:102003'})
        self.adj = pysal.weights.Rook.from_dataframe(self.df)
        self.total_pop = np.array(self.df['white_pop'] + self.df['minority_pop'])
        self.pop_left = np.sum(self.total_pop)
        self.target = self.pop_left / self.n_districts
        self.district_pop_allocated = 0 

        dv_cols = {}
        rv_cols = {}
        for col in self.df.columns:
            # CONVENTION: DV starts with 'dv' and ends with year
            #             RV starts with 'rv' and ends with year
            if col.startswith('dv') and "share" not in col:
                dv_cols[col] = np.array(self.df[col])
            elif col.startswith('rv') and "share" not in col:
                rv_cols[col] = np.array(self.df[col])
        """
        Partisan leans of districts is computed using variable exponential decay, whichcan be adjusted to make historical elections fade more or less quickly.
        If a VTD has 0 Dem votes and 0 Rep votes for a given year, that year is simply not included as a term in the weighted average.
        TODO: interpolation?
        """
        area = np.zeros(len(self.df))
        for row in self.df.itertuples():    
            idx = getattr(row, 'Index')
            area[idx] = getattr(row, 'geometry').area

        # NOTE CONVENTIONS HERE
        total = self.df['white_pop'] + self.df['minority_pop']
        total[total==0] = 1 # avoid divide-by-zero
        self.df['minority_prop'] = self.df['minority_pop'] / total

        # Find geographical bounds (latitude/longitude)
        self.min_x = self.df.bounds['minx'].min()
        self.max_x = self.df.bounds['maxx'].max()
        self.min_y = self.df.bounds['miny'].min()
        self.max_y = self.df.bounds['maxy'].max()

        # Generate rectangular bounding box
        alpha = (self.max_x - self.min_x) / (self.max_y - self.min_y)
        s = np.sqrt(resolution / alpha)
        self.row_squares = int(np.ceil(s*alpha))
        self.col_squares = int(np.ceil(s))

        # Create concordances/rtree
        self.mapping_weights = np.zeros((self.col_squares, self.row_squares, len(self.df)))
        self.vtd_idx = rtree.index.Index()
        for df_row in self.df.itertuples():
            self.vtd_idx.insert(getattr(df_row, 'Index'), getattr(df_row, 'geometry').bounds)

        for col in range(self.col_squares):
            b_min_y = ((self.max_y - self.min_y) * col/self.col_squares) + self.min_y
            b_max_y = ((self.max_y - self.min_y) * (col + 1)/self.col_squares) + self.min_y
            for row in range(self.row_squares):
                b_min_x = ((self.max_x - self.min_x) * row/self.row_squares) + self.min_x
                b_max_x = ((self.max_x - self.min_x) * (row + 1)/self.row_squares) + self.min_x
                bounds = Polygon([(b_min_x, b_min_y), (b_min_x, b_max_y), (b_max_x, b_max_y), (b_max_x, b_min_y)])

                for fid in list(self.vtd_idx.intersection(bounds.bounds)):
                    if getattr(self.df.iloc[fid], 'geometry').intersects(bounds):
                        intersect = getattr(self.df.iloc[fid], 'geometry').intersection(bounds).area
                        self.mapping_weights[col,row,fid] = intersect / getattr(self.df.iloc[fid], 'geometry').area

        """
        Graph for allocation (used to make sure whitespace is not lost) 
        """
        self.districts = np.zeros(len(self.df))
        self.districts[0] = len(self.df)
        self.vtd_by_district = [list(self.df.index), []]
        self.current_district = 1
        
        self.graph = {}
        self.ws_graph = {}
        for idx in range(self.adj.n):
            self.graph[idx] = list(self.adj[idx].keys())
            self.ws_graph[idx] = list(self.adj[idx].keys())


    def next_frame(self):
        """ Render the next frame of the fixed-resolution representation. """
        pass

    #@jit
    def allocate(self, x, y):
        """
        If possible under equal population constraints, allocate the county at (x,y) to the current district if the county has not been previously allocated.
        If not possible, allocate the town at (x,y). If not possible, allocate the VTD at (x,y).
        Raw (x,y) coordinates from the NNs are bounded from (0,0) to (1,1).
        For instance, (0.5, 0.5) is halfway in between the min and max x and halfway in between the min and max y.
        """
        x_abs = ((self.max_x - self.min_x) * x) + self.min_x
        y_abs = ((self.max_y - self.min_y) * y) + self.min_y
        p = Point((x_abs, y_abs))
        vtd_idx = None
        for fid in list(self.vtd_idx.intersection(p.bounds)):
            # API: https://streamhsacker.com/2010/03/23/python-point-in-polygon-shapely/
            if getattr(self.df.iloc[fid], 'geometry').contains(p):
                vtd_idx = fid
                break
        
        if vtd_idx not in self.vtd_by_district[0]:
            return # already allocated

        if self.district_pop_allocated >= self.target * (1 - TOLERANCE) \
           and self.district_pop_allocated <= self.target * (1 + TOLERANCE):
           self.district_pop_allocated = 0
           self.vtd_by_district.append([])
        county_id = self.df.iloc[vtd_idx]['county']
        vtd_in_county = list(self.df[self.df['county'] == county_id].index)
        """
        Algorithm for allocating VTDs:
        1. Figure out if the county, or the remaining unallocated part of the county, 
           can be allocated wholly to the current district. This requires:

           a. The remaining county is contiguous to the current district.
           b. The remaining county’s population plus the current district’s population 
              less than or equal to the expected number of people per district, ± some very small ϵ.

              If the allocating the county results in two isolated regions of whitespace, 
              the smaller region of whitespace will be allocated to the district, and the population
              of this region will be added to the remaining county's population when checking the 
              equal population constraint. 

            If the remaining county can be allocated, do so. 
            If contiguity is violated, abort. Otherwise, proceed to step 2.
        """
        unallocated_in_county = []
        for idx in vtd_in_county:
            if idx in self.vtd_by_district[0]:
                unallocated_in_county.append(idx)
        if self.districts[self.current_district] > 0 and len(self._border_graph(unallocated_in_county)) == 0:
            return # no connection between county and current district

        county_pop = 0
        for idx in unallocated_in_county:
            county_pop += self.total_pop[idx]
        valid, surrounded = self.validate_update(unallocated_in_county)
        unallocated_in_county += surrounded
        for idx in surrounded:
            county_pop += self.total_pop[idx]

        if self.district_pop_allocated + county_pop <= self.target: # TODO (URGENT) tolerance
            print(unallocated_in_county)
            print(county_pop)
            self.update(unallocated_in_county, county_pop)
            return

        """
        2. Randomly remove cities (N times; this is the algorithm's main parameter) from the remaining county. 
        For a given removal process, abort when:
        a. The total population of the modified remaining county plus the population of the current district 
           is less than or equal to the population constraint. When this occurs, retain the modified county.
        b. There is no path between ward containing (x, y) and the current district. When this occurs, 
           discard the modified county.

        From all of the retained modified counties, choose the modification that results in the most number
        of people being allocated. If no modified counties satisfy the population constraint, proceed to step 3.
        """
        candidates = {}
        for i in range(N):
            valid = True
            all_vtd = copy(unallocated_in_county) # master list of possible VTDs; the while loop whittles them down
            candidate_pop = county_pop
            while self.district_pop_allocated + candidate_pop > self.target and len(all_vtd) > 0:
                random_vtd = choice(all_vtd)
                city_id = self.df.iloc[random_vtd]['city']
                vtd_in_city = list(self.df[self.df['city'] == city_id].index)
                for idx in vtd_in_city:
                    if idx in self.vtd_by_district[0]:
                        all_vtd.remove(idx)
                        candidate_pop -= self.total_pop[idx]
                    else:
                        vtd_in_city.remove(idx)
                if len(self._border_graph(all_vtd)) == 0:
                    valid = False
                    break
            ws_valid, _ = self.validate_update(vtd_in_city)
            if valid and ws_valid and self.district_pop_allocated + county_pop <= self.target:
                candidates[candidate_pop] = all_vtd
                print("town elimination iteration %d: %d" % (i, all_vtd))
            print("town elimination iteration %d: invalid" % i)

        if len(candidates) > 0:
            best_pop = sorted(candidates.keys())[-1]
            best = candidates[best_pop]
            self.update(best, best_pop)
            return

        """ 3. Randomly remove individual VTDs from the remaining county under the constraints above. """
        candidates = {}
        for i in range(N):
            valid = True
            all_vtd = copy(unallocated_in_county) # master list of possible VTDs; the while loop whittles them down
            candidate_pop = county_pop
            while self.district_pop_allocated + candidate_pop > self.target and len(all_vtd) > 0:
                random_vtd = choice(all_vtd)
                all_vtd.remove(random_vtd)
                if len(self._border_graph([random_vtd])) == 0:
                    valid = False
                    break
            ws_valid, _ = self.validate_update(vtd_in_city)
            if valid and ws_valid and self.district_pop_allocated + county_pop <= self.target:
                candidates[candidate_pop] = all_vtd
                print("VTD elimination iteration %d: %d" % (i, all_vtd))
            print("VTD elimination iteration %d: invalid" % i)

        if len(candidates) > 0:
            best_pop = sorted(candidates.keys())[-1]
            best = candidates[best_pop]
            self.update(best, best_pop)

        """ 4. If 3 fails, abort. """
        return

    #@jit
    def update(self, allocated, pop):
            self.district_pop_allocated += pop
            for idx in allocated:
                self.vtd_by_district[0].remove(idx)
            self.vtd_by_district[self.current_district] += allocated
            self.districts[self.current_district] += len(allocated)
            for idx in allocated:
                connected_to = self.ws_graph[idx]
                for c_idx in connected_to:
                    self.ws_graph[c_idx].remove(idx)
                del self.ws_graph[idx]

    #@jit
    def _border_graph(self, unallocated):
        border_graph = defaultdict(list)
        for d_idx in self.vtd_by_district[self.current_district]:
            for c_idx in unallocated:
                if c_idx in self.graph[d_idx]:
                    border_graph[c_idx].append(d_idx)
                    border_graph[d_idx].append(c_idx)
        return border_graph
        
    #@jit
    def validate_update(self, allocated):
        """
        Validate a new allocation to verify that a new allocation doesn't create two or more pockets of separated whitespace.
        If valid, update the whitespace graph.
        """
        graph = deepcopy(self.ws_graph)
        for idx in allocated:
            for c in graph[idx]:
                graph[c].remove(idx)
            del graph[idx]
        
        first_key = list(graph.keys())[0]
        traversed = set([first_key])
        queue = graph[first_key]
        while len(queue) > 0:
            # DFS from Wikipedia :)
            key = queue.pop()
            traversed.add(key)
            for idx in graph[key]:
                if idx not in traversed:
                    queue.append(idx)

        if len(traversed) < len(self.ws_graph) - len(allocated):
            # CONVENTION: return the smaller part of the graph
            if len(traversed) < len(self.ws_graph) - len(allocated) - len(traversed):
                return False, list(traversed)
            return False, list(set(self.ws_graph.keys()) - set(allocated) - traversed)
        return True, []

    def finished(self):
        """ Returns None if allocation is not finished; returns district array with pre-generated statistics if finished. """
        return self.districts[0] == 0 # no more left to allocate

class District:
    pass