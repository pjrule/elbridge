import geopandas as gpd
import numpy as np
import rtree
import scipy.sparse
import scipy.optimize
import tqdm
import elbridge.mapgraph as mg
from shapely.geometry import Polygon
from geopandas.geoseries import GeoSeries, Point
from shapely.prepared import prep
from collections import defaultdict
import pysal
from copy import copy, deepcopy
from random import choice
from numba import jit
import matplotlib.pyplot as plt
from math import floor, ceil
from time import time

# districts' populations can differ by TOLERANCE%.
# Right now, this is implemented such that, if a district goes over quota by 0.1%, then the next district must go under quota by 0.1% when possible.
# This prevents "tolerance debt" from accumulating.
# For example, consider10 districts going over quota by 1%, which would require an 11th district to go over quota 10%.
TOLERANCE = 0.0025  
BISECT_TOLERANCE = 0.05
BISECT_MAX_ITER = 100
N = 100

class Map:
    """
    Re-rendering a high-quality version of a district map at every allocation step is obviously not too fast.
    However, we can break a state's map up into a grid of squares of equal area in a rectangular bounding box.
    Then, we can precompute just once how much a particular precinct's allocation contributes to the map as a whole.
    With these precomptued matrices, we can update the map frame-by-frame extremely quickly with arbitrary resolution.   
    """
    def __init__(self, density_resolution, geo_resolution):
        """ Initializes the fixed-resolution representation of the GeoDataFrame. """
        if not hasattr(self, 'df'):
            self.df = gpd.GeoDataFrame() # this should never be true for specific states
            self.n_districts = 1

        self.density_resolution = density_resolution
        self.geo_resolution = geo_resolution
        """
        The U.S. Census Bureau tends to use the Albers Equal Area projection.
        See:
        https://lehd.ces.census.gov/doc/help/onthemap/OnTheMapImportTools.pdf
        http://spatialreference.org/ref/esri/usa-contiguous-albers-equal-area-conic/
        """
        self.df = self.df.to_crs({'init': 'esri:102003'})
        self.adj = pysal.weights.Rook.from_dataframe(self.df)
        # Two VTDs in Wisconsin consist entirely of islands. 
        # For simplicity, we will fuse islands with their nearest neighbors by centroid distance.
        if len(self.adj.islands) > 0:
            for island_idx in self.adj.islands:
                distance = np.zeros(len(self.df))
                cent_x = self.df.iloc[island_idx].geometry.centroid.x
                cent_y = self.df.iloc[island_idx].geometry.centroid.y
                for idx, row in self.df.iterrows():
                    x_dist = getattr(row, 'geometry').centroid.x - cent_x
                    y_dist = getattr(row, 'geometry').centroid.y - cent_y
                    if idx not in self.adj.islands:
                        distance[idx] = np.sqrt((x_dist**2 + y_dist**2))
                
                for idx in np.argsort(distance):
                    if idx not in self.adj.islands and idx != island_idx:
                        closest_idx = idx
                        break

                # Fuse island demography/vote data with data of closest VTD
                for col in self.df.columns:
                    if col.startswith('dv') or col.startswith('rv') or col.endswith('pop'):
                        self.df.loc[closest_idx,col] += self.df.loc[island_idx,col]
            
            self.df = self.df.drop(self.adj.islands).reset_index()
            self.adj = pysal.weights.Rook.from_dataframe(self.df) # recalculate
    
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
        self.area = np.zeros(len(self.df))
        for row in self.df.itertuples():    
            idx = getattr(row, 'Index')
            self.area[idx] = getattr(row, 'geometry').area

        # NOTE CONVENTIONS HERE
        total = self.df['white_pop'] + self.df['minority_pop']
        total[total==0] = 1 # avoid divide-by-zero
        self.df['minority_prop'] = self.df['minority_pop'] / total

        # Find geographical bounds (latitude/longitude)
        self.min_x = self.df.bounds['minx'].min()
        self.max_x = self.df.bounds['maxx'].max()
        self.min_y = self.df.bounds['miny'].min()
        self.max_y = self.df.bounds['maxy'].max()
        # max radius of an inscribed circle within the rectangular bounding box
        if self.max_x - self.min_x > self.max_y - self.min_y:
            self.max_radius = (self.max_y - self.min_y) / 2
        else:
            self.max_radius = (self.max_x - self.min_x) / 2
            

        """
        For efficiency, we rasterize the map twice.
        One rasterization (typically relatively low-resolution) allows for quick updates of the district map.
        The other rasterization (typically relatively high-resolution) allows for quick approximation of population within a given radius.
        """
        self.alpha = (self.max_x - self.min_x) / (self.max_y - self.min_y)
        # Create concordances/rtree
        density_s = np.sqrt(density_resolution / self.alpha)
        self.density_n_rows = int(np.ceil(density_s))
        self.density_n_cols = int(np.ceil(density_s*self.alpha))
        self.density_width = (self.max_x - self.min_x) / self.density_n_cols
        self.density_height = (self.max_y - self.min_y) / self.density_n_rows

        # Placeholder density rasterization
        self.reset_rtree()
        self.square_density = np.zeros((1,1))
        # Placeholder geographical rasterization
        dist_s = np.sqrt(self.geo_resolution / self.alpha)
        self.geo_n_rows = int(np.ceil(dist_s))
        self.geo_n_cols = int(np.ceil(dist_s*self.alpha))
        self.geo_weights = scipy.sparse.lil_matrix((len(self.df), self.geo_n_rows*self.geo_n_cols))
        
        """
        Graph for allocation (used to make sure whitespace is not lost) 
        """
        self.districts = np.zeros(len(self.df))
        self.districts[0] = len(self.df)
        self.vtd_by_district = [list(self.df.index), []]
        self.current_district = 1
        self.debt = 0
        self.done = False

    def reset_rtree(self):
        self.vtd_idx = rtree.index.Index()
        for df_row in self.df.itertuples():
            self.vtd_idx.insert(getattr(df_row, 'Index'), getattr(df_row, 'geometry').bounds)

    def init_graph(self):
        self.graph = mg.MapGraph(self.adj)

    def load_density_mapping(self, density_mapping=None, density_mapping_save=None):
        if density_mapping:
            self.square_density = np.load(density_mapping)
        else:
            density = self.total_pop / self.area
            print("Rendering density rasterization...")
            self.square_density = np.zeros((self.density_n_cols, self.density_n_rows))
            for row in tqdm.tqdm(range(self.density_n_rows)):
                b_min_y = ((self.max_y - self.min_y) * row/self.density_n_rows) + self.min_y
                b_max_y = ((self.max_y - self.min_y) * (row + 1)/self.density_n_rows) + self.min_y
                for col in range(self.density_n_cols):
                    b_min_x = ((self.max_x - self.min_x) * col/self.density_n_cols) + self.min_x
                    b_max_x = ((self.max_x - self.min_x) * (col + 1)/self.density_n_cols) + self.min_x
                    bounds = Polygon([(b_min_x, b_min_y), (b_min_x, b_max_y), (b_max_x, b_max_y), (b_max_x, b_min_y)])

                    for fid in list(self.vtd_idx.intersection(bounds.bounds)):
                        if getattr(self.df.iloc[fid], 'geometry').intersects(bounds):
                            # HACK: buffered
                            # https://stackoverflow.com/questions/13062334/polygon-intersection-error-python-shapely
                            intersect = getattr(self.df.iloc[fid], 'geometry').buffer(0).intersection(bounds).area / bounds.area
                            self.square_density[col,row] += intersect * density[fid]
            if not density_mapping and density_mapping_save:
                np.save(density_mapping_save, self.square_density)

    def load_geo_mapping(self, geo_mapping=None, geo_mapping_save=None): 
        if geo_mapping:
                self.geo_weights = scipy.sparse.load_npz(geo_mapping)
        else:
            # Generate the geographical rasterization
            print("Rendering geographical rasterization...")
            for row in tqdm.tqdm(range(self.geo_n_rows)):
                b_min_y = ((self.max_y - self.min_y) * row/self.geo_n_rows) + self.min_y
                b_max_y = ((self.max_y - self.min_y) * (row + 1)/self.geo_n_rows) + self.min_y
                for col in range(self.geo_n_cols):
                    b_min_x = ((self.max_x - self.min_x) * col/self.geo_n_cols) + self.min_x
                    b_max_x = ((self.max_x - self.min_x) * (col + 1)/self.geo_n_cols) + self.min_x
                    bounds = Polygon([(b_min_x, b_min_y), (b_min_x, b_max_y), (b_max_x, b_max_y), (b_max_x, b_min_y)])

                    for fid in list(self.vtd_idx.intersection(bounds.bounds)):
                        if getattr(self.df.iloc[fid], 'geometry').intersects(bounds):
                            intersect = getattr(self.df.iloc[fid], 'geometry').buffer(0).intersection(bounds).area
                            self.geo_weights[fid,row*self.geo_n_cols+col] = intersect / bounds.area
            self.geo_weights = self.geo_weights.tocsr()
            if geo_mapping_save:
                scipy.sparse.save_npz(geo_mapping_save, self.geo_weights)
        
    def next_frame(self):
        """ Render the next frame of the fixed-resolution representation. """
        pass

    #@jit
    def allocate(self, x, y, i):
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
        county_pop = 0
        for idx in vtd_in_county:
            if idx in self.vtd_by_district[0]:
                unallocated_in_county.append(idx)
                county_pop += self.total_pop[idx]
        if not self.graph.contiguous(unallocated_in_county):
            return # no connection between county and current district

        if self.update(unallocated_in_county, county_pop):
            return # whole county allocated
        """
        2. Build a graph of cities or city fragments.
        A city's contiguity about its border is simply the union of the contiguity of its VTDs' borders.
        """

        # Find outer borders
        cities_in_county = defaultdict(dict)
        city_borders = {}
        orig_county_pop = county_pop

        for idx in unallocated_in_county:
            cities_in_county[self.df.iloc[idx]['city']][idx] = copy(self.graph[idx])

        for city_idx in cities_in_county:   
            city_border = set([])     
            for outer_idx in cities_in_county[city_idx]:
                for inner_idx in cities_in_county[city_idx][outer_idx]:
                    if inner_idx in cities_in_county[city_idx][outer_idx]:
                        cities_in_county[city_idx][outer_idx].remove(inner_idx)
                
                if len(cities_in_county[city_idx][outer_idx]) > 0:
                        city_border = city_border.union(cities_in_county[city_idx][outer_idx])
            city_borders[city_idx] = city_border
    
        # Find inner cities and eliminate
        outer_cities = {}
        for city_idx, border in city_borders.items():
            edges_within = 0
            for border_idx in border:
                if border_idx in unallocated_in_county:
                    edges_within += 1
            if edges_within < len(border):
                outer_cities[city_idx] = border

        removed = []
        while len(outer_cities) > 0 and self.graph.contiguous(unallocated_in_county) and not self.update(unallocated_in_county, county_pop):
            print(len(self.vtd_by_district[0]), '\t', "county_pop:", county_pop)
            # Randomly eliminate a city
            # TODO other objective functions here (population, distance)
            elim = choice(list(outer_cities.keys()))
            vtd = list(self.df[self.df['city'] == elim].index)
            for idx in vtd:
                if vtd in unallocated_in_county:
                    unallocated_in_county.remove(idx)
                    removed.append(idx)
                    county_pop -= self.total_pop[idx]
            del outer_cities[elim]

        if len(outer_cities) > 0:
            self.plot(i)

        else:
            unallocated_in_county += removed
            county_pop = orig_county_pop
            border_vtd = set([])
            while len(unallocated_in_county) > 0 and self.graph.contiguous(unallocated_in_county) and not self.update(unallocated_in_county, county_pop):
                print(len(self.vtd_by_district[0]), '\t', "[VTD] county_pop:", county_pop)
                elim = choice(unallocated_in_county)
                idx_in_county = 0
                
                for idx in self.graph[elim]:
                    if idx in unallocated_in_county:
                        idx_in_county += 1

                if idx_in_county < len(self.graph[elim]):
                    unallocated_in_county.remove(elim)
                    county_pop -= self.total_pop[elim]
                    border_vtd.add(elim)
            
            if len(unallocated_in_county) == 0 or not self.graph.contiguous(unallocated_in_county):
                unallocated_border = set([])
                for vtd in self.vtd_by_district[self.current_district]:
                    graph = self.graph[vtd]
                    for edge in graph:
                        if edge in self.vtd_by_district[0]:
                            unallocated_border.add(edge)

                random_vtd = choice(list(unallocated_border))
                self.update([random_vtd], self.total_pop[random_vtd])

    def plot(self, i):
        alloc = np.zeros(len(self.df))
        for district_idx, district in enumerate(self.vtd_by_district):
            for idx in district:
                alloc[idx] = district_idx
                
        self.df['alloc'] = alloc
        self.df.plot(column='alloc')
        plt.savefig('%d.png' % i, dpi=900, bbox_inches='tight')
        plt.close()
        
    def pop_bounds(self):
        """ Calculate the minimum and maximum population for the current district, taking debt into account. """
        lower_bound = floor(max(self.target * (1-TOLERANCE), self.target * (1-TOLERANCE) - self.debt))
        upper_bound =  ceil(min(self.target * (1+TOLERANCE), self.target * (1+TOLERANCE) - self.debt))
        return (lower_bound, upper_bound)

    #@jit
    def update(self, allocated, pop):
        # Check: equal population
        lower_bound, upper_bound = self.pop_bounds()
        if self.district_pop_allocated + pop >= lower_bound:
            if self.district_pop_allocated >= lower_bound and self.district_pop_allocated + pop >= upper_bound and pop <= lower_bound:
                # -> next district
                self.debt += self.district_pop_allocated - self.target
                self.current_district += 1 
                self.district_pop_allocated = 0
                self.vtd_by_district.append([])
                if self.current_district == self.n_districts:
                    self.vtd_by_district.append(self.vtd_by_district[0])
                    self.districts[self.n_districts] = len(self.vtd_by_district[0])
                    self.vtd_by_district[0] = 0
                    self.districts[0] = 0
                    self.done = True
                    return True

            elif self.district_pop_allocated + pop < upper_bound:
                pass # proceed normally

            else: # too big
                return False
        
        # Check: whitespace pockets
        enclosed_whitespace, extra_vtd = self.graph.validate(allocated)
        if enclosed_whitespace:
            extra_pop = 0
            for idx in extra_vtd:
                extra_pop += self.total_pop[idx]
            print("enclosed whitespace:", extra_vtd)
            return self.update(allocated + extra_vtd, pop + extra_pop)
            
        # Update (population, VTD counts, whitespace...)
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
        return True

    def finished(self):
        """ Returns None if allocation is not finished; returns district array with pre-generated statistics if finished. """
        return self.districts[0] == 0 # no more left to allocate

    @jit
    def abs_coords(self, x_rel, y_rel):
        x = (x_rel * (self.max_x - self.min_x)) + self.min_x
        y = (y_rel * (self.max_y - self.min_y)) + self.min_y
        return (x,y)

    @jit
    def rel_coords(self, x_abs, y_abs):
        x = (x_abs - self.min_x) / (self.max_x - self.min_x)
        y = (y_abs - self.min_y) / (self.max_y - self.min_y)
        return (x,y)

    @jit
    def people_to_geo(self, r_P):
        if r_P <= 0: return 0
        a = 0
        b = self.max_radius
        # https://en.wikipedia.org/wiki/Bisection_method#Algorithm
        for _ in range(BISECT_MAX_ITER):
            c = (a+b) / 2
            pop_c = self.local_pop(self.x, self.y, c)
            if pop_c >= r_P * (1 - BISECT_TOLERANCE) and pop_c <= r_P * (1 + BISECT_TOLERANCE):
                return c
            pop_a = self.local_pop(self.x, self.y, a)
            if (pop_a - r_P) * (pop_c - r_P) > 0:
                a = c
            else:
                b = c
        return 0

    @jit # for a magical ~1000x speedup! 
    def local_pop(self, x, y, r):
        """
        Estimate the population within a circle of radius r with center (x,y).
        Doesn't work well for very large radii; this is intended to be rather rough.

        """ 
        # TODO add bounds and then be done!
        if r <= 0: return 0
        r_orig = r
        min_x = max(0, floor((x - self.min_x - r) / self.density_width))
        max_x = min(ceil((x - self.min_x + r) / self.density_width), self.density_n_cols)
        min_y = max(0, floor((y - self.min_y - r) / self.density_height))
        max_y = min(ceil((y - self.min_y + r) / self.density_height), self.density_n_rows)
        # TODO could this somehow be more precise?
        c_x = int(round((min_x + max_x) / 2))
        c_y = int(round((min_y + max_y) / 2))
        r = int(min(self.max_radius, ceil(r / (0.5*(self.density_width + self.density_height)))))
        # redefine x,y in terms of rasterization coordinates
        
        bounded = self.square_density[min_x:max_x,min_y:max_y]
        mask = np.zeros_like(bounded)
        
        # Wikipedia pseudocode for midpoint algorithm: https://en.wikipedia.org/wiki/Midpoint_circle_algorithm#C_example
        # TODO licensing notes?
        x = r - 1
        y = 0
        dx = 1
        dy = 1
        err = dx - (r << 1)
        while x > y:
            mask[min(c_x + x - min_x, mask.shape[0]-1), min(c_y + y - min_y, mask.shape[1]-1)] = 1
            mask[min(c_x + y - min_x, mask.shape[0]-1), min(c_y + x - min_y, mask.shape[1]-1)] = 1
            mask[max(c_x - y - min_x, 0), min(c_y + x - min_y, mask.shape[1]-1)] = 1
            mask[max(c_x - x - min_x, 0), min(c_y + y - min_y, mask.shape[1]-1)] = 1
            mask[max(c_x - x - min_x, 0), max(c_y - y - min_y, 0)] = 1
            mask[max(c_x - y - min_x, 0), max(c_y - x - min_y, 0)] = 1
            mask[min(c_x + y - min_x, mask.shape[0]-1), max(c_y - x - min_y, 0)] = 1
            mask[min(c_x + x - min_x, mask.shape[0]-1), max(c_y - y - min_y, 0)] = 1

            if err <= 0:
                y += 1
                err += dy
                dy += 2
            if err > 0:
                x -= 1
                dx += 2
                err += dx - (r << 1)
                
        # scanline fill
        masked = 0
        for x in range(mask.shape[0]):
            ones = np.where(mask[x] == 1)[0]
            if len(ones) > 1:
                mask[x][ones[0]:ones[-1]+1] = 1
                masked += (ones[-1] - ones[0] + 1)
            
        if masked > 0:
            return np.sum(mask*bounded) / masked * np.pi * (r_orig ** 2) # * some big scale factor to get things in the right units
        else:
            x = min(max(0, floor((x - self.min_x) / self.density_width)), self.density_n_cols)
            y = min(max(0, floor((y - self.min_y) / self.density_height)), self.density_n_rows)
            return self.square_density[x,y] * np.pi * (r_orig ** 2)