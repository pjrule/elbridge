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
from random import choice, randint, random, shuffle
from numba import jit
import matplotlib.pyplot as plt
from math import floor, ceil
from copy import copy

# districts' populations can differ by TOLERANCE%.
# Right now, this is implemented such that, if a district goes over quota by 0.1%, then the next district must go under quota by 0.1% when possible.
# This prevents "tolerance debt" from accumulating.
# For example, consider10 districts going over quota by 1%, which would require an 11th district to go over quota 10%.
TOLERANCE = 0.0025

BISECT_RTOL = 0.0001
BISECT_REL_XTOL = 0.01
BISECT_MAX_ITER = 80

P_RANDOM_ALLOC = 0.1

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
        self.centroids = np.zeros((2,len(self.df)))
        for row in self.df.itertuples():    
            idx = getattr(row, 'Index')
            self.area[idx] = getattr(row, 'geometry').area
            self.centroids[0][idx] = getattr(row, 'geometry').centroid.x
            self.centroids[1][idx] = getattr(row, 'geometry').centroid.y

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
            
        self.alpha = (self.max_x - self.min_x) / (self.max_y - self.min_y)

        
    def reset(self):
        # TODO move rtree reset into a separate init method? (it doesn't change—just needs to be reloaded once/unpickle)
        self.vtd_idx = rtree.index.Index()
        for df_row in self.df.itertuples():
            self.vtd_idx.insert(getattr(df_row, 'Index'), getattr(df_row, 'geometry').bounds)

        # Initialize graph of VTDs (unallocated)
        self.graph = mg.MapGraph(self.adj)

        # Initialize agent location to a random VTD centroid.
        vtd = randint(0, len(self.df)-1)
        cent = self.df.iloc[vtd].geometry.centroid
        self.x = cent.x
        self.y = cent.y 

        # Initialize county indices
        self.unallocated_in_county = defaultdict(list)
        self.vtd_to_county = {}
        for idx, vtd in self.df.iterrows():
            self.unallocated_in_county[getattr(vtd, 'county')].append(idx)
            self.vtd_to_county[idx] = getattr(vtd, 'county')

        # Initialize city indices
        self.unallocated_in_city = defaultdict(list)
        self.vtd_to_city = {}
        for idx, vtd in self.df.iterrows():
            self.unallocated_in_city[getattr(vtd, 'city')].append(idx)
            self.vtd_to_city[idx] = getattr(vtd, 'city')

        # State management
        self.vtd_by_district = [list(self.df.index), []]
        self.current_district = 1
        self.debt = 0
        self.done = False
        self.i = 0

    def load_density_mapping(self, density_mapping=None, density_mapping_save=None):
        density_s = np.sqrt(self.density_resolution / self.alpha)
        self.density_n_rows = int(np.ceil(density_s))
        self.density_n_cols = int(np.ceil(density_s*self.alpha))
        self.density_width = (self.max_x - self.min_x) / self.density_n_cols
        self.density_height = (self.max_y - self.min_y) / self.density_n_rows

        if density_mapping:
            self.square_density = np.load(density_mapping)
        else:
            density = self.total_pop / self.area
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
            if density_mapping_save:
                np.save(density_mapping_save, self.square_density)

    def load_geo_mapping(self, geo_mapping=None, geo_mapping_save=None): 
        dist_s = np.sqrt(self.geo_resolution / self.alpha)
        self.geo_n_rows = int(np.ceil(dist_s))
        self.geo_n_cols = int(np.ceil(dist_s*self.alpha))
        self.geo_weights = scipy.sparse.lil_matrix((len(self.df), self.geo_n_rows*self.geo_n_cols))

        if geo_mapping:
                self.geo_weights = scipy.sparse.load_npz(geo_mapping)
        else:
            # Generate the geographical rasterization
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
    def allocate(self, r_P, theta, to_vtd=None):
        """
        Coordinates are given in the (r_P, θ) system, where r_P is a proportion of the population (0-1) and θ is a direction (in radians).
        """
        if random() < P_RANDOM_ALLOC and self.vtd_by_district[self.current_district] and not to_vtd:
            border_vtds = []
            self.allocate(0, 0, to_vtd=choice(self.graph.unallocated_on_border(self.current_district)))
            return

        if not to_vtd:
            r_P_abs = min(max(0, r_P), 1) * self.total_pop.sum()
            r_G = self.people_to_geo(r_P_abs)
            to_x = min(max(self.min_x, r_G * np.cos(theta) + self.x), self.max_x)
            to_y = min(max(self.min_y, r_G * np.sin(theta) + self.y), self.max_y)
            p = Point((to_x, to_y))
            
            vtd_idx = None
            for fid in list(self.vtd_idx.intersection(p.bounds)):
                # API: https://streamhsacker.com/2010/03/23/python-point-in-polygon-shapely/
                if getattr(self.df.iloc[fid], 'geometry').contains(p):
                    vtd_idx = fid
                    break
            
            if vtd_idx in self.vtd_by_district[self.current_district]:
                self.x = to_x
                self.y = to_y
                return
            elif not vtd_idx or vtd_idx not in self.vtd_by_district[0]:
                return # already allocated or out of bounds
        else:
            vtd_idx = to_vtd
            to_x = self.centroids[0][vtd_idx]
            to_y = self.centroids[1][vtd_idx]

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
        county_pop = 0
        county = self.df.iloc[vtd_idx]['county']
        for idx in self.unallocated_in_county[county]:
            county_pop += self.total_pop[idx]
        if not self.graph.contiguous(self.unallocated_in_county[county]):
            return # no connection between county and current district
        if to_vtd:
            self.x = to_x
            self.y = to_y
            self.i += 1
        if self.update(self.unallocated_in_county[county], county_pop):
            self.i += 1
            return # whole county allocated
        
        """
        Algorithm for allocating VTDs (cont'd):

        2. If updating fails due to population constraints, remove cities (whole or fractional) on the border of the allocation.
        Do this greedily until the constraints are satisfied, removing the cities farthest from (x,y) first.
    
        3. If updating fails due to population constraints, remove VTDs on the border of the allocation.
        Do this greedily until the constraints are satisfied, removing the VTDs farthest from (x,y) first.
        """
        vtds = copy(self.unallocated_in_county[county])

        tested = set([])
        tested_cities = set([])
        distances = np.flip(np.argsort(np.sqrt((self.centroids[0][vtds] - self.x)**2 + (self.centroids[1][vtds] - self.y)**2)))
        border_vtds = set(self.graph.border_vtds(vtds))
        last_idx = 0
        while len(tested_cities) < len(set([self.vtd_to_city[vtd] for vtd in vtds])) and len(tested) < len(vtds):
            farthest_vtd = None
            for idx, vtd_idx in enumerate(distances[last_idx:]):
                if vtds[vtd_idx] not in tested and self.vtd_to_city[vtds[vtd_idx]] not in tested_cities: #and vtds[vtd_idx] in border_vtds:
                    farthest_vtd = vtds[vtd_idx]
                    last_idx = idx + 1
                    break
            if not farthest_vtd:
                break

            test_vtds = copy(vtds)
            removed = []
            for city_vtd in self.unallocated_in_city[self.vtd_to_city[farthest_vtd]]:
                if city_vtd in test_vtds:
                    test_vtds.remove(city_vtd)
                    removed.append(city_vtd)    

            if self.graph.contiguous(test_vtds):
                if self.update(test_vtds, sum([self.total_pop[vtd] for vtd in test_vtds])):
                    return
                else:
                    for vtd in removed:
                        vtds.remove(vtd)
                    tested = set([])
                    distances = np.flip(np.argsort(np.sqrt((self.centroids[0][vtds] - self.x)**2 + (self.centroids[1][vtds] - self.y)**2)))
                    border_vtds = set(self.graph.border_vtds(vtds))
                    last_idx = 0
            else:
                tested.add(farthest_vtd)
                tested_cities.add(self.vtd_to_city[farthest_vtd])

        # TODO clean up to avoid duplication
        tested = set([]) # TODO should this be here?
        distances = np.flip(np.argsort(np.sqrt((self.centroids[0][vtds] - self.x)**2 + (self.centroids[1][vtds] - self.y)**2)))
        border_vtds = set(self.graph.border_vtds(vtds))
        last_idx = 0
        while len(vtds) > 0 and len(tested) < len(vtds):
            farthest_vtd = None
            for idx, vtd_idx in enumerate(distances[last_idx:]):
                if vtds[vtd_idx] not in tested and vtds[vtd_idx] in border_vtds:
                    farthest_vtd = vtds[vtd_idx]
                    last_idx = idx + 1
                    break
            if not farthest_vtd:
                break

            test_vtds = copy(vtds)
            test_vtds.remove(farthest_vtd)
            if self.graph.contiguous(test_vtds):
                if self.update(test_vtds, sum([self.total_pop[vtd] for vtd in test_vtds])):
                    return
                else:
                    vtds.remove(farthest_vtd)
                    tested = set([])
                    distances = np.flip(np.argsort(np.sqrt((self.centroids[0][vtds] - self.x)**2 + (self.centroids[1][vtds] - self.y)**2)))
                    border_vtds = set(self.graph.border_vtds(vtds))
                    last_idx = 0
            else:
                tested.add(farthest_vtd)
                
        # last resort: allocate a single VTD
        if self.graph.contiguous([vtd_idx]) and vtd_idx in self.vtd_by_district[0]:
            self.update([vtd_idx], self.total_pop[vtd_idx])

    def reset_district(self):
        """ Reset the current district and restart in another location. """
        for idx in self.vtd_by_district[self.current_district]:
            self.unallocated_in_county[self.vtd_to_county[idx]].append(idx)
            self.unallocated_in_city[self.vtd_to_city[idx]].append(idx)
        self.vtd_by_district[0] += self.vtd_by_district[self.current_district]
        self.vtd_by_district[self.current_district] = []
        self.district_pop_allocated = 0
        self.graph.reset_district()
        # restart somewhere else
        border_vtds = set([])
        for district in range(1, self.current_district):
            border_vtds = border_vtds.union(self.graph.unallocated_on_border(district))
        border_vtds = list(border_vtds)
        shuffle(border_vtds)
        #border_vtds = copy(self.vtd_by_district[0])
        shuffle(border_vtds)
        for vtd in border_vtds:
            self.allocate(0, 0, to_vtd=vtd)
            if self.district_pop_allocated > 0:
                break
        
    def f_bounds(self):
        """ Calculate the minimum and maximum population for the current district, taking debt into account. """
        lower_bound = floor(max(self.target * (1-TOLERANCE), self.target * (1-TOLERANCE) - self.debt))
        upper_bound =  ceil(min(self.target * (1+TOLERANCE), self.target * (1+TOLERANCE) - self.debt))
        return (lower_bound, upper_bound)

    #@jit
    def update(self, allocated, pop):
        # Check: equal population
        lower_bound, upper_bound = self.f_bounds()
        allocated = copy(allocated)
        next_district = False
        #print('allocated:', self.district_pop_allocated, '\tbounds:', lower_bound, upper_bound, '\tpop:', pop)
        if self.district_pop_allocated + pop >= lower_bound:
            if self.district_pop_allocated >= lower_bound and self.district_pop_allocated + pop >= upper_bound and pop <= lower_bound:
                # -> next district
                next_district = True
                if self.current_district == self.n_districts - 1:
                    self.vtd_by_district.append(copy(self.vtd_by_district[0]))
                    self.vtd_by_district[0] = []
                    self.done = True
                    return True
 
            elif self.district_pop_allocated + pop >= upper_bound: # too big
                return False
        
        # Check: whitespace pockets
        enclosed_whitespace = self.graph.validate(allocated)
        self.enclosed = enclosed_whitespace
        if enclosed_whitespace:
            extra_pop = 0
            for idx in enclosed_whitespace:
                extra_pop += self.total_pop[idx]
            return self.update(allocated + enclosed_whitespace, pop + extra_pop)
            
        # Update (population, VTD counts, whitespace...)
        if next_district:
            self.debt += self.district_pop_allocated - self.target
            self.current_district += 1 
            self.district_pop_allocated = 0
            self.vtd_by_district.append([])
        self.district_pop_allocated += pop
        for idx in allocated:
            #print('allocating', idx)
            self.vtd_by_district[0].remove(idx)
            self.unallocated_in_city[self.vtd_to_city[idx]].remove(idx)
            self.unallocated_in_county[self.vtd_to_county[idx]].remove(idx)
        self.vtd_by_district[self.current_district] += allocated
        self.graph.allocate(allocated, next_district)
        return True

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
        if r_P <= 0:
            return 0
        elif r_P > self.total_pop.sum():
            return self.max_radius

        a = 0
        b = self.max_radius
        # https://en.wikipedia.org/wiki/Bisection_method#Algorithm
        f_a = self.local_pop(self.x, self.y, a) - r_P
        for _ in range(BISECT_MAX_ITER):            
            c = (a+b) / 2
            f_c = self.local_pop(self.x, self.y, c) - r_P
            if abs(f_c) <= BISECT_REL_XTOL*r_P or b - a <= BISECT_RTOL:
                return c
            if f_a*f_c > 0:
                a = c
                f_a = f_c
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
        r = int(min(self.max_radius, ceil(r_orig / (0.5*(self.density_width + self.density_height)))))
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
            mask[max(c_x - x - min_x, 0), max(c_y - y - min_y, 0)] = 1
            mask[max(c_x - x - min_x, 0), min(c_y + y - min_y, mask.shape[1]-1)] = 1
            mask[min(c_x + x - min_x, mask.shape[0]-1), max(c_y - y - min_y, 0)] = 1
            mask[min(c_x + x - min_x, mask.shape[0]-1), min(c_y + y - min_y, mask.shape[1]-1)] = 1
            
            mask[max(c_x - y - min_x, 0), max(c_y - x - min_y, 0)] = 1
            mask[max(c_x - y - min_x, 0), min(c_y + x - min_y, mask.shape[1]-1)] = 1
            mask[min(c_x + y - min_x, mask.shape[0]-1), max(c_y - x - min_y, 0)] = 1
            mask[min(c_x + y - min_x, mask.shape[0]-1), min(c_y + x - min_y, mask.shape[1]-1)] = 1
            
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
                masked += ones[-1] - ones[0] + 1

        if masked == 0:
            x = min(max(0, floor((x - self.min_x) / self.density_width)), self.density_n_cols)
            y = min(max(0, floor((y - self.min_y) / self.density_height)), self.density_n_rows)
            return min(self.square_density[x,y] * 4 * (r_orig ** 2), self.total_pop.sum())
        
        return min(np.mean(mask*bounded) * 4 * (r_orig ** 2), self.total_pop.sum())

    def gen_alloc(self):
        """ Refresh the 'alloc' column in self.df for debugging. """
        alloc = np.zeros(len(self.df))
        for district_idx, district in enumerate(self.vtd_by_district):
            for idx in district:
                alloc[idx] = district_idx
        self.df['alloc'] = alloc

    def plot(self, name, x=None, y=None):
        """ Plots the map for debugging. """
        self.gen_alloc()
        self.df.plot(column='alloc', vmin=0, vmax=self.n_districts)
        plt.plot(self.x, self.y, 'g+')
        if x and y:
            plt.plot(x, y, 'r+')
        plt.savefig(name, bbox_inches='tight')
        plt.close()

    @jit
    def true_local_pop(self, x, y, r):
        """
        Calculates local population by geometric intersection.
        More precise but much slower than the rasterization-based method; included for validation.
        """
        bounds = Point((x, y)).buffer(r)
        pop = 0
        for fid in list(self.vtd_idx.intersection(bounds.bounds)):
            if getattr(self.df.iloc[fid], 'geometry').intersects(bounds):
                intersect = getattr(self.df.iloc[fid], 'geometry').buffer(0).intersection(bounds).area
                pop += self.total_pop[fid] * (intersect / getattr(self.df.iloc[fid], 'geometry').area) 
        return pop
