import rtree
import scipy
import warnings
import numpy as np
from numba import jit
from math import ceil
from geopandas import GeoDataFrame
from shapely.geometry import box
from shapely.topology import TopologicalError
from typing import Iterable, Collection, Tuple, Dict

class Bitmap:
    """
    For a given districting plan, Elbridge maintains two bitmap images:
    a static bitmap image of the underlying state's population density, and
    a continuously updated bitmap showing the districting plan colored by
    district. The ``Bitmap`` class contains methods for creating and updating 
    these bitmaps; it also contains methods for computing geographical radii
    from radii specified in terms of population.
    """
    def __init__(self, gdf: GeoDataFrame, pop_col: str,
                 density_resolution: int, district_resolution: int,
                 bisect_rel_xtol: float = 0.01, bisect_rtol: float = 0.0001, 
                 bisect_max_iter: int = 80):
        """
        :param gdf: The state's GeoDataFrame (with each VTD as a row).
        :param pop_col: The column in the GeoDataFrame containing the total
            population of each VTD.
        :param density_resolution: The approximate number of pixels in the
            density bitmap.
        :param district_resolution: The approximate number of pixels in the 
            districting plan bitmap.
        :param bisect_rel_xtol: (optional) A tuning parameter for the
            bisection algorithm used to estimate geographical radii from
            people-based radii.
        :param bisect_rtol: (optional) A tuning parameter for the
            bisection algorithm used to estimate geographical radii from
            people-based radii.
        :param bisect_max_iter: (optional) The maximum number of iterations
            of the bisection algorithm used to estimate geographical radii from
            people-based radii.
        
        Though the bisection algorithm parameters are exposed, there is little
        reason to modify them--they have been carefully selected through
        experimentation to balance accuracy and speed.
        """
        self.df = gdf
        self.bisect_rtol = bisect_rtol
        self.bisect_rel_xtol = bisect_rel_xtol
        self.bisect_max_iter = bisect_max_iter
        self.rtree = self._rtree()
        self.state_pop = gdf[pop_col].sum()

        # All bitmaps are framed by the state's rectangular box.
        # Calculate rectangular bounding box dimensions
        self.min_x = self.df.bounds['minx'].min()
        self.max_x = self.df.bounds['maxx'].max()
        self.min_y = self.df.bounds['miny'].min()
        self.max_y = self.df.bounds['maxy'].max()
        # Calculate the radius of the rectangular bounding box's
        # maximally large inscribed circle
        if self.max_x - self.min_x > self.max_y - self.min_y:
            self.max_radius = (self.max_y - self.min_y) / 2
        else:
            self.max_radius = (self.max_x - self.min_x) / 2
        # Calculate the width:height ratio of the rectangular bounding box
        self.alpha = (self.max_x - self.min_x) / (self.max_y - self.min_y)
        # Calculate centroids
        self.centroids = np.zeros((len(self.df), 2))
        for idx, geom in enumerate(self.df.geometry):
            self.centroids[idx] = geom.centroid.coords[0]

        # Compute density bitmap
        self.density_map = self._density_map(density_resolution, pop_col)
        # Each pixel in the density bitmap corresponds to a rectangle of 
        # the calculated width and height on the map. This rectangle *should*
        # be very close to square, though it may deviate slightly at very
        # low resolutions.
        self.density_cols, self.density_rows = self.density_map.shape
        self.density_width = (self.max_x - self.min_x) / self.density_cols
        self.density_height = (self.max_y - self.min_y) / self.density_rows

        # Compute district map (used to quickly render a low-resolution
        # version of the districting plan)
        self.district_map, cols, rows = self._district_map(district_resolution)
        # Create an empty canvas for the low-resolution district plan rendering 
        self.frame = np.zeros((cols, rows))
        
    @jit
    def update_districts(self, vtds: Dict[int, int]) -> np.ndarray:
        """
        Updates the districting plan bitmap given a dictionary mapping
        incrementally allocated VTD indices to congressional districts.
        Intended for high-speed, low-resolution rendering of the districting
        plan at each allocation step. 

        :param vtds: A dictionary with VTD indices as keys and congressional
        district numbers as values.

        Returns the updated bitmap.
        """
        for vtd_idx, district in vtds.items():
            vtd_map = self.district_map[vtd_idx].reshape(self.frame.shape)
            self.frame += district * vtd_map
        return self.frame

    @jit
    def people_to_geo(self, x_rel: float, y_rel: float, r_P: float) -> float:
        """
        Estimates the geographical radius (in the map's units) for a given
        circle `(x_rel, y_rel, r_P)` in the people-based coordinate system
        using bisection.
        :param x_rel: The x-coordinate of the circle's center,
            specified in relative coordinates.
        :param x_rel: The y-coordinate of the circle's center,
            specified in relative coordinates.
        :param r_P: the radius of the circle, in people.

        Returns the estimated radius of the circle (in the map's units).
        """
        # Handle r_P values outside of bounds
        if r_P <= 0:
            return 0
        elif r_P >= self.state_pop:
            return self.max_radius

        # Based on https://en.wikipedia.org/wiki/Bisection_method#Algorithm
        a = 0
        b = self.max_radius
        f_a = self.local_pop(x_rel, y_rel, a) - r_P
        for _ in range(self.bisect_max_iter):
            c = (a * b) / 2
            f_c = self.local_pop(x_rel, y_rel, c) - r_P
            if (abs(f_c) <= self.bisect_rel_xtol * r_P or
               b - a <= self.bisect_rtol):
                return c
            if f_a * f_c > 0:
                a = c
                f_a = f_c
            else:
                b = c
        return 0

    @jit # for a magical ~1000x speedup! 
    def local_pop(self, x_rel: float, y_rel: float, r_abs: float) -> float:
        """
        Estimates the population of the circle `(x_rel, y_rel, r_abs)`
        based on the density bitmap using the midpoint algorithm for drawing
        rasterized circles.

        This technique doesn't work well for very large radii and in very dense
        regions; it's not guaranteed intended to work well with very small
        radii, either. It's rather rough, but it's accurate enough to make
        the allocation process dramatically more efficient.

        :param x_rel: The x-coordinate of the circle's center,
            specified in relative coordinates.
        :param y_rel: The y-coordinate of the circle's center,
            specified in relative coordinates.
        :param r_abs: The radius of the circle's center,
            specified in the map's units.
        """
        # Handle `r_abs` values outside of bounds
        if r_abs <= 0: 
            return 0
        elif r_abs >= self.max_radius:
            return self.state_pop

        # Find the bounds (in pixels) of the rectangle that inscribes
        # the circle
        min_x = max(0, int((x_rel - self.min_x - r_abs) / self.density_width))
        max_x = min(ceil((x_rel - self.min_x + r_abs) / self.density_width), 
                    self.density_cols)
        min_y = max(0, int((y_rel - self.min_y - r_abs) / self.density_height))
        max_y = min(ceil((y_rel - self.min_y + r_abs) / self.density_height),
                    self.density_rows)

        # Find the center and radius of the circle in pixels
        # TODO: Consider re-evaluating these calculations for special cases.
        # What happens when the rectangle exceeds the state's bounding box and
        # is cut off by the min() and/or max() operations above?
        #
        # Center point (c_x, c_y): approximate center of the rectangle
        # Radius: the absolute radius, converted into pixels by dividing
        # by the average of `self.density_width` and `self.density_height`,
        # which dimensions of the area of the map covered by each pixel.
        # For a sufficiently high-resolution density map, these values should
        # be nearly identical, but we average them to try to avoid a system-
        # atic bias.
        # TODO: What is the mathematical basis for using arithmetic mean here,
        # if one even exists?
        c_x = int(round((min_x + max_x) / 2))
        c_y = int(round((min_y + max_y) / 2))
        density_dim = (self.density_width + self.density_height) / 2
        r = int(min(self.max_radius, r_abs) / density_dim)

        bounded = self.density_map[min_x:max_x, min_y:max_y]
        mask = np.zeros_like(bounded)
        mask_x_max, mask_y_max = mask.shape - 1

        # This implementation of the midpoint algorithm is heavily based on
        # Wikipedia's pseudocode. (See https://en.wikipedia.org/wiki/
        # Midpoint_circle_algorithm#C_example for the original code.)
        x = r - 1
        y = 0
        dx = 1
        dy = 1
        err = dx - (r << 1)
        while x > y:
            # The midpoint algorithm takes advantage of a circle's
            # symmetry and fills in eight pixels per iteration.
            to_mask = [
                (max(c_x - x - min_x, 0), max(c_y - y - min_y, 0)),
                (max(c_x - x - min_x, 0), min(c_y + y - min_y, mask_y_max)),
                (min(c_x + x - min_x, mask_x_max), max(c_y - y - min_y, 0)),
                (min(c_x + x - min_x, mask_x_max),
                 min(c_y + y - min_y, mask_y_max)),
                (max(c_x - y - min_x, 0), max(c_y - x - min_y, 0)),
                (max(c_x - y - min_x, 0), min(c_y + x - min_y, mask_y_max)),
                (min(c_x + y - min_x, mask_x_max), max(c_y - x - min_y, 0)),
                (min(c_x + y - min_x, mask_x_max),
                 min(c_y + x - min_y, mask_y_max))
            ]
            mask[to_mask] = 1
            if err <= 0:
                y += 1
                err += dy
                dy += 2
            if err > 0:
                x -= 1
                dx += 2
                err += dx - (r << 1)
            
        # The midpoint algorithm only fills in the boundary of the circle.
        # A scanline fill needs to be applied to fill in the mask.
        masked = 0
        for x in range(mask.shape[0]):
            ones = np.where(mask[x] == 1)[0]
            if len(ones) > 1:
                mask[x][ones[0]:ones[-1]+1] = 1
                masked += ones[-1] - ones[0] + 1
        
        # If no pixels are masked, we approximate population based on a pixel
        # close to the circle's center.
        if masked == 0:
            x = min(max(0, int((x - self.min_x) / self.density_width)),
                    self.density_cols)
            y = min(max(0, int((y - self.min_y) / self.density_height)),
                    self.density_rows)
            est_pop = self.density_map[x, y] * 4 * (r_abs ** 2)
        # Otherwise, we approximate population based on the mean density
        # of the mask.
        else:
            est_pop = np.mean(mask*bounded) * 4 * (r_abs ** 2)
        return min(est_pop, self.state_pop)

    @jit
    def abs_coords(self, x_rel: float, y_rel: float) -> Tuple[float]:
        """
        Converts relative coordinates (proportional distances along the
        axes of the rectangular bounding box) to absolute coordinates (defined
        by the map's coordinate system).
        :param x_rel: The relative x-position (typically from 0 to 1).
        :param y_rel: The relative y-position (typically from 0 to 1).

        Returns a 2-tuple with absolute coordinates `(x_abs, y_abs)`.
        """
        x_abs = (x_rel * (self.max_x - self.min_x)) + self.min_x
        y_abs = (y_rel * (self.max_y - self.min_y)) + self.min_y
        return (x_abs, y_abs)

    @jit
    def rel_coords(self, x_abs: float, y_abs: float) -> Tuple[float]:
        """
        Converts absolute coordinates (defined by the map's coordinate system)
        to relative coordinates (proportional distances along the axes of the
        rectangular bounding box).
        :param x_abs: The absolute x-coordinate.
        :param y_abs: The absolute y-coordiante.

        Returns a 2-tuple with relative coordinates `(x_rel, y_rel)`.
        """
        x_rel = (x_abs - self.min_x) / (self.max_x - self.min_x)
        y_rel = (y_abs - self.min_y) / (self.max_y - self.min_y)
        return (x_rel, y_rel)

    def reset(self):
        """ Resets the Bitmap object to its initial state. """
        # Create an empty canvas for the low-resolution district plan rendering 
        self.frame = np.zeros_like(self.frame)
        # Recreate r-tree (not preserved by pickling)
        self.rtree = self._rtree()

    def _density_map(self, resolution: int, pop_col: str) -> np.ndarray:
        """
        Renders a bitmap of the state's density.
        :param resolution: The approximate number of pixels in the bitmap.
            The number of pixels in the bitmap returned will be close to this
            value but may vary slightly to ensure the creation of a valid
            rectangle with integer-pixel side lengths.
        :param pop_col: The column containing total population in `self.df`.
        """
        area = np.zeros(len(self.df))
        for idx, geom in enumerate(self.df.geometry):   
            area[idx] = geom.area
        
        s_len = np.sqrt(resolution / self.alpha) # side length
        n_rows = int(np.ceil(s_len))
        n_cols = int(np.ceil(s_len * self.alpha))

        # Compute the density (people per unit area) for each VTD
        density = self.df[pop_col].as_matrix() / area
        # Calculate the local density for each pixel
        density_map = np.zeros((n_cols, n_rows))
        for col, row, bounds in self._bitmap_squares(n_cols, n_rows):
            for fid in list(self.rtree.intersection(bounds.bounds)):
                intersected = self.df.iloc[fid].geometry
                if intersected.intersects(bounds):
                    try:
                        inter_area = intersected.intersection(bounds).area
                    except TopologicalError:
                        # If the intersection fails, we use the .buffer(0)
                        # trick. This is a bit of a hack, as it is not
                        # guaranteed to fix the underlying issue and
                        # may distort the map (see https://github.com/mggg/
                        # GerryChain/pull/262), but the density map is only
                        # intended for rough approximations anyway.
                        buffered = intersected.buffer(0)
                        inter_area = buffered.intersection(bounds).area
                        warnings.warn("Could not compute density map "
                                        "for {}. Buffering.".format(fid))
                    density_map[col, row] = inter_area / bounds.area
                    density_map[col, row] *= density[fid]
        return density_map
                        
    def _district_map(self, resolution: int) \
    -> Collection[scipy.sparse.csr_matrix, int, int]:
        """
        Renders a district map, which maps VTD indices to regions in the
        map's rectangular bounding box for easy rendering of the districting
        plan.
        :param resolution: The approximate number of pixels in the bitmap.
            The number of pixels in the bitmap returned will be close to this
            value but may vary slightly to ensure the creation of a valid
            rectangle with integer-pixel side lengths.
        
        Returns: (district map, columns in bitmap, rows in bitmap)
        """
        s_len = np.sqrt(resolution / self.alpha)
        n_rows = int(np.ceil(s_len))
        n_cols = int(np.ceil(s_len * self.alpha))
        geo_weights = scipy.sparse.lil_matrix((len(self.df), n_rows * n_cols))

        for col, row, bounds in self._bitmap_squares(n_cols, n_rows):
            for fid in list(self.rtree.intersection(bounds.bounds)):
                intersected = self.df.iloc[fid].geometry
                if intersected.intersects(bounds):
                    try:
                        inter_area = intersected.intersection(bounds).area
                    except TopologicalError:
                        # If the intersection fails, we use the .buffer(0)
                        # trick. This is a bit of a hack, as it is not
                        # guaranteed to fix the underlying issue and
                        # may distort the map (see https://github.com/mggg/
                        # GerryChain/pull/262), but the district map is
                        # only intended for rough approximations anyway.
                        buffered = intersected.buffer(0)
                        inter_area = buffered.intersection(bounds).area
                        warnings.warn("Could not compute district map "
                                        "for {}. Buffering.".format(fid))
                    w_idx = (row * n_cols) + col
                    geo_weights[fid, w_idx] = inter_area / bounds.area
        return geo_weights.tocsr(), n_cols, n_rows
            
    def _bitmap_squares(self, n_cols: int, n_rows: int) -> Iterable:
        """
        A helper generator for the bitmap methods that divides the rectangular
        bounding box into a grid.
        :param n_cols: The number of columns to break the box into.
        :param n_rows: The number of rows to break the box into.

        Yields: `(col, row, bounds)`, where `bounds` is an approximately
        square grid element corresponding to `(col, row)` and represented
        as a Shapely polygon.
        """
        for row in range(n_rows):
            b_min_y = ((self.max_y - self.min_y) * row / n_rows) 
            b_max_y = ((self.max_y - self.min_y) * (row + 1) / n_rows)
            for col in range(n_cols):
                b_min_x = ((self.max_x - self.min_x) * col / n_cols)
                b_max_x = ((self.max_x - self.min_x) * (col + 1) / n_cols) 
                bounds = box(b_min_x + self.min_x, b_min_y + self.min_y,
                             b_max_x + self.min_x, b_max_y + self.min_y)
                yield col, row, bounds
            
    def _rtree(self) -> rtree.Rtree:
        """ Creates an rtree for fast queries against the VTDs' geometries. """
        tr = rtree.index.Index()
        for idx, row in self.df.iterrows():
            tr.insert(idx, row.geometry.bounds)
        return tr    