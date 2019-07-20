from typing import Tuple, List, Callable
from random import random
from geopandas import GeoDataFrame
from elbridge.bitmap import Bitmap
from elbridge.graph import Graph
from elbridge.fusions import fuse_islands, fuse_enclosed


def random_init() -> Tuple[float, float]:
    """
    Returns uniformly sampled random relative coordinates
    for plan initialization.
    """
    return random(), random()


class Plan:
    """
    Represents a districting plan, which maps voting districts (VTDs) to
    congressional districts. For Elbridge's purposes, a districting plan
    contains a graph of VTDs and several continuously updated bitmaps
    that represent the plan's geometry and population density.
    """
    def __init__(self,
                 gdf: GeoDataFrame,
                 n_districts: int,
                 pop_col: str,
                 county_col: str,
                 city_col: str,
                 density_resolution: int,
                 district_resolution: int,
                 proj: str = 'esri:102003',
                 contiguity: str = 'rook',
                 pop_tolerance: float = 0.01,
                 init: Callable = random_init,
                 fusions: List[Callable] = (fuse_islands, fuse_enclosed)):
        """
        :param gdf: The GeoDataFrame containing the plan's underlying
            geographical and population data.
        :param n_districts: The number of congressional districts to be
            allocated by the plan. This should correspond to the number of
            congressional districts allocated to the state based on the latest
            U.S. Census.
        :param density_resolution: The approximate resolution (in pixels) of
            the population density bitmap used for area-based population
            estimations.
        :param district_resolution: The approximate resolution (in pixels) of
            the bitmap used to render the districting plan (intended as input
            into a CNN-based model).
        :param pop_col: The column in the shapefile with the total population
            of each VTD.
        :param proj: The map projection used for the shapefile. Defaults to
            USA Contiguous Albers Equal Area Conic (ESRI:102003). If ``None``,
            the shapefile's original projection is retained.
        :param contiguity: The contiguity criterion used to generate the
            adjacency matrix of the VTDs. Options: ``rook`` (contiguous
            edges) and ``queen`` (contiguous edges and vertices).
        :param pop_tolerance: The tolerance on the equal population constraint.
            Defaults to 1%, which is roughly consistent with currently enacted
            districting plans.
        :param init: A function for initialization of the plan's
           location state (x-coordinate and y-coordinate).
        :param fusions: A list of fusions used to preprocess allocations. By
           default, the :meth:`~elbridge.fusions.fuse_islands` and
           :meth:`~elbridge.fusions.fuse_enclosed` fusions are used, which
           fuse islands (VTDs with no neighbors) to the nearest mainland VTDs
           by Euclidean distance and fuse completely enclosed VTDs with their
           outer VTDs, respectively.
        """
        if proj:
            gdf = gdf.to_crs({'init': proj})

        self.bitmap = Bitmap(gdf, pop_col, density_resolution,
                             district_resolution)
        self.graph = Graph(gdf, n_districts, contiguity, pop_col, city_col,
                           county_col, pop_tolerance, fusions)
        self.init_func = init
        self._init_geo()

    def _init_geo(self) -> None:
        """
        Chooses an initial geographical location to begin allocation
        from given an initialization function. If the initialization
        function does not return a valid point—for instance, if
        the point returned is within the geography's rectangular
        bounding box but outside the geography's true border, or
        the point returned is within a body of water—a new point
        is chosen using the same initialization function until a valid
        point is found.
        """
        state = self.init_func()
        while not self.bitmap.vtd_at_point(*state):
            state = self.init_func()
        self.x, self.y = state

    def reset(self) -> None:
        """
        Resets the plan and its ``Graph`` and ``Bitmap`` to their
        initial states. Useful for unpickling.
        """
        self.graph.reset(0)
        self.bitmap.reset()
        self._init_geo()

    def update(self, vtds: List[int]) -> bool:
        """
        Attempts to update the graph and the bitmap by allocating
        ``vtds``. If the allocation fails, returns ``False``; otherwise,
        returns ``True``.

        :param vtds: The list of VTD indices to allocate.
        """
        updated = self.graph.update(vtds)
        if updated:
            self.bitmap.update(updated)
            return len(updated) > 0

    def __repr__(self):
        """
        Returns a string representation of the districting plan's state
        with some vital statistics.
        """
        return ('Plan with {} VTDs ({} allocated) and {} districts'
                ' (district={}, pop={}, bounds={},'
                ' x={:.5f}, y={:.5f})').format(
                    len(self.graph.all_vtds),
                    len(self.graph.all_vtds) - len(self.graph.vtds_left),
                    self.graph.n_districts, self.graph.current_district,
                    self.graph.current_district_pop,
                    self.graph.current_district_pop_bounds, self.x, self.y)

    @property
    def frame(self):
        """
        Returns the current bitmap rendering of the districting plan.
        """
        return self.bitmap.frame

    @property
    def done(self):
        """
        Returns the boolean state of the districting plan's graph.
        If all VTDs have been allocated to congressional districts,
        allocation is done.
        """
        return self.graph.done
