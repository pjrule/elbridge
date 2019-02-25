import geopandas as gpd
from typing import Tuple, List, Callable
from random import random
from elbridge.bitmap import Bitmap
from elbridge.graph import Graph
from elbridge.fusions import fuse_islands, fuse_enclosed


class Plan:
    """
    Represents a districting plan, which maps voting districts (VTDs) to
    congressional districts. For elbridge's purposes, a districting plan
    contains a graph of VTDs and several continuously updated bitmaps
    that represent the plan's geometry and population density.
    """
    def __init__(self, gdf: gpd.GeoDataFramee, n_districts: int,
                 density_resolution: int, district_resolution: int,
                 pop_col: str = 'total_pop', proj: str = 'esri:102003',
                 contiguity: str = 'rook', pop_tolerance: float = 0.01,
                 init: Callable = random_init):
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
        :param init: A function for initializat
        """
        if proj:
            gdf = gdf.to_crs({'init': proj})

        self.bitmap = Bitmap(gdf, density_resolution, district_resolution)
        self.graph = Graph(gdf, contiguity, pop_col, n_districts,
                           pop_tolerance, (fuse_islands, fuse_enclosed))
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
            while not self.plan.vtd_at_point(*state):
                state = self.init_func()
            self.x, self.y = state

        def reset(self) -> None:
            """
            Resets the plan and its ``Graph`` and ``Bitmap`` to their
            initial states. Useful for unpickling.
            """
            self._init_geo()
            self.graph.reset(0)
            self.bitmap.reset()

        def reset_district(self, district_idx: int) -> None:
            """
            Resets
            """
            pass

        def update(self, vtds: List[int]) -> bool:
            """
            Attempts
            """
            pass


def random_init() -> Tuple[float, float]:
    """
    Returns uniformly sampled random relative coordinates
    for plan initialization.
    """
    return random(), random()
