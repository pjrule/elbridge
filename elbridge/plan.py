import geopandas as gpd
from elbridge.bitmap  import Bitmap
from elbridge.graph   import Graph
from elbridge.fusions import fuse_islands, fuse_enclosed

class Plan:
    """
    Represents a districting plan, which maps voting districts (VTDs) to
    congressional districts. For elbridge's purposes, a districting plan
    contains a graph of VTDs and several continuously updated bitmaps
    that represent the plan's geometry and population density.
    """
    def __init__(self, shapefile: str, n_districts: int,
                 density_resolution: int, district_resolution: int,
                 pop_col: str = 'total_pop', proj: str = 'esri:102003',
                 contiguity: str = 'rook', pop_tolerance: float = 0.01):
        """
        :param shapefile: The path of the shapefile containing the plan's
            underlying geographical and population data.
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
        """
        if proj:
            gdf = gpd.read_file(shapefile).to_crs({'init': proj})
        else:
            gdf = gpd.read_file(shapefile)

        self.bitmap = Bitmap(gdf, density_resolution, district_resolution) 
        self.graph = Graph(gdf, contiguity, pop_col, n_districts,
                           pop_tolerance, (fuse_islands, fuse_enclosed))

        self.total_pop = gdf[pop_col].as_matrix()
        self.pop_tolerance = pop_tolerance
        self._reset_state()

    def _reset_state(self):
        self.state = {
            'debt': 0
        }