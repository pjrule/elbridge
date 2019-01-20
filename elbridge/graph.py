from geopandas import GeoDataFrame
from elbridge.cgraph import CGraph
from elbridge.errors import ContiguityError
from libpysal.weights import Rook, Queen, W

class Graph:
    def __init__(self, gdf: GeoDataFrame, contiguity: str):
        """
        :param gdf: The GeoDataFrame of VTDs to construct the graph from.
        :param contiguity: The contiguity criterion to construct the adjacency
            matrix from.
        """ 
        adj = _adj_matrix(gdf, contiguity)
        self.graph = CGraph(adj)

def _adj_matrix(gdf: GeoDataFrame, contiguity: str) -> W:
    """
    Calculates the adjacency matrix for the VTDs in the given
    GeoDataFrame.

    :param gdf: The GeoDataFrame.
    :param contiguity: The contiguity criterion used for VTDs.
    """
    if contiguity == 'rook':
        return Rook.from_dataframe(gdf) 
    elif contiguity == 'queen':
        return Queen.from_dataframe(gdf)
    else:
        raise ContiguityError("Invalid contiguity criterion. Valid options"
                                " are 'queen' and 'rook'.")
