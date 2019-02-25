""" Trimming algorithms used by district allocation strategies. """
from typing import List, Optional
import numpy as np
from elbridge.graph import Graph


def trim_farthest_city(graph: Graph, vtds: List[int],
                       x_abs: float, y_abs: float) -> Optional[str]:
    """
    A trimming algoruthm that attempts to remove the city of the farthest
    VTD from the current location while maintaining contiguity.

    :param graph: the ``elbridge.Graph`` to load VTD/city metadata from.
    :param vtds: the list of VTD indices in the porposed allocation.
    :param x_abs: the x-coordinate of the point to calculate distances
        relative to.
    :param y_abs: the y-coordinate of the point to calculate distances
        relative to.

    If a city can be removed, the string identifier of the city is returned.
    Otherwise, ``None`` is returned.
    """
    test_vtds = graph.border_vtds(vtds)
    tested = set([])
    tested_cities = set([])
    farthest_vtds = _farthest_vtds(graph, test_vtds, x_abs, y_abs)
    city_count = len(set([graph.indices['vtd_to_city'][vtd] for vtd in vtds]))

    while len(tested_cities) < city_count and len(tested) < len(vtds):
        farthest_vtd = farthest_vtds[0]
        farthest_city = graph.indices['vtd_city'][farthest_vtd]
        if farthest_city not in tested_cities:
            farthest_vtds = farthest_vtds[1:]
            tested = tested.union(vtds - test_vtds)
            if graph.contiguous(test_vtds):
                return farthest_city
        farthest_vtds = farthest_vtds[1:]


def trim_next_vtd(graph: Graph, vtds: List[int],
                  x_abs: float, y_abs: float) -> Optional[int]:
    """
    A trimming algoruthm that attempts to remove the farthest VTD
    from a given point while maintaining contiguity.

    :param graph: the ``elbridge.Graph`` to load VTD/city metadata from.
    :param vtds: the list of VTD indices in the porposed allocation.
    :param x_abs: the x-coordinate of the point to calculate distances
        relative to.
    :param y_abs: the y-coordinate of the point to calculate distances
        relative to.

    If a VTD can be removed, the index of the VTD is returned.
    Otherwise, ``None`` is returned.
    """
    test_vtds = graph.border_vtds(vtds)
    farthest_vtds = _farthest_vtds(graph, test_vtds, x_abs, y_abs)
    tested = set([])
    while len(tested) < len(vtds):
        farthest_vtd = farthest_vtds[0]
        test_vtds.remove(farthest_vtd)
        if graph.contiguous(test_vtds):
            return farthest_vtd
        else:
            test_vtds.append(farthest_vtd)
        farthest_vtds = farthest_vtds[1:]


def _farthest_vtds(graph: Graph, vtds: List[int],
                   x_abs: float, y_abs: float) -> np.array:
    distances = np.sqrt((graph.centroids[vtds][0] - x_abs) ** 2 +
                        (graph.centroids[vtds][1] - y_abs) ** 2)
    return np.flip(np.argsort(distances))

