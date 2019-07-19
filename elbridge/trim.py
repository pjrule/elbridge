""" Trimming algorithms used by district allocation strategies. """
from typing import List, Optional, Set
import numpy as np  # type: ignore
from elbridge.plan import Plan


def trim_farthest_city(plan: Plan, vtds: List[int],
                       x_abs: float, y_abs: float) -> Optional[str]:
    """
    A trimming algorithm that attempts to remove the city of the farthest
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
    test_vtds = plan.graph.border_vtds(vtds)
    print('[trim_farthest_city] vtds:', vtds)
    print('[trim_farthest_city] border vtds:', test_vtds)
    tested: Set[int] = set([])
    tested_cities: Set[str] = set([])
    farthest_vtds = _farthest_vtds(plan, test_vtds, x_abs, y_abs)
    city_count = len(set([plan.graph.indices['vtd_to_city'][vtd]
                          for vtd in vtds]))

    while len(tested_cities) < city_count and len(tested) < len(vtds):
        farthest_vtd = farthest_vtds[0]
        farthest_city = plan.graph.indices['vtd_to_city'][farthest_vtd]
        if farthest_city not in tested_cities:
            farthest_vtds = farthest_vtds[1:]
            tested = tested.union(vtds, test_vtds)
            if plan.graph.contiguous(test_vtds):
                return farthest_city
        farthest_vtds = farthest_vtds[1:]
    return None


def trim_farthest_vtd(plan: Plan, vtds: List[int],
                  x_abs: float, y_abs: float) -> Optional[int]:
    """
    A trimming algorithm that attempts to remove the farthest VTD
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
    test_vtds = plan.graph.border_vtds(vtds)
    farthest_vtds = _farthest_vtds(plan, test_vtds, x_abs, y_abs)
    tested: Set[int] = set([])
    while len(tested) < len(vtds):
        farthest_vtd = farthest_vtds[0]
        test_vtds.remove(farthest_vtd)
        if plan.graph.contiguous(test_vtds):
            return farthest_vtd
        else:
            test_vtds.append(farthest_vtd)
        farthest_vtds = farthest_vtds[1:]
    return None


# TODO finish docstring
def _farthest_vtds(plan: Plan, vtds: List[int],
                   x_abs: float, y_abs: float) -> List[int]:
    """
    Sorts a list of VTD indices by Euclidean distance from point (``x_abs``,
    ``y_abs``), farthest to nearest.

    :param plan:
    :param vtds:
    :param x_abs:
    :param y_abs:

    Returns a list of 
    """
    print('distances from', vtds)
    print('centroids shape:', plan.bitmap.centroids.shape)
    distances = np.sqrt((plan.bitmap.centroids[vtds][0] - x_abs) ** 2 +
                        (plan.bitmap.centroids[vtds][1] - y_abs) ** 2)
    return list(np.flip(np.argsort(distances)))
