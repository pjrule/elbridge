""" Congressional district allocation algorithms. """
from copy import copy
from random import random, choice
from typing import Callable
import numpy as np
from elbridge.plan import Plan
from elbridge.graph import remove_city
from elbridge.common import bound
from elbridge.trim import trim_farthest_city, trim_farthest_vtd


def stochastic_pop_coords(plan: Plan,  r_P: float, theta: float,
                          city_trimmer: Callable = trim_farthest_city,
                          vtd_trimmer: Callable = trim_farthest_vtd,
                          p_random: float = 0.1, vtd_idx: int = None) -> None:
    """
    Greedily allocates VTDs to the current congressional district.

    A position delta vector `〈Δx, Δy〉` is derived from coordinates given in
    the (r_P, θ) system ("people-based coordinates"), where r_P is a
    proportion of the overall state population (0-1) and θ is a direction
    (in radians).

    With probability 1 - `p_random`, we find the VTD that corresponds to the
    calculated position. We attempt to allocate the entire unallocated portion
    of the VTD's county to the current district.

    With probability `p_random`, we choose a random VTD contiguous to the
    border of the current congressional district. As in the non-random case,
    we attempt to allocate the entire unallocated portion of the VTD's county
    to the current district.

    Allocation requires:
        a. The remaining portion of the county is contiguous to the current
           district.
        b. The population of the remaining portion of the county fits
           within the population constraints of the current congressional
           district.

    If contiguity constraint is violated, allocation is aborted; if the
    population constraint is violated, we remove cities (or fractional
    cities, if an included city has already been partially allocated to
    a congressional district) from the proposed allocation. If this still
    fails to meet the population constraint, we remove individual VTDs
    until the population constraint is satisfied. We refer to algorithms that
    prioritize the removal of cities and VTDs from an allocation as _trimmers_.
    The farthest-distance trimming algorithm, which eliminates cities and VTDs
    based on the farthest VTDs from the point derived from `〈Δx, Δy〉`,
    is used by default.

    :param plan: . 
    :param r_P: .
    :param theta: .
    :param city_trimmer: .
    :param vtd_trimmer: .
    :param p_random: .


    Returns: ``None`` (``plan`` is modified in place).
    """
    if random() < p_random:
        # With probability `p_random`, go to a random VTD contiguous
        # with the border of the current district.
        # TESTME does this properly handle the case where the current district
        # has no VTDs? (Is that possible, anyway?)
        border_vtds = plan.graph.graph.border_vtds(plan.graph.current_vtds())
        stochastic_pop_coords(plan, r_P, theta, p_random, choice(border_vtds))

    if vtd_idx:
        to_x_abs, to_y_abs = plan.bitmap.centroids[vtd_idx]
    else:
        r_people_abs = bound(r_P, 0, 1) * plan.bitmap.pops.sum()
        r_geo = plan.bitmap.people_to_geo(plan.x, plan.y, r_people_abs)
        d_x = r_geo * np.cos(theta)
        d_y = r_geo * np.sin(theta)
        to_x_abs = bound(d_x + plan.x, plan.bitmap.min_x, plan.bitmap.max_x)
        to_y_abs = bound(d_y + plan.y, plan.bitmap.min_y, plan.bitmap.max_y)
        vtd_idx = plan.bitmap.vtd_at_point(to_x_abs, to_y_abs)

    county = plan.graph.indices['vtd_to_county'][vtd_idx]
    vtds_in_county = plan.graph.state['unallocated_in_county'][county]

    # STEP 1: attempt to allocate the entire county.
    # None of the VTD's county is not contiguous with the current congressional
    # district; thus, the VTD itself cannot possibly be contiguous with the
    # current congressional district. We reject the allocation.
    if not plan.graph.contiguous(vtds_in_county):
        return
    # The entire county can be added to the congressional district. Done!
    elif plan.update(vtds_in_county):
        return

    vtds = copy(vtds_in_county)

    # STEP 2: Remove cities from the county to fit population constraints.
    # If the update fails for the entire county, we remove cities using an
    # arbitrary trimmer until the population constraints are satisfied.
    proposed_city = city_trimmer(plan.graph, border_vtds)
    while not proposed_city:
        vtds = remove_city(plan.graph, vtds, proposed_city)
        if plan.update(vtds):
            return
        proposed_city = city_trimmer(plan.graph, border_vtds)

    # STEP 3: Remove individual VTDs from the county to fit population
    # constraints using an arbitrary trimmer. This only occurs if removing
    # whole cities fails to reduce the allocation's population enough
    # to satisfy equal population constraints.
    proposed_vtd = vtd_trimmer(plan.graph, border_vtds, to_x_abs, to_y_abs)
    while proposed_vtd:
        vtds.remove(proposed_vtd)
        if plan.update(vtds):
            return
        proposed_vtd = vtd_trimmer(plan.graph, border_vtds, to_x_abs, to_y_abs)

    # STEP 4: attempt to allocate a single VTD (last resort)
    if plan.graph.contiguous([vtd_idx]):
        plan.update([vtd_idx])



