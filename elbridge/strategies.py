""" Congressional district allocation algorithms. """
from copy import copy
from random import random, choice
from typing import Callable
import numpy as np  # type: ignore
from elbridge.plan import Plan
from elbridge.graph import remove_city
from elbridge.common import bound
from elbridge.trim import trim_farthest_city, trim_farthest_vtd
import matplotlib.pyplot as plt

def stochastic_pop_coords(plan: Plan, r_P: float, theta: float,
                          city_trimmer: Callable = trim_farthest_city,
                          vtd_trimmer: Callable = trim_farthest_vtd,
                          p_random: float = 0.1, vtd_idx: int = None) -> None:
    """
    Greedily allocates voting districts to congressional districts using
    population coordinates.

    Elbridge is designed for use with reinforcement learning models that walk
    around on the redistricting plan; we've discovered empirically that these
    models often perform poorly when they encounter large population gradients
    if their output is expressed in terms of geographical deltas. For instance,
    walking ten miles in a city might mean moving through dozens of voting
    districts; in a rural area, it may be possible to walk ten miles and remain
    within the same voting district.  To address these issues, we specify
    coordinates in terms of population. ``r_P``specifies the radius of a
    geographical circle **in terms of people**. A position delta vector
    :math:`\\langle \\Delta x, \\Delta y \\rangle` is derived from coordinates
    given in the (:math:`(r_P, \\theta)`) system (population coordinates),
    where :math:`r_P` is a proportion of the overall state population (0-1) and
    :math:`\\theta` is a direction (in radians).

    With probability 1 - ``p_random``, we find the VTD that corresponds to the
    calculated position. We attempt to allocate the entire unallocated portion
    of the VTD's county to the current district.

    With probability ``p_random``, we choose a random VTD contiguous to the
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
    based on the farthest VTDs from the point derived from :math`\\langle
    \\Delta x, \\Delta y \\rangle` is used by default.

    :param plan: the ``Plan`` to update.
    :param r_P: the radius to move along, specified as a proportion of the
        total state population. For instance, an ``r_P`` of 0.1 is the radius
        of the circle encompassing 10% of the total state population, centered
        at ``(plan.x, plan.y)``.
    :param theta: The direction to move in, specified in radians.
    :param city_trimmer: the trimming algorithm used to remove cities from
        a proposed allocation when the equal population constraint is violated.
    :param vtd_trimmer: the trimming algorithm used to remove individual VTDs
        from a proposed allocation when the equal population constraint is
        violated and removing cities (using ``city_trimmer``) fails to satisfy
        the constraint.
    :param p_random: The probability that :math:`(r_P, \\theta)` will be
        ignored in favor of a random allocation.

    Returns: ``None`` (``plan`` is modified in place).
    """
    if random() < p_random:
        # With probability `p_random`, go to a random VTD contiguous
        # with the border of the current district.
        # TODO: other than the initial case, does this properly handle the case
        # where the current district has no VTDs? (Is that possible, anyway?)
        if len(self.graph.current_vtds) == len(self.graph.all_vtds):
            random_vtds = self.graph.all_vtds
        else:

        stochastic_pop_coords(plan=plan,
                              r_P=r_P,
                              theta=theta,
                              city_trimmer=city_trimmer,
                              vtd_trimmer=vtd_trimmer,
                              p_random=p_random,
                              vtd_idx=choice(border_vtds))

    if vtd_idx:
        to_x_abs, to_y_abs = plan.bitmap.centroids[vtd_idx]
    else:
        x_abs, y_abs = plan.bitmap.abs_coords(plan.x, plan.y)
        r_people_abs = bound(r_P, 0, 1) * plan.bitmap.pops.sum()
        r_geo = plan.bitmap.people_to_geo(x_abs, y_abs, r_people_abs)
        d_x = r_geo * np.cos(theta)
        d_y = r_geo * np.sin(theta)
        to_x_abs = bound(d_x + x_abs, plan.bitmap.min_x, plan.bitmap.max_x)
        to_y_abs = bound(d_y + y_abs, plan.bitmap.min_y, plan.bitmap.max_y)
        to_x_rel, to_y_rel = plan.bitmap.rel_coords(to_x_abs, to_y_abs)
        vtd_idx = plan.bitmap.vtd_at_point(to_x_rel, to_y_rel)

    print(to_x_abs, to_y_abs, vtd_idx)
    # It is possible for `vtd_idx` to be None. This occurs when
    # agent wanders into the whitespace surrounding a state's geography
    # or into a geographical feature (such a large body of water) not
    # assigned to a VTD. In this case, allocation is impossible.
    # Likewise, if the VTD has already been allocated, we abort
    # allocation.
    if not vtd_idx or vtd_idx not in plan.graph.vtds_left:
        return

    county = plan.graph.indices['vtd_to_county'][vtd_idx]
    print('county:', county)
    vtds_in_county = plan.graph.state['unallocated_in_county'][county]
    border_vtds = plan.graph.border_vtds(vtds_in_county)
    print('vtds in county:', vtds_in_county)

    # Attempt to allocate the entire county.
    if not plan.graph.contiguous(vtds_in_county):
        # None of the VTD's county is contiguous with the current congressional
        # district; thus, the VTD itself cannot possibly be contiguous with the
        # current congressional district. We reject the allocation.
        alloc = np.zeros(len(plan.graph.df))
        for dist_idx, districts in enumerate(plan.graph.state['vtd_by_district']):
            alloc[districts] = dist_idx
        alloc[vtds_in_county] = len(plan.graph.state['vtd_by_district'])
        plan.graph.df['alloc'] = alloc
        plan.graph.df.plot(column='alloc')
        plt.show()
        print('not contiguous with county')
        return
    elif plan.update(vtds_in_county):
        # The entire county can be added to the congressional district. Done!
        return

    # Allocation has failed due to a violation of the equal population
    # constraint. We need to remove VTDs until the constraint is satisfied.
    vtds = copy(vtds_in_county)

    # Remove cities from the county to fit population constraints.
    # If the update fails for the entire county, we remove cities using an
    # arbitrary trimmer until the population constraints are satisfied.
    proposed_city = city_trimmer(plan, border_vtds, to_x_abs, to_y_abs)
    while not proposed_city:
        print('\tremoving cities...')
        new_vtds = remove_city(plan.graph, border_vtds, proposed_city)
        print('\t', set(vtds) - set(new_vtds))
        vtds = new_vtds
        if plan.update(vtds):
            return
        proposed_city = city_trimmer(plan, border_vtds, to_x_abs, to_y_abs)

    # Remove individual VTDs from the county to fit population
    # constraints using an arbitrary trimmer. This only occurs if removing
    # whole cities fails to reduce the allocation's population enough
    # to satisfy equal population constraints.
    proposed_vtd = vtd_trimmer(plan, border_vtds, to_x_abs, to_y_abs)
    while proposed_vtd:
        vtds.remove(proposed_vtd)
        if plan.update(vtds):
            return
        proposed_vtd = vtd_trimmer(plan, vtds, to_x_abs, to_y_abs)

    # Last resort: attempt to allocate a single VTD
    if plan.graph.contiguous([vtd_idx]):
        plan.update([vtd_idx])
