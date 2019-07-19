""" Graph-related classes and functions for elbridge. """
# pylint: disable=no-name-in-module
from math import ceil
from copy import deepcopy as dc
from collections import defaultdict
from typing import List, Callable, Dict, Tuple, Set, DefaultDict, Optional
from mypy_extensions import TypedDict
from geopandas import GeoDataFrame  # type: ignore
from libpysal.weights import Rook, Queen, W  # type: ignore
from elbridge.cgraph import CGraph  # type: ignore


# typing for heterogenous dictionaries:
# https://mypy.readthedocs.io/en/latest/more_types.html#typeddict
Indices = TypedDict('Indices', {
    'vtd_to_city': Dict[int, str],
    'vtds_in_city': Dict[str, List[int]],
    'vtd_to_county': Dict[int, str],
    'vtds_in_county': Dict[str, List[int]],
    'vtd_pop': Dict[int, int]
})  # pylint:disable=invalid-name


class Graph:
    """
    A graph of voting districts (VTDs) intended to be partitioned into
    congressional districts, with equal population and contiguity constraints.
    """
    def __init__(self, gdf: GeoDataFrame, n_districts: int, contiguity: str,
                 pop_col: str, city_col: str, county_col: str,
                 pop_tolerance: float, fusions: List[Callable] = None):
        """
        :param gdf: The GeoDataFrame of VTDs to construct the graph from.
        :param n_districts: The number of districts to be allocated.
        :param contiguity: The contiguity criterion to construct the adjacency
            matrix from.
        :param pop_col: The column in the GeoDataFrame containing the VTDs'
            total populations.
        :param city_col: The column in the GeoDataFrame containing the VTDs'
            cities or towns.
        :param county_col: The column in the GeoDataFrame containing the VTDs'
            counties.
        :param pop_tolerance: The proportion by which the population of
            fully allocated districts can vary from the average congressional
            district population (state population / number of districts).
        :param fusions: The fusions (from ``elbridge.fusions``) to be applied
            to the graph.
        """
        self.df = gdf  # pylint: disable=C0103
        self.adj = _adj_matrix(gdf, contiguity)
        self.n_districts = n_districts
        self.pop_target = gdf[pop_col].sum() / n_districts
        self.pop_tolerance = pop_tolerance
        self.pops = gdf[pop_col].values
        self.fusion_graph = FusionGraph(gdf, self.adj, fusions or [])

        vtd_to_city, vtds_in_city = _vtd_indices(gdf, city_col)
        vtd_to_county, vtds_in_county = _vtd_indices(gdf, county_col)
        vtd_pop = {idx: row[pop_col] for idx, row in gdf.iterrows()}

        self.indices = {
            'vtd_to_city': vtd_to_city,
            'vtds_in_city': vtds_in_city,
            'vtd_to_county': vtd_to_county,
            'vtds_in_county': vtds_in_county,
            'vtd_pop': vtd_pop
        }  # type: Indices

        self._build_graph()

        self.states = [self._init_state()]
        self.state = dc(self.states[0])
        self.dist_state = self._next_dist_state()

    def update(self, vtds: List[int]) -> Dict[int, int]:
        """
        Attempts to assign a list of VTDs to the current congressional
        district. If the VTDs _can_ be assigned after applying fusions, they
        are assigned and ``True`` is returned. If the VTDs cannot be assigned
        due to equal-population constraints, ``False`` is returned. If the
        current congressional district's population exceeds the minimum
        allowable population, a new congressional district will be created in
        some cases.

        :param vtds: A list of VTD indices (an index of n corresponds to the
        nth row in the Graph's GeoDataFrame) to allocate.

        Returns a dictionary mapping newly allocated VTDs (if any) to their
        congressional district. (This format is intended for consumption by
        :class:`~elbridge.bitmap.Bitmap`).
        """
        print('[graph.py] update vtds:', vtds)
        vtds = self.fusion_graph.fuse(vtds)  # apply fusions first
        print('fused:', vtds)
        if not self.graph.contiguous(vtds):
            print('not contiguous')
            return {}  # bail early if the allocation isn't contiguous

        pop = sum(self.pops[idx] for idx in vtds)
        allocated = self.dist_state['pop']
        min_pop = self.dist_state['min_pop']
        max_pop = self.dist_state['max_pop']
        if allocated + pop >= min_pop:  # Close to the population bounds
            print('close to pop bounds')
            print('proposed pop:', allocated + pop)
            # Case 1: the current district's population is greater than its
            # minimum population, and allocating ``vtds`` to the district
            # will push its population beyond its maximum population.
            # To handle the overflow, we move to the next district.
            # Note that doing so theoretically gives up an opportunity
            # to get as close as possible to equal-population districts--
            # just because one allocation exceeds the maximum population bound
            # does not imply that there's not a smaller allocation that fits
            # within the bounds.  However, this greedy approach, where we
            # advance to the next  district at the first opportunity, seems to
            #  be acceptable in practice.
            if allocated >= min_pop and allocated + pop >= max_pop:
                # For a state with N congressional districts, districts 1..N-1
                # imply district N.
                if self.state['district'] == self.n_districts - 1:
                    self.states.append(self.state)
                    self.state = dc(self.state)
                    self.state['debt'] = 0  # TODO is this in the right place?
                    last_vtds = self.state['vtd_by_district'][0]
                    self.graph.allocate(last_vtds, True)
                    self.state['vtd_by_district'][self.n_districts] = last_vtds
                    self.state['vtd_by_district'][0] = []
                    self.state['done'] = True
                    return {vtd: self.n_districts for vtd in last_vtds}
                else:
                    debt_delta = self.pop_target - self.dist_state['pop']
                    self.state['debt'] += debt_delta
                    self.states.append(self.state)
                    self.state = dc(self.state)
                    self.state['district'] += 1
                    self.dist_state = self._next_dist_state()

            # Case 2: the congressional district's allocated population is
            # below the minimum acceptable population, but the allocation
            # is too big to fit within the maximum acceptable population.
            # Thus, we reject the update.
            elif allocated + pop > max_pop:
                return {}

        # To avoid unallocated, inaccessible holes of whitespace within
        # allocations, we check for enclosed whitespace. If it exists, we
        # add it to the allocation and recursively call update().
        enclosed_vtds = self.graph.validate(vtds)
        print('enclosed vtds:', enclosed_vtds)
        if enclosed_vtds:
            return self.update(vtds + enclosed_vtds)

        # If the population bounds are satisfied and there is no enclosed
        # whitespace, we accept the update.
        for idx in vtds:
            city = self.indices['vtd_to_city'][idx]
            county = self.indices['vtd_to_county'][idx]
            self.state['vtd_by_district'][0].remove(idx)
            self.state['unallocated_in_city'][city].remove(idx)
            self.state['unallocated_in_county'][county].remove(idx)
        self.graph.allocate(vtds, self.dist_state['pop'] == 0)
        self.state['vtd_by_district'][self.state['district']] += vtds
        self.dist_state['pop'] += pop
        return {vtd: self.state['district'] for vtd in vtds}

    def _build_graph(self):
        """ Constructs a CGraph object from the adjacency natrix. """
        self.graph = CGraph()
        adj = {k: list(neighbors.keys()) for k, neighbors in self.adj}
        self.graph.init(adj)

    def _init_state(self) -> Dict:
        """ Constructs the initial state dictionary of the graph. """
        vtd_by_district = [self.all_vtds] + ([[]] * self.n_districts)
        return {
            'done': False,
            'pop_debt': 0,
            'district': 1,
            'unallocated_in_city': dc(self.indices['vtds_in_city']),
            'unallocated_in_county': dc(self.indices['vtds_in_county']),
            'vtd_by_district': vtd_by_district
        }

    def _next_dist_state(self) -> Dict:
        """
        Initializes the state of a new congressional district.
        When a new congressional district is initialized, the rolling
        population debt of the last state is used to calculated the new
        state's population bounds.

        `Wesberry v. Sanders (1964)`_ holds:
            The constitutional requirement in Art. I, § 2,that Representatives
            be chosen "by the People of the several States" means that, as
            nearly as is practicable, one person's vote in a congressional
            election is to be worth as much as another's.

        `Reynolds v. Sims (1964)`_, which is commonly cited as
        the source of the "one person, one vote" doctrine, adds:
            The federal constitutional requirement that both houses of a state
            legislature must be apportioned on a population basis means that,
            as nearly as practicable, districts be of equal population,
            though mechanical exactness is not required. Somewhat more
            flexibility may be constitutionally permissible for state
            legislative apportionment than for congressional districting.

        In keeping with these rulings, the tolerances on congressional
        district populations are rather tight. Out of the eighteen districts
        in `Pennsylvania's 2018 congressional district plan`_, the lowest
        district population is 686,892, and the highest district population
        is 732,595, constituting a difference of 6.65%. A typical tolerance,
        which we define to be the percent deviation from the target district
        population, is 1%, allowing for a maximum population difference of ~2%
        between the least and most populous districts. As of this writing,
        GerryChain uses `a tolerance of 1% by default`_.

        District populations are not sequentially independent. For instance,
        consider a state with ten districts. If nine districts are allocated
        with populations 1% below the target, the leftover district must
        necessarily have a population 9% above the target.
        To resolve this, we employ a notion of "population debt": if a
        congressional district's population is above the target, we add the
        surplus population (a negative debt) to a running debt, and if a
        district's population is below the target, we add the deficit
        (a positive debt) to that running debt. We then use this debt to
        bound the population of the next congressional district.

        Concretely, consider a state with districts with a tolerance of 2% and
        a population target of 100 people per district. For the first district,
        the population can range from 98 to 102. If we choose the lower bound,
        then we've incurred a debt of 2. If we add that debt to both sides of
        the normal population bounds, we have a range of 100 to 104 for the
        next congressional district. However, we are still bounded by the
        tolerance--the population debt should always make the constraints
        tighter, not looser--so the true range for the second congressional
        district is 100 to 102. This process continues throughout allocation.


        .. _Wesberry v. Sanders (1964): https://www.oyez.org/cases/1963/22
        .. _Reynolds v. Sims (1964): https://www.oyez.org/cases/1963/23
        .. _Pennsylvania's 2018 congressional district plan:
            https://ballotpedia.org/Redistricting_in_Pennsylvania#Demographics
        .. _a tolerance of 1% by default: https://github.com/mggg/GerryChain/
            blob/52a606c88f1c2648d1247c658b1992783ec79de7/gerrychain/
            constraints/validity.py#L75
        """
        lower_tol = self.pop_target * (1 - self.pop_tolerance)
        upper_tol = self.pop_target * (1 + self.pop_tolerance)
        # TODO is this safe?
        debt = self.states[-1]['pop_debt']

        return {
            'pop': 0,
            # If the debt is positive--that is, not enough people have been
            # allocated with respect to the target--the lower bound will
            # increase. For a debt ≥0, the lower bound will not change.
            # The lower bound is converted to its integer floor.
            'min_pop': int(max(lower_tol, lower_tol + debt)),
            # Likewise, if the debt is negative--that is, too many people
            # have been allocated--the upper bound will decrease.
            # The upper bound is converted to its integer ceiling.
            'max_pop': int(ceil(min(upper_tol, upper_tol + debt)))
        }

    def reset(self, district_idx: int = 0):
        """
        Rolls the graph back to the state where ``district_idx`` has been
        completely allocated, but no more districts have been allocated.
        For instance, ``.reset(1)`` rolls the graph back to the state where
        only the first congressional district has been allocated, and
        ``.reset(0)`` rolls the graph back to a state where no districts
        have been allocated.

        :param district_idx: The congressional district to reset to.
        """
        if len(self.states) < district_idx + 1:
            raise ResetError('Cannot reset to district {}. The district '
                             'does not yet exist.'.format(district_idx))
        self.state = self.states[district_idx]
        self.states = self.states[:district_idx + 1]
        self.dist_state = self._next_dist_state()
        self.graph.reset(district_idx)

    def to_dict(self) -> Dict[int, int]:
        """
        Returns a dictionary mapping VTD indices to congressional district
        assignments.
        """
        transformed = {}
        for district_idx, vtds in enumerate(self.state['vtd_by_district']):
            for vtd in vtds:
                transformed[vtd] = district_idx
        return transformed

    def unallocated_in_city(self, city: str) -> List[int]:
        """
        Returns a list of VTD indices corresponding to the unallocated VTDs
        in the specified city.

        :param city: The name of the city.
        """
        return dc(self.state['unallocated_in_city'][city])

    def unallocated_in_county(self, county: str) -> List[int]:
        """
        Returns a list of VTD indices corresponding to the unallocated VTDs
        in the specified county.

        :param city: The name of the county.
        """
        return dc(self.state['unallocated_in_county'][county])

    def border_vtds(self, vtds: List[int]) -> List[int]:
        """
        Given a list of VTD indices, returns the indices of the VTDs
        that are on the border of the allocation—that is, the VTDs that
        will border unallocated VTDs after allocation.

        :param vtds: The indices of the VTDs defining the border.
        """
        print("[graph.py] calling border_vtds...")
        return self.graph.border_vtds(vtds)

    def contiguous(self, vtds: List[int]) -> bool:
        """
        Determines if the geography described by ``vtds`` is contiguous
        with itself and not allocated to a congressional district,
        and that one of the following conditions is met:
        * The geography is contiguous with the current congressional district
          (if the current congressional district is non-empty).
        * The geography is contiguous with the previous congressional district
          (if the current congressional district is empty).
        * No congressional districts have been allocated yet.

        Returns ``True`` if the conditions for contiguity are met; otherwise,
        returns ``False``. No distinction is made between a violation of the
        first condition (the geometry described by ``vtds`` is contiguous with
        itself) and the second condition (the geometry described by ``vtds`` is
        contiguous with the districting plan).

        :param vtds: the list of VTD indices representing the geometry.
        """
        return self.graph.contiguous(vtds)  # wraps C++

    @property
    def done(self) -> bool:
        """
        True if all voting districts have been allocated
        to congressional districts.
        """
        return self.state['done']

    @property
    def current_district(self) -> int:
        """ The current congressional district (1-indexed). """
        return self.state['district']

    @property
    def current_district_pop(self) -> int:
        """ Returns the population of the current congressional district. """
        return self.dist_state['pop']

    @property
    def current_vtds(self) -> List[int]:
        """
        Returns the indices of all VTDs allocated to the current district.
        """
        return dc(self.state['vtd_by_district'][self.state['district']])

    @property
    def all_vtds(self) -> List[int]:
        """
        Returns the indices of all VTDs in the graph.
        """
        return list(self.df.index)

    @property
    def vtds_left(self) -> List[int]:
        """
        Returns the indices of all unallocated VTDs in the graph.
        """
        return dc(self.state['vtd_by_district'][0])

    @property
    def current_district_pop_bounds(self) -> Tuple[int, int]:
        """
        Returns the minimum and maximum population of the current
        district when fully allocated.
        """
        return self.dist_state['min_pop'], self.dist_state['max_pop']

    @property
    def min_pop_left(self) -> int:
        """
        The minimum population to be allocated to the current congressional
        district to satisfy the equal population constraint.
        """
        return self.dist_state['min_pop'] - self.dist_state['pop']

    @property
    def max_pop_left(self) -> int:
        """
        The maximum population to be allocated to the current congressional
        district to satisfy the equal population constraint.
        """
        return self.dist_state['max_pop'] - self.dist_state['pop']

    def __repr__(self) -> str:
        """ Returns a summary of the graph's node count and partitioning. """
        n_vtd = len(self.df)
        if self.state['done']:
            msg = ('A graph of {} voting districts (nodes) partitioned into {}'
                   ' congressional districts')
        else:
            msg = ('A graph of {} voting districts (nodes) partially'
                   ' partitioned into {} congressional districts')
        return msg.format(n_vtd, self.n_districts)


def remove_city(graph: Graph, vtds: List[int], city: str) -> List[int]:
    """
    Returns a copy of ``vtds`` with all VTDs within ``city`` removed.

    :param graph: the graph of VTDs (used for city<->VTD indices).
    :param vtds: the list of VTDs to remove ``city`` from.
    :param city: the identifier of the city to remove.
    """
    city_vtds = graph.indices['vtds_in_city'][city]
    return [vtd for vtd in vtds if vtd not in city_vtds]


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


def _vtd_indices(gdf: GeoDataFrame, div_col: str) \
       -> Tuple[Dict[int, str], Dict[str, List[int]]]:
    """
    Generates indices that map between VTDs and administrative divisions.
    Two indices are returned: a ``vtd_to_div`` dictionary, which maps the row
    indices of the GeoDataFrame (each corresponding to a VTD) to the
    administrative divisions corresponding to those VTDs; and a ``vtds_in_div``
    dictionary, which maps each administrative division to a list of
    row indices.

    :param gdf: The GeoDataFrame to generate the index from.
    :param div_col: The column in the GeoDataFrame that lists each VTD's
        administrative division.
    """
    vtd_to_div: Dict[int, str] = {}
    vtds_in_div: Dict[str, List[int]] = defaultdict(list)
    for vtd_idx, vtd in gdf.iterrows():
        vtd_to_div[vtd_idx] = vtd[div_col]
        vtds_in_div[vtd[div_col]].append(vtd_idx)
    return vtd_to_div, vtds_in_div


class FusionGraph:
    def __init__(self, gdf: GeoDataFrame, adj: W, fusions: List[Callable]):
        """
        Builds a fusion graph of VTD indices from a GeoDataFrame and a list
        of fusions.

        :param gdf: The GeoDataFrame of VTDs to generate the graph from.
        :param adj: An adjacency matrix generated from the GeoDataFrame.
        :param fusions: The list  of fusions.
        """
        self.graph: DefaultDict[int, Set[int]] = defaultdict(set)
        for fusion in fusions:
            fused = fusion(gdf, adj)
            for f1, f2 in fused.items():
                self.graph[f1].add(f2)
                self.graph[f2].add(f1)

    def fuse(self, vtds: List[int]) -> List[int]:
        """
        Given a list of VTD indices, returns a list of VTD indices
        with fusions applied. It is guaranteed that ``len(fused)`` ≥
        ``len(vtds)``.

        :param vtds: The list of VTD indices to apply fusions to.
        """
        fused: Set[int] = set([])
        for vtd in vtds:
            fused.add(vtd)
            if self.graph[vtd]:
                fused = fused.union(self._get_children(vtd))
        return list(fused)

    def _get_children(self, vtd_idx: int,
                      discovered: Optional[Set[int]] = None) -> Set[int]:
        """
        Recursively generates the set of VTDs that a VTD is fused with
        given that VTD's index.

        :param vtd_idx: The VTD's index.
        :param discovered: The set of children already disovered (used
            internally).
        """
        children: Set[int] = set([])
        if discovered:
            children = discovered
        for child in self.graph[vtd_idx]:
            if child not in children:
                children.add(child)
                children = children.union(self._get_children(child, children))
        return children


class ResetError(Exception):
    """ Wrapper class for errors related to resetting a Graph's state. """


class ContiguityError(Exception):
    """
    Raised when an invalid contiguity criterion is passed
    to the Plan constructor.
    """
