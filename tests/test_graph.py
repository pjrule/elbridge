""" Unit tests for elbridge.graph. """
import pytest
import geopandas as gpd
from copy import deepcopy as dc
from shapely.geometry import box
from elbridge.graph import Graph, FusionGraph, ResetError, ContiguityError, \
                           remove_city, _adj_matrix, _vtd_indices
from elbridge.cgraph import CGraph


@pytest.fixture(scope='module')
def grid_gdf():
    """
    Returns a GeoDataFrame to build a Basic ``elbridge.Graph`` grid fixture:
        * 64 VTDs of equal population
        * 8 cities (one per row)
        * 4 counties (one every two rows)
   """
    pop = [1] * 64  # uniform population density
    polys = []
    cities = []
    counties = []
    for row in range(8):
        for col in range(8):
            polys.append(box(col, row, col + 1, row + 1))
            cities.append('city{}'.format(row))
            counties.append('county{}'.format(row // 2))
    return gpd.GeoDataFrame({'pop': pop, 'city': cities,
                             'county': counties, 'geometry': polys})


@pytest.fixture
def grid_8x8(grid_gdf):
    """
    Builds an ``elbridge.Graph`` grid fixture from ``grid_gdf``:
        * n_districts: 2
        * Tolerance of 6.25% (districts are 32 VTDs +/- 2 VTDs)
        * No fusions
    """
    return Graph(gdf=grid_gdf, n_districts=2, contiguity='rook', pop_col='pop',
                 city_col='city', county_col='county', pop_tolerance=2/32)


@pytest.fixture
def grid_8x8_one_fusion(grid_gdf):
    """
    Builds an ``elbridge.Graph`` grid fixture from ``grid_gdf``:
        * n_districts: 2
        * Tolerance of 6.25% (districts are 32 VTDs +/- 2 VTDs)
        * One fusion (forces VTD 0 and VTD 1 to be allocated together)
    """
    return Graph(gdf=grid_gdf, n_districts=2, contiguity='rook', pop_col='pop',
                 city_col='city', county_col='county', pop_tolerance=2/32,
                 fusions=[lambda gdf, adj: {0: 1}]), [0, 1]


@pytest.fixture
def grid_8x8_two_fusions(grid_gdf):
    """
    Builds an ``elbridge.Graph`` grid fixture from ``grid_gdf``:
        * n_districts: 2
        * Tolerance of 6.25% (districts are 32 VTDs +/- 2 VTDs)
        * Two fusions (force VTDs 0, 1, and 2 to be allocated together)
    """
    def f1_0(gdf, adj):
        """ VTD 1 and VTD 0 must be allocated together. """
        return {1: 0}

    def f1_2(gdf, adj):
        """ VTD 1 and VTD 2 must be allocated together. """
        return {1: 2}

    return Graph(gdf=grid_gdf, n_districts=2, contiguity='rook', pop_col='pop',
                 city_col='city', county_col='county', pop_tolerance=2/32,
                 fusions=[f1_0, f1_2]), [0, 1, 2]


@pytest.fixture
def grid_8x8_allocated(grid_8x8):
    """ Returns the 8x8 grid fixture with one half allocated. """
    grid_8x8.update(list(range(32)))
    return grid_8x8


def test_build_graph(grid_8x8_allocated):
    # This test is essentially a smoke test to ensure that the CGraph
    # object is created; it does not verify the CGraph's nodes and edges.
    # TODO: write more graph tests if we decide to retain CGraph
    grid_8x8_allocated.graph = None
    grid_8x8_allocated._build_graph()
    assert type(grid_8x8_allocated.graph) == CGraph


def test_index_generation(grid_8x8):
    # This test verifies that indices are generated from the proper
    # columns using _vtd_indices(); it does not verify their contents.
    # see test_vtd_indices() for further validation of index generation.
    vtd_to_city, vtds_in_city = _vtd_indices(grid_8x8.df, 'city')
    vtd_to_county, vtds_in_county = _vtd_indices(grid_8x8.df, 'county')
    assert grid_8x8.indices['vtd_to_city'] == vtd_to_city
    assert grid_8x8.indices['vtds_in_city'] == vtds_in_city
    assert grid_8x8.indices['vtd_to_county'] == vtd_to_county
    assert grid_8x8.indices['vtds_in_county'] == vtds_in_county
    assert list(grid_8x8.indices['vtd_pop'].keys()) == list(range(64))
    assert list(grid_8x8.indices['vtd_pop'].values()) == [1] * 64


def test_init_state(grid_8x8):
    state = grid_8x8._init_state()
    exp_city = {'city{}'.format(idx):
                list(range(idx * 8, (idx + 1) * 8)) for idx in range(8)}
    exp_county = {'county{}'.format(idx):
                  list(range(idx * 16, (idx + 1) * 16)) for idx in range(4)}
    assert not state['done']
    assert state['pop_debt'] == 0
    assert state['district'] == 1
    assert state['unallocated_in_city'] == exp_city
    assert state['unallocated_in_county'] == exp_county
    assert state['vtd_by_district'] == [list(range(64)), [], []]


def test_next_dist_state_no_debt(grid_8x8):
    state = grid_8x8._next_dist_state()
    assert state['pop'] == 0
    assert state['min_pop'] == 30  # tolerance of 6.25%
    assert state['max_pop'] == 34


def test_next_dist_state_with_positive_debt(grid_8x8, monkeypatch):
    monkeypatch.setattr(grid_8x8, 'states', [{'pop_debt': 1}])
    state = grid_8x8._next_dist_state()
    assert state['pop'] == 0
    assert state['min_pop'] == 31
    assert state['max_pop'] == 34


def test_next_dist_state_with_negative_debt(grid_8x8, monkeypatch):
    monkeypatch.setattr(grid_8x8, 'states', [{'pop_debt': -1}])
    state = grid_8x8._next_dist_state()
    assert state['pop'] == 0
    assert state['min_pop'] == 30
    assert state['max_pop'] == 33


def test_reset(grid_8x8, grid_8x8_allocated):
    # Before resetting, update() should fail--VTDs 0-31 are
    # already allocated.
    assert not grid_8x8_allocated.update(list(range(32)))
    grid_8x8_allocated.reset(0)
    assert grid_8x8_allocated.state == grid_8x8.state
    # After resetting, update() should succeed.
    assert grid_8x8_allocated.update(list(range(32)))


def test_reset_invalid(grid_8x8):
    with pytest.raises(ResetError):
        grid_8x8.reset(1)


def test_update_non_contiguous(grid_8x8_allocated):
    assert not grid_8x8_allocated.update([41, 42])


def test_update_valid(grid_8x8):
    assert grid_8x8.update([0, 1, 2, 3])


def test_update_max_pop_exceeded(grid_8x8):
    assert not grid_8x8.update(list(range(35)))  # max: 34 VTDs


def test_update_last_district(grid_8x8_allocated):
    assert grid_8x8_allocated.update([33, 34])
    assert grid_8x8_allocated.done


def test_update_enclosed_valid(grid_8x8):
    #  0  *  *  *  4  *  *  *
    #  8  *  *  * 12  *  *  *
    # 16 17 18 19 20  *  *  *
    assert grid_8x8.update([0, 4, 8, 12, 16, 17, 18, 19, 20])


def test_update_enclosed_too_big(grid_8x8):
    assert grid_8x8.update(list(range(16)))
    assert not grid_8x8.update(list(range(32, 40)))


@pytest.mark.parametrize('fused', ['grid_8x8_one_fusion',
                                   'grid_8x8_two_fusions'])
def test_update_with_fusion(fused, request):
    # We verify that the same allocation occurs for all VTDs in the
    # list of fused VTDs. For instance, if VTD 0 and VTD 1 are always
    # allocated together, then grid.update([0]) should have the same
    # result as grid.update([1]).
    # getting fixture values: https://stackoverflow.com/a/46420704
    orig_grid, fused_vtds = request.getfixturevalue(fused)
    for vtd in fused_vtds:
        # make a fresh copy of the grid to avoid state issues
        grid = dc(orig_grid)
        grid.update([vtd])
        assert grid.current_vtds == fused_vtds


def test_to_dict(grid_8x8, monkeypatch):
    monkeypatch.setattr(grid_8x8, 'state', {'vtd_by_district':
                        [list(range(32)), list(range(32, 64))]})
    district_assignments = grid_8x8.to_dict()
    for vtd_idx, district in district_assignments.items():
        assert district == vtd_idx // 32


def test_unallocated_in_city(grid_8x8):
    assert grid_8x8.unallocated_in_city('city0') == list(range(0, 8))


def test_unallocated_in_city_deep_copy(grid_8x8):
    unallocated_city0 = grid_8x8.unallocated_in_city('city0')
    internal_state = grid_8x8.state['unallocated_in_city']['city0']
    assert internal_state is not unallocated_city0
    assert internal_state == unallocated_city0


def test_unallocated_in_county(grid_8x8):
    assert grid_8x8.unallocated_in_county('county0') == list(range(0, 16))


def test_unallocated_in_county_deep_copy(grid_8x8):
    unallocated_county0 = grid_8x8.unallocated_in_county('county0')
    internal_state = grid_8x8.state['unallocated_in_county']['county0']
    assert internal_state is not unallocated_county0
    assert internal_state == unallocated_county0


def test_all_vtds_unallocated_on_init(grid_8x8):
    assert grid_8x8.state['vtd_by_district'][0] == list(range(64))


def test_current_vtds_unallocated(grid_8x8):
    current_vtds = grid_8x8.current_vtds
    assert current_vtds == []  # district 1 is empty


def test_current_vtds_allocated(grid_8x8_allocated):
    current_vtds = grid_8x8_allocated.current_vtds
    assert current_vtds == list(range(32))


def test_current_vtds_deep_copy(grid_8x8):
    current_vtds = grid_8x8.current_vtds
    internal_state = grid_8x8.state['vtd_by_district'][1]
    assert current_vtds is not internal_state
    assert current_vtds == internal_state


def test_border_vtds_unallocated(grid_8x8):
    assert grid_8x8.border_vtds([9, 10, 16, 17]) == [9, 10, 16, 17]


def test_border_vtds_allocated(grid_8x8_allocated):
    assert grid_8x8_allocated.border_vtds([9, 10, 16, 17]) == []
    border = list(range(24, 32))
    assert grid_8x8_allocated.border_vtds(border) == border


def test_contiguous_unallocated_with_contiguous_region(grid_8x8):
    assert grid_8x8.contiguous([0, 1, 8, 9])


def test_contiguous_unallocated_with_non_contiguous_region(grid_8x8):
    assert not grid_8x8.contiguous([1, 3])


def test_contiguous_with_allocated_region(grid_8x8_allocated):
    assert grid_8x8_allocated.contiguous([32, 33, 40, 41])


def test_contiguous_with_contiguous_region(grid_8x8_allocated):
    assert grid_8x8_allocated.contiguous(list(range(32, 40)))


def test_contiguous_with_inner_non_contiguous_region(grid_8x8_allocated):
    assert not grid_8x8_allocated.contiguous([1, 3])


def test_contiguous_with_outer_non_contiguous_region(grid_8x8_allocated):
    assert not grid_8x8_allocated.contiguous([32, 34])


def test_contiguous_with_disconnected_region(grid_8x8_allocated):
    assert not grid_8x8_allocated.contiguous([62, 63])


def test_done(grid_8x8, monkeypatch):
    assert not grid_8x8.done
    monkeypatch.setattr(grid_8x8, 'state', {'done': True})
    assert grid_8x8.done


def test_current_district(grid_8x8, monkeypatch):
    assert grid_8x8.current_district == 1  # initialized to 1, not 0
    monkeypatch.setattr(grid_8x8, 'state', {'district': 0})
    assert grid_8x8.current_district == 0


def test_min_pop_left(grid_8x8):
    # In the grid's initial state, we expect a minimum population of 30
    # (tolerance of 6.25% on a total population of 32), and we expect
    # no VTDs (and therefore no population) to be allocated yet.
    assert grid_8x8.min_pop_left == 30


def test_max_pop_left(grid_8x8):
    # In the grid's initial state, we expect a maximum population of 34
    # (tolerance of 6.25% on a total population of 32), and we expect
    # no VTDs (and therefore no population) to be allocated yet.
    assert grid_8x8.max_pop_left == 34


def test_graph_as_str_not_done(grid_8x8):
    assert str(grid_8x8) == ('A graph of 64 voting districts (nodes) partially'
                             ' partitioned into 2 congressional districts')


def test_graph_as_str_done(grid_8x8, monkeypatch):
    monkeypatch.setattr(grid_8x8, 'state', {'done': True})
    assert str(grid_8x8) == ('A graph of 64 voting districts (nodes) '
                             'partitioned into 2 congressional districts')


def test_fusiongraph_with_one_fusion():
    # This test verifies that FusionGraph.fuse() returns a proper list based on
    # its fusion graph in the one-fusion case. Higher-level functionality
    # (initialization of the fusion graph, application of fusions to
    # allocations) is tested in test_update_with_fusion().
    def fusion(gdf, adj):
        return {0: 1, 2: 3}

    f = FusionGraph(None, None, [fusion])
    combinations = [[0, 2], [0, 3], [1, 2], [1, 3]]
    for combo in combinations:
        assert f.fuse(combo) == list(range(4))


def test_fusiongraph_with_two_fusions():
    # This test verifies that FusionGraph.fuse() returns a proper list based on
    # its fusion graph in the case of multiple fusions. Higher-level
    # functionality (initialization of the fusion graph, application of
    # fusions to allocations) is tested in test_update_with_fusion().
    def f1(gdf, adj):
        return {1: 0}

    def f2(gdf, adj):
        return {1: 2}

    f = FusionGraph(None, None, [f1, f2])
    for vtd in range(3):
        assert f.fuse([vtd]) == list(range(3))


def test_remove_city(grid_8x8):
    vtds = list(range(64))
    vtds_city_removed = remove_city(grid_8x8, vtds, 'city0')
    assert vtds_city_removed == list(range(8, 64))


def test_adj_matrix_rook(grid_8x8):
    adj = _adj_matrix(grid_8x8.df, 'rook')
    assert sorted(adj[0].keys()) == [1, 8]  # corner (2 neighbors)
    assert sorted(adj[1].keys()) == [0, 2, 9]  # top (3 neighbors)
    assert sorted(adj[12].keys()) == [4, 11, 13, 20]  # inner (4 neighbors)


def test_adj_matrix_queen(grid_8x8):
    adj = _adj_matrix(grid_8x8.df, 'queen')
    assert sorted(adj[0].keys()) == [1, 8, 9]  # corner (3 neighbors)
    assert sorted(adj[1].keys()) == [0, 2, 8, 9, 10]  # top (5 neighbors)
    # inner (8 neighbors)
    assert sorted(adj[12].keys()) == [3, 4, 5, 11, 13, 19, 20, 21]


def test_adj_matrix_invalid(grid_8x8):
    with pytest.raises(ContiguityError):
        _adj_matrix(grid_8x8.df, 'invalid')


def test_vtd_indices(grid_8x8):
    vtd_to_div, vtds_in_div = _vtd_indices(grid_8x8.df, 'city')
    for idx, city in vtd_to_div.items():
        assert grid_8x8.df['city'].iloc[idx] == city
    for city, vtds in vtds_in_div.items():
        assert sorted(grid_8x8.df.iloc[vtds].index) == vtds
