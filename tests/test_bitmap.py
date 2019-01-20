""" Unit tests for the elbridge.Bitmap class. """
import os
import pytest
import numpy as np
import geopandas as gpd
from math import pi
from elbridge import Bitmap
from elbridge.bitmap import _mask, _bound
from shapely.geometry import box

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
FIXTURE_FILES = {
    'pa129': os.path.join('pa129', 'pa129.shp')
}


@pytest.fixture(scope='module')
def grid_2x2():
    meta = {
        'width': 2,
        'height': 2,
        'alpha': 1,  # width / height
        'centroids': [(0.5, 0.5), (1.5, 0.5), (0.5, 1.5), (1.5, 1.5)],
        'mean_density': 1
    }
    pop = [1] * 4  # uniform population density
    polys = [
        box(0, 0, 1, 1), box(1, 0, 2, 1),
        box(0, 1, 1, 2), box(1, 1, 2, 2)
    ]
    gdf = gpd.GeoDataFrame({'pop': pop, 'geometry': polys})
    return Bitmap(gdf, 'pop', 1, 1), meta  # undersample bitmaps (1x1, 1x1)


@pytest.fixture(scope='module')
def grid_5x3():
    meta = {
        'width': 5,
        'height': 3,
        'alpha': 5 / 3,
        'centroids': [],
        'mean_density': 3
    }
    pop = list(range(1, 6)) * 3
    polys = []
    for row in range(3):
        for col in range(5):
            meta['centroids'].append((col + 0.5, row + 0.5))
            polys.append(box(col, row, col + 1, row + 1))
    gdf = gpd.GeoDataFrame({'pop': pop, 'geometry': polys})
    return Bitmap(gdf, 'pop', 60, 60), meta  # oversample bitmaps (10x6, 10x6)


@pytest.fixture(scope='module')
def pa129():
    pa = gpd.read_file(os.path.join(FIXTURES_DIR, FIXTURE_FILES['pa129']))
    meta = {
        'width': pa.bounds['maxx'].max() - pa.bounds['minx'].min(),
        'height': pa.bounds['maxy'].max() - pa.bounds['miny'].min()
    }
    meta['alpha'] = meta['width'] / meta['height']
    # TODO pick a reasonable district resolution (if needed)
    return Bitmap(pa, 'TOT_POP', 100, 1), meta


@pytest.fixture(params=['grid_2x2', 'grid_5x3', 'pa129'])
def all_maps(request, grid_2x2, grid_5x3, pa129):
    if request.param == 'grid_2x2':
        return grid_2x2
    elif request.param == 'grid_5x3':
        return grid_5x3
    elif request.param == 'pa129':
        return pa129


@pytest.fixture(params=['grid_2x2', 'grid_5x3'])
def grid_maps(request, grid_2x2, grid_5x3):
    if request.param == 'grid_2x2':
        return grid_2x2
    elif request.param == 'grid_5x3':
        return grid_5x3


def test_bounding_box(all_maps):
    bitmap, meta = all_maps
    assert bitmap.max_x - bitmap.min_x == meta['width']
    assert bitmap.max_y - bitmap.min_y == meta['height']


def test_alpha(all_maps):
    bitmap, meta = all_maps
    assert bitmap.alpha == meta['width'] / meta['height']


def test_centroids(grid_maps):
    bitmap, meta = grid_maps
    assert [tuple(p) for p in bitmap.centroids.tolist()] == meta['centroids']


def test_abs_coords(all_maps):
    bitmap, _ = all_maps
    mid_x = (bitmap.min_x + bitmap.max_x) / 2
    mid_y = (bitmap.min_y + bitmap.max_y) / 2
    assert np.allclose(bitmap.abs_coords(0, 0), (bitmap.min_x, bitmap.min_y))
    assert np.allclose(bitmap.abs_coords(1, 1), (bitmap.max_x, bitmap.max_y))
    assert np.allclose(bitmap.abs_coords(0.5, 0.5), (mid_x, mid_y))


def test_rel_coords(all_maps):
    bitmap, _ = all_maps
    mid_x = (bitmap.min_x + bitmap.max_x) / 2
    mid_y = (bitmap.min_y + bitmap.max_y) / 2
    assert np.allclose(bitmap.rel_coords(bitmap.min_x, bitmap.min_y), (0, 0))
    assert np.allclose(bitmap.rel_coords(bitmap.max_x, bitmap.max_y), (1, 1))
    assert np.allclose(bitmap.rel_coords(mid_x, mid_y), (0.5, 0.5))


def test_reset(grid_maps):
    bitmap, _ = grid_maps
    bitmap.frame = np.ones_like(bitmap.frame)
    bitmap.reset()
    assert np.array_equal(bitmap.frame, np.zeros_like(bitmap.frame))


def test_rtree_size(grid_maps):
    bitmap, _ = grid_maps
    assert len(bitmap.df) == bitmap.rtree.count(bitmap.rtree.bounds)


def test_update_districts_undersample(grid_2x2):
    bitmap, _ = grid_2x2
    frame = bitmap.update_districts({0: 1, 1: 2, 2: 3, 3: 4})
    assert frame[0][0] == 2.5  # mean district: 2.5


def test_update_districts_oversample(grid_5x3):
    bitmap, _ = grid_5x3
    assignments = {idx: idx for idx in range(15)}
    frame = bitmap.update_districts(assignments)
    for row in range(3):
        for col in range(5):
            # Get the 2x2 square of pixels corresponding to the district
            frame_dist = frame[(2 * row):(2 * (row + 1)),
                               (2 * col):(2 * (col + 1))]
            # The first unique pixel value shoud be the district number
            assert np.unique(frame_dist)[0] == (5 * row) + col
            # Only one district number should exist within the 2x2 square
            assert np.unique(frame_dist).size == 1


def test_density_map_undersample(grid_2x2):
    bitmap, _ = grid_2x2
    assert bitmap.density_map[0][0] == 1
    assert bitmap.density_map.size == 1


def test_density_map_oversample(grid_5x3):
    bitmap, _ = grid_5x3
    assert bitmap.density_map.shape == (6, 10)
    map_t = bitmap.density_map.T
    for row in range(10):
        assert np.unique(map_t[row])[0] == (row // 2) + 1
        assert np.unique(map_t[row]).size == 1


def test_bitmap_squares_undersample(grid_2x2):
    bitmap, _ = grid_2x2
    bitmap_squares = list(bitmap._bitmap_squares(1, 1))
    assert len(bitmap_squares) == 1
    assert bitmap_squares[0][0:2] == (0, 0)
    assert bitmap_squares[0][2].area == 4


def test_people_to_geo_relative(pa129):
    bitmap, _ = pa129
    left_geo_radius = bitmap.people_to_geo(0.33, 0.33, 10000)  # upper left
    right_geo_radius = bitmap.people_to_geo(0.66, 0.66, 10000)  # lower right

    # Radii should be non-zero
    assert left_geo_radius > 0
    assert right_geo_radius > 0

    # Density decreases from the upper left to the lower right.
    # Thus, for the same population, we expect the geographical radius to
    # *increase* from upper left to lower right--the population is packed in
    # more tightly in the upper left.
    assert right_geo_radius > left_geo_radius


# center coordinates chosen randomly
@pytest.mark.parametrize('center', [(0.289, 0.951), (0.981, 0.476),
                                    (0.864, 0.710), (0.549, 0.589)])
@pytest.mark.parametrize('radius', [1000, 3162, 10000, 31622, 100000])
def test_est_local_pop_vs_true_local_pop(pa129, center, radius):
    bitmap, _ = pa129
    est_pop = bitmap.local_pop(*center, radius)
    true_pop = bitmap._true_local_pop(*center, radius)
    # The estimated population and the true population should be within
    # 20% of an order of magnitude of each other--that is,
    # 10^-0.2 < est_pop / true_pop < 10^0.2. (10^0.2 is approximately 1.6.)
    assert abs(np.log10(est_pop / true_pop)) < 0.2


def test_true_local_pop(grid_2x2):
    bitmap, _ = grid_2x2
    pop = bitmap._true_local_pop(0.5, 0.5, 1)  # unit circle (centered)
    assert abs(pop - pi) < 0.01


@pytest.mark.parametrize('radius', [10, 20, 30])
def test_mask(radius):
    mask, n_masked = _mask(0, 60, 0, 60, 30, 30, radius)
    # Verify that the center of the circle is masked
    assert mask[30][30] == 1
    # Verify that n_masked matches the number of masked pixels
    assert mask[mask == 1].size == n_masked
    # Verify that the ratio between area of the rasterized
    # circle and the area of the ideal circle is 1±0.1
    assert abs((n_masked / (radius * radius * pi)) - 1) < 0.1


def test_mask_zero():
    mask, n_masked = _mask(0, 60, 0, 60, 30, 30, 0)
    assert mask[30][30] == 0
    assert mask[mask == 1].size == 0
    assert n_masked == 0


def test_bound():
    assert _bound(-1, 0, 5) == 0
    assert _bound(2, 0, 5) == 2
    assert _bound(6, 0, 5) == 5
