import pytest
from geopandas import GeoDataFrame
from shapely.geometry import box
from libpysal.weights import Rook
from elbridge.fusions import fuse_islands, fuse_enclosed


# filter PySAL's disconnected observations warning (expected)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_fuse_islands_one_island():
    # Create a GeoDataFrame and an adjacency matrix for a map with
    # a mainland (two adjacent VTDs) and an island (one free-standing VTD).
    polys = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(3, 3, 4, 4)]
    gdf = GeoDataFrame({'geometry': polys})
    adj = Rook.from_dataframe(gdf)

    # The island is closest to the rightmost mainland VTD, so we expect
    # a mapping from the island to that VTD.
    assert fuse_islands(gdf, adj) == {2: 1}


# filter PySAL's disconnected observations warning (expected)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_fuse_islands_two_islands():
    # Create a GeoDataFrame and an adjacency matrix for a map with
    # a mainland (two adjacent VTDs) and two islands (two free-standing VTDs).
    polys = [box(0, 0, 1, 1),     box(1, 0, 2, 1),
             box(3, 3, 3.5, 3.5), box(4, 4, 4.5, 4.5)]
    gdf = GeoDataFrame({'geometry': polys})
    adj = Rook.from_dataframe(gdf)

    # We expect mappings from both islands to the rightmost VTD.
    assert fuse_islands(gdf, adj) == {2: 1, 3: 1}


def test_fuse_enclosed_with_enclosed_vtds():
    # The inner VTD is completely enclosed within the outer VTD.
    # In this case, both VTDs are (loosely) squares, but the outer VTD
    # has an inner VTD-sized hole in the middle.
    inner = box(1, 1, 3, 3)
    outer = box(0, 0, 4, 4).difference(inner)
    gdf = GeoDataFrame({'geometry': [inner, outer]})
    adj = Rook.from_dataframe(gdf)

    # We expect a mapping from the inner VTD to the outer VTD.
    assert fuse_enclosed(gdf, adj) == {0: 1}


def test_fuse_enclosed_with_adjacent_vtds():
    # Construct two adjacent VTDs
    polys = [box(0, 0, 1, 1), box(1, 0, 2, 1)]
    gdf = GeoDataFrame({'geometry': polys})
    adj = Rook.from_dataframe(gdf)

    # We do not expect to find any enclosed VTDs.
    assert fuse_enclosed(gdf, adj) == {}
