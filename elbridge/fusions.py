"""
Functions for identifying VTDs that should always be allocated together.
"""
import numpy as np  # type: ignore
from typing import Dict
from geopandas import GeoDataFrame  # type: ignore
from libpysal.weights import W  # type: ignore


def fuse_islands(gdf: GeoDataFrame, adj: W) -> Dict[int, int]:
    """
    Generates a fusion mapping for islands.
    Islands are fused with the nearest inland VTD by centroid distance.

    :param gdf: the GeoDataFrame with the VTDs.
    :param adj: the VTDs' adjacency matrix.
    """
    closest = {}
    if len(adj.islands) > 0:
        # Find distances between island centroids and all VTD centroids.
        distances = np.zeros((len(adj.islands), len(gdf)))
        centroids = [gdf.iloc[idx].geometry.centroid.coords[0]
                     for idx in adj.islands]
        for idx, row in enumerate(gdf.geometry):
            r_x, r_y = row.centroid.coords[0]
            dist = [np.sqrt((x - r_x)**2 + (y - r_y)**2) for x, y in centroids]
            distances[:, idx] = dist

        # For each island, find the inland VTD with the closest nonzero
        # distance to the island.
        for island_idx, island in enumerate(adj.islands):
            closest_dist = np.max(distances[island_idx])
            closest_idx = 0
            for vtd_idx, vtd_dist in enumerate(list(distances[island_idx])):
                if (vtd_dist > 0 and vtd_dist < closest_dist and
                   vtd_idx not in adj.islands):
                    closest_dist = vtd_dist
                    closest_idx = vtd_idx
            closest[island] = closest_idx
    return closest


def fuse_enclosed(gdf: GeoDataFrame, adj: W) -> Dict[int, int]:
    """
    Generates a fusion mapping for VTDs that are completely enclosed within
    other VTDs.

    :param gdf: the GeoDataFrame with the VTDs.
    :param adj: the VTDs' adjacency matrix.
    """
    enclosed = {}
    for idx, neighbors in enumerate(adj):
        if len(neighbors[1]) == 1:
            first_neighbor = list(neighbors[1].keys())[0]
            inner = gdf.iloc[idx].geometry
            outer = gdf.iloc[first_neighbor].geometry
            if _is_enclosed(inner, outer):
                enclosed[idx] = first_neighbor
    return enclosed


def _is_enclosed(inner, outer, tol: float = 0.001) -> bool:
    """
    Determines whether one VTD is completely enclosed by another VTD given
    two VTDs' Shapely objects (Polygon, MultiPolygon, etc.) by comparing
    their perimeters with the perimeters of their union.

    If one VTD is completely enclosed, we expect the perimeter of the union to
    be _smaller_ than the perimeter of the outer VTD. This is because Shapely's
    notion of perimeter (technically, length) takes the hole within the outer
    VTD left by the inner VTD into account! This hole disappears in the union,
    leaving only the outer perimeter. In the case of enclosure, we expect the
    union's perimeter to decrease by exactly as much as the perimeter of the
    inner VTD due to this phenomenon. However, if the VTDs are overlapping or
    adjacent, this conservation does not apply.

    :param inner: a VTD with one neighbor (suspected to be enclosed).
    :param outer: the VTD that neighbors ``inner``.
    :param tol: the relative tolerance of the perimeter comparison
        with respect to the perimeter of the union. The default of 0.1%
        is based on an experiment with MGGG's Wisconsin shapefile.
    """
    union = inner.union(outer)
    length_diff = abs(union.length - (outer.length - inner.length))
    return length_diff < (tol * union.length)
