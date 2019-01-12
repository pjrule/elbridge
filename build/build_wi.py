"""
This script loads transformed Wisconsin state data 
into a `states.Wisconsin` object with the proper bitmap
dimensions.
"""
import click
import states
import pickle

@click.command('build')
@click.argument('ward_map')
@click.argument('vtd_elections')
@click.argument('vtd_demographics')
@click.argument('geo_resolution',     type=int)
@click.argument('density_resolution', type=int)
@click.argument('out_pkl')
def load(ward_map, vtd_elections, vtd_demographics, geo_resolution,
         density_resolution, out_pkl):
    """
    Uses geographical, voting, and demographic data to construct
    a `states.Wisconsin` object and its associated bitmap renderings.

    Arguments:
    - ward_map: the path of the .shp file with 2017 Wisconsin wards.
    - vtd_elections: the CSV file with per-ward election results.
    - vtd_demographics: the CSV with per-ward demographic data.
    - geo_resolution: the resolution of rendered bitmap frames (intended
      for use in ML pipeline), specified in log2. For instance, a value
      of 7 will result in frame dimensions of 2**7x2**7, or 128x128.
    - density_resolution: the resolution of the density bitmap used in the
      people -> geo coordinate system, specified in approximate total pixels.
    - out_pkl: the name of the output .pkl file (contains
    the `states.Wisconsin` object).

    When the object is loaded, the reset() method must be run to initialize
    the graph and the R-tree.
    """
    wi = states.Wisconsin(ward_map)
    # size: alpha*(2**geo_resolution) x (2**geo_resolution),
    # where we expect alpha < 1
    scaled_res = wi.alpha * (2**geo_resolution)**2
    wi.reset()
    wi.load_geo_mapping(scaled_res)
    wi.load_density_mapping(density_resolution)
    wi.graph = None # can't pickle graph
    wi.rtree = None # can't pickle rtree
    with open(out_pkl, 'wb') as f:
      pickle.dump(wi, f)