"""
This script is intended to facilitate reproducibility.
It downloads and parses all the data used for redistricting from the exact sources used in all published work.

Downloading these files presumably constitutes agreement to Harvard Dataverse's terms of service.
License:
Harvard Election Data Archive by Stephen Ansolabehere and Jonathan Rodden is licensed under a Creative Commons Attribution 3.0 Unported License.
Government data (the files from Census.gov) is in the public domain.
"""

# Default directory names/filenames (per state)
# These are 1:1 mappings by default but are defined globally for flexibility's sake.
DIRECTORIES = {
    "vtd_map": "vtd_map",
    "demographics": "pl94_171",
    "county_map": "county_map",
    "openelections": "openelections",
    "ward_maps": "ward_maps",
    "md_gov": "md_gov"
}
FILENAMES = {
    "demographics": "vtd_demographics.csv",
    "elections": "vtd_elections.csv",
    "county_names": "county_names.csv",
}

import click
import os
import sys
import yaml
import convert
import numpy as np
import pandas as pd
import geopandas as gpd
import shapefile
import fuse
from zipfile import ZipFile
from shutil import rmtree, move
from urllib.request import urlretrieve

@click.command()
@click.argument('state')
@click.argument('dir')
@click.option('--no-download', is_flag=True)
@click.option('--keep-census', is_flag=True)
@click.option('--keep-openelections', is_flag=True)
def download(state, dir, no_download, keep_census, keep_openelections):
    all_data = yaml.load(open('data.yml'))
    sources = all_data[state]['sources']
    filenames = all_data[state]['filenames']

    if no_download:
        print("Skipping download...")
        os.chdir(dir)
    else:
        print('Downloading raw files...')
        new_dir_and_cd(dir)
        for s in ['vtd_map', 'demographics', 'ward_maps']:
            if s in sources:
                if type(sources[s]) == str:
                    dump_zip(sources[s], DIRECTORIES[s])
                else:
                    os.mkdir(s)
                    for year in sources[s]:
                        dump_zip(sources[s][year], os.path.join(s, str(year)))
                        os.chdir('..')

        if 'county_names' in sources:
            dl(sources['county_names'], FILENAMES['county_names'])

        if 'openelections' in sources:
            os.mkdir('openelections')
            for year in sources['openelections']:
                dl(sources['openelections'][year], os.path.join('openelections', '%s_%d.csv' % (state, year)))

        # Due to the way Harvard Dataverse's download API works, some ZIP files containing shapefiles are stored within parent zipfiles.
        # Thus, it is necessary to unzip not one but *two* ZIPs.
        if 'vtd_map_inner_zip' in filenames:
            os.chdir(DIRECTORIES['vtd_map'])
            ZipFile(filenames['vtd_map_inner_zip']).extractall('.')
            if 'vtd_map_inner_zip_dir' in filenames:
                for f in os.listdir(filenames['vtd_map_inner_zip_dir']):
                    move(os.path.join(filenames['vtd_map_inner_zip_dir'], f), f)
                rmtree(filenames['vtd_map_inner_zip_dir'])
            os.remove(filenames['vtd_map_inner_zip'])
            os.chdir('..')

        # Some Harvard Dataverse shapefile packages lack .shx index files. 
        # These are necessary for GeoPandas to function; thankfully, they can be generated trivially.
        if 'vtd_map' in sources:
            shp_name = os.path.join(DIRECTORIES['vtd_map'], filenames['vtd_map'])
            if not os.path.isfile(shp_name.replace('.shp', '.shx')):
                print("\nGenerating geospatial index...")
                # lifted from http://geospatialpython.com/2011/11/generating-shapefile-shx-files.html
                # doesn't have a license, but I think I think fair use applies here—this is just loading files and rewriting them
                # thus, anyone with reasonable knowledge of the art of pyshp could probably have figured this out
                shp = open(shp_name, 'rb')
                dbf = open(shp_name.replace('.shp', '.dbf'), 'rb')
                r = shapefile.Reader(shp=shp, dbf=dbf, shx=None)
                w = shapefile.Writer(r.shapeType)
                w._shapes = r.shapes()
                w.records = r.records()
                w.fields = list(r.fields)
                w.save(shp_name.replace('.shp', ''))

        print('\nProcessing demographic data...')
        # For most states, demographic data is separate from geodata and needs to be fetched from the U.S. Census Bureau and aggregated.
        if 'demographics' in sources:
            vtd_map = os.path.join(DIRECTORIES['vtd_map'], filenames['vtd_map'])
            demographics_02 = os.path.join(DIRECTORIES['demographics'], filenames['demographics_02'])
            demographics_geo = os.path.join(DIRECTORIES['demographics'], filenames['demographics_geo'])
            convert.dataverse_to_demographics(vtd_map, demographics_geo, demographics_02, FILENAMES['demographics'])
            if not no_download and not keep_census:
                rmtree(DIRECTORIES['demographics'])
        # For Wisconsin, demographic data is included with the geodata.
        elif state == 'WI':
            convert.wi_to_demographics(os.path.join(DIRECTORIES['ward_maps'], '2020', filenames['ward_maps'][2020]), FILENAMES['demographics'])

    """ 
    Each state has distinct fusion/conversion steps.
    """
    if state == "MD":
        if not no_download:
            print("Downloading supplementary election data...")
            os.mkdir('md_gov')
            for year in sources['md_gov']:
                dl(sources['md_gov'][year], os.path.join('md_gov', '%s_%d.csv' % (state, year)))
        fuse.fuse_MD(filenames, sources)
    elif state == "WI":
        fuse.fuse_WI(filenames, sources)
    elif state == "PA":
        fuse.fuse_PA(filenames, sources)

    if not no_download and not keep_openelections and 'openelections' in sources:
        rmtree(DIRECTORIES['openelections'])

def new_dir_and_cd(dir):
    try:
        os.makedirs(dir)
        os.chdir(dir)
    except OSError:
        print("Could not create directory. Does it already exist?")
        sys.exit(1)

def dump_zip(url, dir):
    os.makedirs(dir)
    os.chdir(dir)
    dl(url, 'data.zip')
    ZipFile('data.zip').extractall('.')
    os.remove('data.zip')
    os.chdir('..')

def dl(url, f):
    print(url)
    urlretrieve(url, f)

if __name__ == "__main__":
    download()