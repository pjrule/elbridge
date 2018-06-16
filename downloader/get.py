"""
This script is intended to facilitate reproducibility.
It downloads and parses all the data used for redistricting from the exact sources used in all published work.

Downloading these files presumably constitutes agreement to Harvard Dataverse's terms of service.
License:
Harvard Election Data Archive by Stephen Ansolabehere and Jonathan Rodden is licensed under a Creative Commons Attribution 3.0 Unported License.
Government data (the files from Census.gov) is in the public domain.
"""

# Default directory names/filenames (per state)
DIRECTORIES = {
    "vtd_map": "vtd_map",
    "demographics": "pl94_171",
    "county_map": "county_map",
    "openelections": "openelections"
}
FILENAMES = {
    "demographics": "vtd_demographics.csv",
    "elections": "vtd_elections.csv",
    "county_names": "county_names.csv",
}

import wget # recommended: https://stackoverflow.com/questions/13137817/how-to-download-image-using-requests
import click
import os
import sys
import yaml
from zipfile import ZipFile
import convert
import numpy as np
import pandas as pd

@click.command()
@click.argument('state')
@click.argument('dir')
@click.option('--no-download', is_flag=True)
def download(state, dir, no_download):
    all_data = yaml.load(open('data.yml'))
    if state == "PA":
        sources = all_data['PA']['sources']
        filenames = all_data['PA']['filenames']

        if no_download:
            print("Skipping download...")
            os.chdir(dir)
        else:
            print('Downloading raw files...')
            new_dir_and_cd(dir)
            for s in ['vtd_map', 'demographics', 'county_map']:
                dump_zip(sources[s], DIRECTORIES[s])
            wget.download(sources['county_names'], FILENAMES['county_names'])
            os.mkdir('openelections')
            for year in sources['openelections']:
                wget.download(sources['openelections'][year], out=os.path.join('openelections', '%s_%d.csv' % (state, year)))
           
        filenames['openelections'] = {}    
        for year in range(2008, 2018):
            filenames['openelections'][year] = '%s_%d.csv' % (state, year)

        print('\nProcessing demographic data...')
        vtd_map = os.path.join(DIRECTORIES['vtd_map'], filenames['vtd_map'])
        demographics_02 = os.path.join(DIRECTORIES['demographics'], filenames['demographics_02'])
        demographics_geo = os.path.join(DIRECTORIES['demographics'], filenames['demographics_geo'])
        convert.dataverse_to_demographics(vtd_map, demographics_geo, demographics_02, FILENAMES['demographics'])

        print("Fusing OpenElections and Harvard Dataverse voting data...")
        """
        We want to fuse the OpenElections data with the Harvard Dataverse data while keeping the Harvard GeoID conventions.
        That way, the data will line up with the Dataverse geodata as much as possible.
        Sources:
            - Dataverse voting data from 2006-2010 (preferred when overlapping; 2002-2004 data appears incomplete and doesn't matter much anyway)
            - OpenElections voting data from 2008-2016
        """
        PA_STATE_PREFIX = '42'
        harvard_file = os.path.join(DIRECTORIES['vtd_map'], filenames['vtd_map'])  
        open_files = {}
        for year in [2008, 2010, 2014]:
            open_files[year] = os.path.join(DIRECTORIES['openelections'], filenames['openelections'][year])
        eight_conv, harvard_df, _ = convert.map_open_harvard(harvard_file, open_files[2008], FILENAMES['county_names'], "USCDV2008", "USCRV2008", "GEOID10", PA_STATE_PREFIX)
        ten_conv, _, _ = convert.map_open_harvard(harvard_file, open_files[2010], FILENAMES['county_names'], "USCDV2010", "USCRV2010", "GEOID10", PA_STATE_PREFIX)
        open_to_harvard_union = {**eight_conv, **ten_conv}

        vote_cols = {}
        for year in range(2006, 2018, 2):
            vote_cols['dv_%d' % year] = np.zeros(len(harvard_df))
            vote_cols['rv_%d' % year] = np.zeros(len(harvard_df))
        vote_cols['geo_id'] = []
        
        i = 0
        geo_id_to_idx = {}
        for row in harvard_df.itertuples():
            geo_id = getattr(row, "GEOID10")
            geo_id_to_idx[geo_id] = i
            vote_cols['geo_id'].append(geo_id)
            for year in range(2006, 2012, 2):
                vote_cols['dv_%d' % year][i] = getattr(row, "USCDV%d" % year)
                vote_cols['rv_%d' % year][i] = getattr(row, "USCRV%d" % year)
            i += 1
    
        for year in range(2012, 2018, 2):
            not_found = 0
            open_df_file = os.path.join(DIRECTORIES['openelections'], filenames['openelections'][year])
            open_df = convert.load_openelections_data(open_df_file, FILENAMES['county_names'], 'dv', 'rv', 'geo_id', PA_STATE_PREFIX)
            for row in open_df.itertuples():
                try:
                    harvard_geo_id = open_to_harvard_union[getattr(row, 'geo_id')]
                    vote_cols['dv_%d' % year][geo_id_to_idx[harvard_geo_id]] = getattr(row, 'dv')
                    vote_cols['rv_%d' % year][geo_id_to_idx[harvard_geo_id]] = getattr(row, 'rv')
                except KeyError:
                    not_found += 1
            print("\tWarning: %d precincts in OpenElections data (%d) not found in Harvard Dataverse data." % (not_found, year))

        pd.DataFrame(vote_cols).to_csv(FILENAMES['elections'])  

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
    wget.download(url, out='data.zip')
    ZipFile('data.zip').extractall('.')
    os.remove('data.zip')
    os.chdir('..')

if __name__ == "__main__":
    download()
