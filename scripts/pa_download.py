"""
This script is intended to facilitate reproducibility.
It downloads and parses all the data used to redistrict Pennsylvania from the exact sources used in all published work.

Downloading these files presumably constitutes agreement to Harvard Dataverse's terms of service.
License:
Harvard Election Data Archive by Stephen Ansolabehere and Jonathan Rodden is licensed under a Creative Commons Attribution 3.0 Unported License.
Government data (the files from Census.gov) is in the public domain.
"""

PA_VTD_MAP_ZIP    =  "https://dataverse.harvard.edu/api/access/datafiles/2301829,2301833,2301830,2301834,2301827,2301831,2301835,2301828?gbrecs=true" # shapefiles for precincts/voting districts (2010)
PA_VTD_VOTING_CSV =  "https://dataverse.harvard.edu/api/access/datafile/2710709?format=original&gbrecs=true" # precinct voting data
PA_BG_DEM_ZIP    =  "ftp://ftp2.census.gov/census_2010/01-Redistricting_File--PL_94-171/Pennsylvania/pa2010.pl.zip" # CSV-ish race data by block group (2010)
PA_COUNTIES_MAP_ZIP =  "http://www2.census.gov/geo/tiger/TIGER2013/COUSUB/tl_2013_42_cousub.zip" # shapefiles for Pennsylvania counties

VTD_MAP_DIR = "vtd_map"
VTD_ELECTIONS_CSV = "vtd_elections.csv"
VTD_DEMOGRAPHIC_CSV = "vtd_demographics.csv"
DEMOGRAPHIC_DIR = "pl94_171"
COUNTY_MAP_DIR = "county_map"

import wget # recommended: https://stackoverflow.com/questions/13137817/how-to-download-image-using-requests
import click
import os
import sys
import geopandas as gpd
from zipfile import ZipFile
from glob import glob 

@click.command()
@click.argument('dir')
def download(dir):
    try:
        #os.makedirs(dir)
        os.chdir(dir)
    except OSError:
        print("Could not create directory. Does it already exist?")
        sys.exit(1)

    print("Downloading precinct map and election data (Harvard Dataverse)...")
    dump_zip(PA_VTD_MAP_ZIP, VTD_MAP_DIR)

    print("\nDownloading additional election data (Harvard Dataverse)...")
    wget.download(PA_VTD_VOTING_CSV, out=VTD_ELECTIONS_CSV)

    print("\nDownloading raw demographic data (2010 U.S. Census)...")
    dump_zip(PA_BG_DEM_ZIP, DEMOGRAPHIC_DIR)

    print("\nDownloading county map (U.S. Census, as of 2013)...")
    dump_zip(PA_COUNTIES_MAP_ZIP, COUNTY_MAP_DIR)

    """
    Census data is challenging to parse, so it's convenient to build a CSV with demographic data indexed by geographical region.
    SCOTUS (Evenwel v. Abbott, https://www.oyez.org/cases/2015/14-940) holds that districts should be allocated by total population, not voter-eligible population.
    Thus, we shall use table P2 (Hispanic or Latino over total pop.) to build a demographic summary table.
    For more information, see the PL 94-171 datasheet (https://www.census.gov/prod/cen2010/doc/PL 94-171.pdf).
    
    The table will have the following columns:
    geo_id (mapped to Harvard Dataverse)
    white (non-Hispanic white population in VTD)
    non_white (total VTD population - VTD non-Hispanic white population)

    This could very easily be expanded to more columns if needed, but the common definition of "majority-minority"
    in the United States is "majority *not* non-Hispanic white"; thus, we really only need two quantities to determine
    whether or not a district is majority-minority (true if white/(non_white+white) > 0.5, false otherwise).
    Of course, we can always calculate margins, etc.
    """
    # Map VTD names to GeoIDs from Harvard Dataverse data
    geo = gpd.read_file(glob(os.path.join(VTD_MAP_DIR, '*.shp'))[0])
    names_to_geo_id = {}
    for n, a, id in zip(list(geo['NAME10_1']), list(geo['ALAND10']), list(geo['GEOID10'])):
        names_to_geo_id[(n+str(a)).replace(" Voting District", "").strip()] = id

    # Map Census (PL 94-171) log record numbers to names, changing names to match Harvard conventions
    # https://stackoverflow.com/questions/4309684/split-a-string-with-unknown-number-of-spaces-as-separator-in-python
    record_to_names = {}
    with open(glob(os.path.join(DEMOGRAPHIC_DIR, '*geo*.pl'))[0]) as f:
        for line in f:
            # Only consider records at summary level 700 ("State-County-Voting District/Remainder")
            # That's basically synonymous with "VTD-level"
            if "70000000" in line: # faster to check twice
                raw_cols = line.split('  ')
                cols = []
                for r in raw_cols:
                    if r.strip() != '':
                        cols.append(r.strip())

                if "70000000" in cols[1]:
                    record_to_names[cols[2][:7]] = (cols[7]+cols[6]).lstrip('0123456789').replace(" Voting District", "").strip()

    """
    Generate a summary CSV the hacky way!
    
    Relevant columns in PL 94-171 (0-indexed):
    col 4: record number
    cols 76-end: Table P2
      - P2 col 0: total population (not used)
      -    col 1: Hispanic or Latino population
      -    col 2: non-{Hispanic or Latino} population
      -    col 4: population of "white alone" demographic group (non-Hispanic/Latino)
    """    
    with open(glob(os.path.join(DEMOGRAPHIC_DIR, "*012010.pl"))[0]) as in_f:
        with open(VTD_DEMOGRAPHIC_CSV, 'w') as out_f:
            out_f.write('geo_id,white,other\n')
            for line in in_f:
                data = line.split(',')
                if data[4] in record_to_names:
                    geo_id = names_to_geo_id[record_to_names[data[4]]]
                    p2 = data[76:]
                    hispanic = int(p2[1])
                    all_non_hispanic = int(p2[2])
                    white = int(p2[4])
                    # formatting: cleverly compute minority population total 
                    #             strip two-character state prefix from GeoID to match Harvard conventions 
                    out_f.write('%s,%d,%d\n' % (geo_id[2:], white, hispanic+(all_non_hispanic - white)))

def dump_zip(url, dir):
    os.makedirs(dir)
    os.chdir(dir)
    wget.download(url, out='data.zip')
    ZipFile('data.zip').extractall('.')
    os.remove('data.zip')
    os.chdir('..')

if __name__ == "__main__":
    download()
