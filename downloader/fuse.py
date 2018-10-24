"""
Fusion of multiple data sources with details specific to individual states.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from get import FILENAMES, DIRECTORIES
import convert
import os

def fuse_MD(filenames, sources):
    """
    Fuse data from Harvard Dataverse (2006-2008), OpenElections (2008-2014), and Maryland.gov (2014-2016)
    """
    HARVARD_DV_2006 = "USH_DVOTE0"
    HARVARD_RV_2006 = "USH_RVOTE0"
    HARVARD_DV_2008 = "USH_DVOTE_"
    HARVARD_RV_2008 = "USH_RVOTE_"
    HARVARD_GEO_ID  = "GEOID10"

    harvard = gpd.read_file(os.path.join(DIRECTORIES['vtd_map'], filenames['vtd_map']))
    openelections = {}
    for year in sources['openelections']:
        open_file = os.path.join(DIRECTORIES['openelections'], 'MD_%d.csv' % year)
        openelections[year] = convert.load_md_raw_openelections(open_file, FILENAMES['county_names'], 'dv', 'rv', 'geo_id')
    open_to_harvard = a_to_b_concordance(openelections[2008], harvard, 'dv', 'rv', 'geo_id', HARVARD_DV_2008, HARVARD_RV_2008, HARVARD_GEO_ID)
    md_gov = {}
    for year in sources['md_gov']:
        md_file = os.path.join(DIRECTORIES['md_gov'], 'MD_%d.csv' % year)
        md_gov[year] = convert.load_md_gov_data(md_file, FILENAMES['county_names'], 'dv', 'rv', 'geo_id')
    md_gov_to_open = a_to_b_concordance(md_gov[2012], openelections[2012], 'dv', 'rv', 'geo_id', 'dv', 'rv', 'geo_id')

    cols = {
        'geo_id':  harvard[HARVARD_GEO_ID],
        'dv_2006': harvard[HARVARD_DV_2006],
        'rv_2006': harvard[HARVARD_RV_2006],
        'dv_2008': harvard[HARVARD_DV_2008],
        'rv_2008': harvard[HARVARD_RV_2008]
    }
    harvard_id_to_idx = {}
    i = 0
    for row in harvard.itertuples():
        harvard_id_to_idx[getattr(row, HARVARD_GEO_ID)] = i
        i += 1
    for year in range(2008, 2018, 2):
        cols['dv_%d' % year] = np.zeros(len(harvard), dtype=np.int64) 
        cols['rv_%d' % year] = np.zeros(len(harvard), dtype=np.int64)

    # There's a bit of code duplication here, but it's in the interest of clarity :)
    # 2010-2014: OpenElections
    for year in openelections:
        not_found = 0
        for row in openelections[year].itertuples():
            id = getattr(row, 'geo_id')
            dv = getattr(row, 'dv')
            rv = getattr(row, 'rv')
            if dv > 0 or rv > 0:
                try:
                    idx = harvard_id_to_idx[open_to_harvard[id]]
                    cols['dv_%d' % year][idx] = dv
                    cols['rv_%d' % year][idx] = rv
                except KeyError:
                    not_found += 1
        if not_found > 0:
            print("\tWarning: %d precincts in OpenElections data (%d) not found in Harvard Dataverse data." % (not_found, year))
    # 2016: Maryland.gov
    not_found = 0
    for row in md_gov[2016].itertuples():
        id = getattr(row, 'geo_id')
        dv = getattr(row, 'dv')
        rv = getattr(row, 'rv')
        if dv > 0 or rv > 0:
            try:
                idx = harvard_id_to_idx[open_to_harvard[md_gov_to_open[id]]]
                cols['dv_%d' % year][idx] = dv
                cols['rv_%d' % year][idx] = rv
            except KeyError:
                not_found += 1
    if not_found > 0:
        print("\tWarning: %d precincts in Maryland.gov data (%d) not found in Harvard Dataverse data." % (not_found, year))
        
    pd.DataFrame(cols).to_csv(FILENAMES['elections'], index=False)

def fuse_PA(filenames, sources):
    """
    We want to fuse the OpenElections data with the Harvard Dataverse data while keeping the Harvard GeoID conventions.
    That way, the data will line up with the Dataverse geodata as much as possible.
    Sources:
        - Dataverse voting data from 2006-2010 (preferred when overlapping; 2002-2004 data appears incomplete and doesn't matter much anyway)
        - OpenElections voting data from 2008-2016
    """
    filenames['openelections'] = {}
    for year in range(2008, 2018):
        filenames['openelections'][year] = '%s_%d.csv' % ("PA", year)

    print("\nFusing OpenElections and Harvard Dataverse voting data...") # TODO: move to fuse.py
    PA_STATE_PREFIX = '42'
    harvard_file = os.path.join(DIRECTORIES['vtd_map'], filenames['vtd_map'])  
    open_files = {}
    for year in [2008, 2010]:
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
    pd.DataFrame(vote_cols).to_csv(FILENAMES['elections'], index=False)  

def fuse_WI(filenames, sources):
    """
    Wisconsin election data is simple relative to other states.
    Demographic data is included with geodata, and all of this data is officially available from the Wisconsin government.
    It's even relatively up to date!
    """
    print("Parsing shapefiles...")
    gdf = {}
    for year in sources['ward_maps']:
        gdf[year] = gpd.read_file(os.path.join(DIRECTORIES['ward_maps'], str(year), filenames['ward_maps'][year]))
    cols = {
        "geo_id": gdf[2010]['WARD_FIPS']
    }
    # data is available before 2006, but other states cut off at 2006, so it's probably best to follow that convention
    for year in range(2006, 2018, 2):
        gdf_year = 2010 if year < 2012 else 2020
        cols['dv_%d' % year] = gdf[gdf_year]['USHDEM%s' % str(year)[-2:]]
        cols['rv_%d' % year] = gdf[gdf_year]['USHREP%s' % str(year)[-2:]]
    pd.DataFrame(cols).to_csv(FILENAMES['elections'], index=False)

def a_to_b_concordance(a_df, b_df, a_dv, a_rv, a_geo_id, b_dv, b_rv, b_geo_id):
    """Generalized A->B concordance. """
    a_geo = set(a_df[a_geo_id])
    b_geo = set(b_df[b_geo_id])
    diff = a_geo ^ b_geo
    a_to_b = {}

    # Generate identity mapping for intersection
    # (assumes that if the same GeoIDs occur in both DataFrames, they refer to the same VTD)
    for id in a_geo & b_geo:
        a_to_b[id] = id

    b_geo_id_to_votes = {}
    for row in b_df.itertuples():
        geo_id = getattr(row, b_geo_id)
        if geo_id in diff:
            dv = getattr(row, b_dv)
            rv = getattr(row, b_rv)
            if dv > 0 or rv > 0:
                b_geo_id_to_votes[geo_id] = (dv,rv)

    a_votes_to_geo_id = {}
    for row in a_df.itertuples():
        geo_id = getattr(row, a_geo_id)
        if geo_id in diff:
            dv = getattr(row, a_dv)
            rv = getattr(row, a_rv)
            if dv > 0 or rv > 0:
                if (dv,rv) in a_votes_to_geo_id:
                    print("Warning: vote count overlap (%d, %d)" % (dv,rv))
                a_votes_to_geo_id[(dv,rv)] = geo_id

    for b_id in b_geo_id_to_votes:
        votes = b_geo_id_to_votes[b_id]
        if votes in a_votes_to_geo_id:
            a_to_b[a_votes_to_geo_id[votes]] = b_id 

    return a_to_b