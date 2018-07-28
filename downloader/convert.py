"""
Converters to wrangle precinct-level House of Representatives voting data from many sources (Harvard Dataverse, U.S. Census Bureau, OpenElections) in many formats (.shp, .csv, etc.)
"""
import pandas as pd
import geopandas as gpd
from collections import defaultdict

"""
Census data is challenging to parse, so it's convenient to build a CSV with demographic data indexed by geographical region.
SCOTUS (Evenwel v. Abbott, https://www.oyez.org/cases/2015/14-940) holds that districts should be allocated by total population, not voter-eligible population.
Thus, we shall use table P2 (Hispanic or Latino over total pop.) to build a demographic summary table.
For more information, see the PL 94-171 datasheet (https://www.census.gov/prod/cen2010/doc/pl94-171.pdf).

The table will have the following columns:
geo_id (mapped to Harvard Dataverse)
white (non-Hispanic white population in VTD)
non_white (total VTD population - VTD non-Hispanic white population)

This could very easily be expanded to more columns if needed, but the common definition of "majority-minority"
in the United States is "majority *not* non-Hispanic white"; thus, we really only need two quantities to determine
whether or not a district is majority-minority (true if white/(non_white+white) > 0.5, false otherwise).
"""
def dataverse_to_demographics(vtd_shapefile, demographic_geo, demographic_02, outfile):
    """Parse U.S. Census demographic data (PL 94-171, 2010) based on Harvard Dataverse precinct-level data.
       
       Arguments:
       - vtd_shapefile:    shapefile from Harvard Dataverse with precinct-level voting data.
       - demographic_geo:  Geographic Header Record file in PL 94-171 data.
       - demographic_02:   File02 in PL 94-171 data.
       - outfile:          CSV to dump VTD-aggregated demographic data to.

       Returns: None.
    """
    # Map VTD names to GeoIDs from Harvard Dataverse data
    geo = gpd.read_file(vtd_shapefile)
    names_to_geo_id = {}
    for n, a, id in zip(list(geo['NAME10']), list(geo['ALAND10']), list(geo['GEOID10'])):
        names_to_geo_id[(n+str(a)).replace(" Voting District", "").strip()] = id

    # Map Census (PL 94-171) log record numbers to names, changing names to match Harvard conventions
    # https://stackoverflow.com/questions/4309684/split-a-string-with-unknown-number-of-spaces-as-separator-in-python
    record_to_names = {}
    with open(demographic_geo) as f:
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
    with open(demographic_02) as in_f:
        with open(outfile, 'w') as out_f:
            out_f.write('geo_id,white,other\n')
            for line in in_f:
                data = line.split(',')
                if data[4] in record_to_names:
                    geo_id = names_to_geo_id[record_to_names[data[4]]]
                    p2 = data[76:]
                    hispanic = int(p2[1])
                    all_non_hispanic = int(p2[2])
                    white = int(p2[4])
                    # formatting: compute minority population total 
                    #             strip two-character state prefix from GeoID to match Harvard conventions 
                    out_f.write('%s,%d,%d\n' % (geo_id[2:], white, hispanic+(all_non_hispanic - white)))


def wi_to_demographics(ward_shapefile, outfile):
    """Parse demographic data included in the Wisconsin shapefile into a separate CSV consistent with dataverse_to_demographics(). 
    Note that the statewide totals of the per-ward demographic data seems to vary from the statewide totals given by the 2010 U.S. Census.
    See https://www.census.gov/quickfacts/fact/table/wi/PST045217.

    Arguments:
    - ward_shapefile:  Wisconsin shapefile from Wisconsin.gov/ArcGIS with ward-level voting data.
    - outfile:        CSV to dump VTD-aggregated demographic data to.

    Returns: None.
    """
    wards = gpd.read_file(ward_shapefile)
    cols = {
        "geo_id": wards['WARD_FIPS'],
        "white": wards['WHITE'],
        "other": wards['PERSONS'] - wards['WHITE']
    }
    pd.DataFrame(cols).to_csv(outfile, index=False)

"""
Harvard Dataverse conveniently includes precinct-level election data from ~2002 (±2 years, depending on the state) to 2010 inline with geographical data.
However, it doesn't go beyond 2010, which is obviously problematic—it's rather unrealistic to redistrict based on rather ancient election data.
The most comprehensive source available is OpenElections, which appears reputable and even seems to be a source for the MIT Election Lab data (https://electionlab.mit.edu/data).

Unfortunately, the conventions used in the OpenElections data don't quite line up with the Harvard data *exactly*.
There are some trivial things—for instance, the Harvard data uses FIPS county codes, whereas the OpenElections data tends to sort the county names alphabetically and assign sequential codes. 
This can be fixed with small tables easily sourced from the Census website. However...
A more troublesome issue is that some number of precincts (~1% in PA) don't match between the two sources, even after conversion.
To fix this, we can cross-reference vote tallies. This isn't *guaranteed* to be unique--CHECK FOR THIS WHEN ADDING NEW STATES--but it typically is (you get two numbers: DV & RV).
To further fix these discrepancies, we can cross reference the Dataverse data with multiple years of OpenElections data and take a union.

Once again, these techniques aren't guaranteed to produce a perfect result, but they're better than nothing.
If you're adding new states, be very cautious and do some sanity checks.
"""

def convert_counties(counties_list, counties_file):
    """Convert alphabetically indexed county IDs to FIPS-indexed county IDs.
       Arguments:
       - counties_list: list of county IDs to be converted. Typically, this might be a converted row from a DataFrame.
       - counties_file: county names -> FIPS codes table from Census Bureau

       Returns a list of FIPs-indexed county IDs corresponding with counties_list. 
    """
    counties = pd.read_csv(counties_file, names=['state', 'state_code', 'county_code', 'county_name', ''])
    name_to_fips_code = {}
    for row in counties.itertuples():
        name = getattr(row, 'county_name').rstrip(' County').upper()
        name_to_fips_code[name] = str(getattr(row, 'county_code')).zfill(3)

    open_code_to_name = {}
    names = sorted(list(name_to_fips_code.keys()))
    for i, n in enumerate(names):
        open_code_to_name[i+1] = n

    fips_counties = []
    for c in counties_list:
        fips_counties.append(name_to_fips_code[open_code_to_name[c]])
    return fips_counties

def load_openelections_data(open_file, counties_file, dv_col, rv_col, geo_col, state_prefix):
    """Load precinct-level House of Representatives election data into a Pandas DataFrame from an OpenElections CSV.
       Arguments:
       - open_file: CSV (specific to a particular year) from OpenElections with precinct-level voting data
       - counties_file: county names -> FIPS codes table from Census Bureau
       - dv_col:        name of *D*emocratic *v*otes column in processed data
       - rv_col:        name of *R*epublican *v*otes column in processed data
       - geo_col:       name of GeoID column in processed data
       - state_prefix:  two-digit numerical identifier for each state based on alphabetization

       Returns a Pandas DataFrame with processed data.
    """

    cols = ['year', 'type', 'open_county', 'precinct', 'cand_office_rank', 'cand_district', 'cand_party_rank', 'cand_ballot_pos',
            'cand_office', 'cand_party', 'cand_number', 'cand_last', 'cand_first', 'cand_middle', 'cand_suffix', 'vote_total', 
            'us_district', 'state_senate_district', 'state_house_district', 'municipality_type', 'municipality_name',
            'municipality_bd_code_1', 'municipality_bd_name_1', 'municipality_bd_code_2', 'municipality_bd_name_2', 'bi_county_code',
            'mcd', 'fips', 'vtd', 'previous_precinct', 'previous_district', 'previous_state_senate_district', 'previous_state_house_district']
    open_raw = pd.read_csv(open_file, low_memory=False, names=cols)
    house_data = open_raw[open_raw.cand_office=="USC"][['open_county', 'precinct', 'vtd','cand_party','vote_total']]
    house_data['fips_county'] = convert_counties(list(house_data['open_county']), counties_file)

    # Generate final table with Dem/Rep vote in one row per precinct
    dem_vote_by_vtd = {}
    rep_vote_by_vtd = {}
    for row in house_data.itertuples():
        party = getattr(row, 'cand_party')
        county = getattr(row, 'fips_county')
        vtd = getattr(row, 'vtd')
        # convert VTD to Dataverse convention
        full_vtd = state_prefix + county + str(vtd).lstrip('0')
        votes = getattr(row, 'vote_total')

        if party == "DEM" and \
          (full_vtd not in dem_vote_by_vtd or (dem_vote_by_vtd[full_vtd] == 0 and votes > 0)):
            dem_vote_by_vtd[full_vtd] = votes
            # Fill in the *other* column
            if full_vtd not in rep_vote_by_vtd:
                rep_vote_by_vtd[full_vtd] = 0

        elif party == "REP" and \
            (full_vtd not in rep_vote_by_vtd or (rep_vote_by_vtd[full_vtd] == 0 and votes > 0)):
            rep_vote_by_vtd[full_vtd] = votes
            # Fill in the *other* column (see above)
            if full_vtd not in dem_vote_by_vtd:
                dem_vote_by_vtd[full_vtd] = 0

    return _render_to_df(dem_vote_by_vtd, rep_vote_by_vtd, dv_col, rv_col, geo_col)


def load_md_raw_openelections(open_file, counties_file, dv_col, rv_col, geo_col, state_prefix='24'):
    """Load precinct-level House of Representatives election data into a Pandas DataFrame from an OpenElections Marylnd-format CSV.
       Note that this format is distinct from that processed by load_openelections_data().
       It seems to be raw data downloaded off a government website or copied directly from an Excel spreadsheet.
       Arguments:
       - open_file: CSV (specific to a particular year) from OpenElections with precinct-level voting data
       - counties_file: county names -> FIPS codes table from Census Bureau
       - dv_col:        name of *D*emocratic *v*otes column in processed data
       - rv_col:        name of *R*epublican *v*otes column in processed data
       - geo_col:       name of GeoID column in processed data
       - state_prefix:  two-digit numerical identifier for each state based on alphabetization

       Returns a Pandas DataFrame with processed data.
    """
    raw = pd.read_csv(open_file, low_memory=False)
    congress = raw[(raw.office=='U.S. Congress') & ((raw.party=='REP') | (raw.party=='DEM'))]

    county_name_to_fips = {}
    with open(counties_file) as f:
        for row in f:
            county_name = row.split(',')[3].replace(' County', '').replace(' city', '').replace('St.', 'St').lower()
            fips = row.split(',')[2].zfill(3)
            county_name_to_fips[county_name] = fips
    
    dem_vote_by_vtd = defaultdict(int)
    rep_vote_by_vtd = defaultdict(int)
    # Convert jurisdiction columns in OpenElections to Harvard Dataverse GeoID conventions
    for row in congress.itertuples():
        county = getattr(row, 'division').split('/')[-2].split(':')[1].replace('_', ' ').replace('~', "'")
        county_fips = county_name_to_fips[county]
        precinct = getattr(row, 'jurisdiction')[1:] # strip leading 0
        geo_id = state_prefix + county_fips + precinct
        
        party = getattr(row, 'party')
        votes = getattr(row, 'votes')
        if party == 'DEM' and votes > dem_vote_by_vtd[geo_id]:
            dem_vote_by_vtd[geo_id] = votes
            rep_vote_by_vtd[geo_id] = rep_vote_by_vtd[geo_id] # identity init
        elif party == 'REP' and votes > rep_vote_by_vtd[geo_id]:
            rep_vote_by_vtd[geo_id] = votes
            dem_vote_by_vtd[geo_id] = dem_vote_by_vtd[geo_id]
            
    # TODO: make X_votes_by_vtd variable name consistent throughout file
    return _render_to_df(dem_vote_by_vtd, rep_vote_by_vtd, dv_col, rv_col, geo_col)

def load_md_gov_data(md_gov_file, counties_file, dv_col, rv_col, geo_col, state_prefix='24'):
    """Load precinct-level House of Representatives election data into a Pandas DataFrame from official Maryland government data (currently unused).
        Arguments:
        - open_file: CSV (specific to a particular year) from Maryland.gov with precinct-level voting data
        - counties_file: county names -> FIPS codes table from Census Bureau
        - dv_col:        name of *D*emocratic *v*otes column in processed data
        - rv_col:        name of *R*epublican *v*otes column in processed data
        - geo_col:       name of GeoID column in processed data
        - state_prefix:  two-digit numerical identifier for each state based on alphabetization

        Returns a Pandas DataFrame with processed data.
    """
    raw = pd.read_csv(md_gov_file, low_memory=False, encoding='latin-1')
    # filter out non Rep/Dem votes for consistency with other datasets
    congress = raw[(raw['Office Name'] == 'Rep in Congress') & ((raw['Party']=='REP') | (raw['Party']=='DEM'))].copy()
    congress["fips_county"] = convert_counties(list(congress['County']), counties_file)
    rv_by_precinct = defaultdict(int) # highest Republican vote by precinct
    dv_by_precinct = defaultdict(int) # highest Democratic vote by precinct

    for _, row in congress.iterrows():
        geo_id = ('%s%s%s-%s') % (state_prefix, row['fips_county'], str(row['Election District']).zfill(2), str(row['Election Precinct']).zfill(3))
        try:
            votes = int(row['Election Night Votes'])
        except ValueError:
            votes = 0

        if row.Party == "REP":
            rv_by_precinct[geo_id] = max(rv_by_precinct[geo_id], votes)
            if geo_id not in dv_by_precinct:
                dv_by_precinct[geo_id] = 0
        elif row.Party == "DEM":
            dv_by_precinct[geo_id] = max(dv_by_precinct[geo_id], votes)
            if geo_id not in rv_by_precinct:
                rv_by_precinct[geo_id] = 0

    return _render_to_df(dv_by_precinct, rv_by_precinct, dv_col, rv_col, geo_col)
        
def _diff_to_votemap(df, diff, dv_col, rv_col, geo_col):
    """ Helper method for building vote-based concordance tables. """
    votes_to_id = {}
    for row in df.itertuples():
        geo_id = getattr(row, geo_col)
        if geo_id in diff:
            dv = getattr(row, dv_col)
            rv = getattr(row, rv_col)
            if dv > 0 or rv > 0:
                if (dv,rv) not in votes_to_id:
                    votes_to_id[(dv,rv)] = geo_id
                else:
                    print('Warning: overlap for votes (%d, %d)' % (dv,rv))
    return votes_to_id

def map_open_harvard(harvard_file, open_file, counties_file, dv_col, rv_col, geo_col, state_prefix):
    """Try to munge Harvard and OpenElections data into a pleasing mélange.
       This is best run multiple times with data from multiple years to get a union of the OpenElections->Harvard map.

       Arguments:
       - harvard_file:  CSV or shapefile from Harvard Dataverse with precinct-level voting data.
       - open_file:     CSV (specific to a particular year) from OpenElections with precinct-level voting data.
       - counties_file: county names -> FIPS codes table from Census Bureau
       - dv_col:        name of *D*emocratic *v*otes column in Dataverse (varies by year and state); OpenElections data will be converted to match
       - rv_col:        name of *R*epublican *v*otes column in Dataverse (varies by year and state); OpenElections data will be converted to match
       - geo_col:       name of GeoID column in Dataverse (varies by year); OpenElections data will be converted to match
       - state_prefix:  two-digit numerical identifier for each state based on alphabetization

       Returns (3-tuple):
       [0] a concordance dictionary (OpenElections GeoID -> Harvard GeoID)
       [1] Harvard Dataverse GeoDataFrame
       [2] Processed OpenElections DataFrame
    """
    harvard_df = gpd.read_file(harvard_file) # shapefile (2010)
    open_df = load_openelections_data(open_file, counties_file, dv_col, rv_col, geo_col, state_prefix)
    
    open_geo = set(open_df[geo_col]) # s_geo
    harvard_geo = set(harvard_df[geo_col]) # t_geo
    
    # orphans: in Harvard but not in OpenElections
    in_harvard = {}
    harvard_not_open = harvard_geo - open_geo
    for row in harvard_df.itertuples():
        geo_id = getattr(row, geo_col)
        dv = int(getattr(row, dv_col))
        rv = int(getattr(row, rv_col))
        if geo_id in harvard_not_open and (dv != 0 or rv != 0):
            in_harvard[(dv,rv)] = geo_id
    
    # orphans: in OpenElections but not in Harvard
    in_open = {}
    open_not_harvard = open_geo - harvard_geo
    for row in open_df.itertuples():
        geo_id = getattr(row, geo_col)
        dv = int(getattr(row, dv_col))
        rv = int(getattr(row, rv_col))
        if geo_id in open_not_harvard and (dv != 0 or rv != 0):
            in_open[(dv,rv)] = geo_id
            
    open_to_harvard = {}
    # map OpenElections IDs to Harvard IDs
    for votes in in_open:
        try:
            open_to_harvard[in_open[votes]] = in_harvard[votes]
        except KeyError: pass
    
    # add identity mappings (that is, cases where the GeoID in Harvard == the GeoID in Open)
    intersection = harvard_geo & open_geo
    for row in harvard_df.itertuples():
        geo_id = getattr(row, geo_col)
        if geo_id in intersection:
            open_to_harvard[geo_id] = geo_id

    return (open_to_harvard, harvard_df, open_df)

def _render_to_df(dv, rv, dv_col, rv_col, geo_col):
    """ Return tallied precinct-level Democratic and Republican votes as a Pandas DataFrame. """
    rows = {
        geo_col: list(dv.keys()),
        dv_col:  list(dv.values()),
        rv_col:  list(rv.values())
    }
    return pd.DataFrame(rows)