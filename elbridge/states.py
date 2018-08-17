from map import Map
"""
Wrapper code to import per-state processed data (as downloaded by the downloader script) and apply a bit more transformation.
"""

import pandas    as  pd
import geopandas as gpd
import numpy     as  np
import warnings

# as of 2010
N_DISTRICTS = {
    "MD": 8,
    "PA": 18,
    "WI": 8 
}

"""
Conventions:
rv_vote_share = rv / (dv+rv), but if dv+rv = 0, we let rv_vote_share = 0.
This won't ultimately affect the final vote tally in each district when evaluating maps--the precinct simply will not count, as raw votes are summed.
The same convention applies to U.S. Census population totals.
"""

class Wisconsin(Map):
    def __init__(self, ward_map_file, vtd_elections_file, vtd_demographics_file, resolution, decay):
        ward_map = gpd.read_file(ward_map_file)
        vtd_elections    = pd.read_csv(vtd_elections_file)
        vtd_demographics = pd.read_csv(vtd_demographics_file)

        columns = {
            "geo_id": ward_map['GEOID'],
            "geometry": ward_map['geometry'],
            "county": ward_map['CNTY_FIPS'],
            "city": ward_map['MCD_FIPS'],
            "white_pop": vtd_demographics['white'], # demographic data is a subset of all ward data, so it should map 1:1
            "minority_pop": vtd_demographics['other']
        }
        for year in range(2006, 2018, 2):
            columns['dv_%d' % year] = vtd_elections['dv_%d' % year]
            columns['rv_%d' % year] = vtd_elections['rv_%d' % year]
            total = vtd_elections['dv_%d' % year] + vtd_elections['rv_%d' % year]
            total[total==0] = 1 # avoid divide-by-zero
            columns['rv_vote_share_%d' % year] = columns['rv_%d' % year] / total

        self.df = gpd.GeoDataFrame(columns)
        self.df.crs = ward_map.crs
        self.n_districts = N_DISTRICTS["WI"]
        super().__init__(resolution, decay)

class Pennsylvania(Map):
    # TODO: add county map?
    def __init__(self, vtd_map_file, vtd_elections_file, vtd_demographics_file, resolution, decay):
        vtd_map          = gpd.read_file(vtd_map_file)
        vtd_elections    = pd.read_csv(vtd_elections_file)
        vtd_demographics = pd.read_csv(vtd_demographics_file)


        """
        Columns:
        - geo_id
        - dv_2006 ... dv_2016
        - rv_2006 ... rv_2016
        - rv_share_2006 ... rv_share_2016
        - total_pop
        - white_pop
        - minority_pop
        - minority_prop
        - geometry
        """
        columns = {}
        columns['geo_id']   = vtd_map['GEOID10']
        columns['geometry'] = vtd_map['geometry']
        columns['name']     = vtd_map['NAME10']
        columns['county']   = vtd_map['COUNTYFP10']

        for y in range(2006, 2018, 2):
            columns['dv_%d' % y] = vtd_elections['dv_%d' % y]
            columns['rv_%d' % y] = vtd_elections['rv_%d' % y]
            columns['rv_share_%d' % y] = []

        geo_id_to_idx = {}
        # https://stackoverflow.com/questions/15933741/how-do-i-catch-a-numpy-warning-like-its-an-exception-not-just-for-testing
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx in range(len(vtd_elections)):
                row = vtd_elections.iloc[idx]
                geo_id_to_idx[getattr(row, 'geo_id')[2:]] = idx
                for y in range(2006, 2018, 2):
                    try:
                        columns['rv_share_%d' % y].append(row['rv_%d' % y] / (row['rv_%d' % y] + row['dv_%d' % y]))
                    except RuntimeWarning:
                        columns['rv_share_%d' % y].append(0)

        # demographics
        white_pop = [0] * len(columns['dv_2010'])
        minority_pop = [0] * len(columns['dv_2010'])
        minority_prop = [0] * len(columns['dv_2010'])
        for row in vtd_demographics.itertuples():
            idx = geo_id_to_idx[getattr(row, "geo_id")]
            white = getattr(row, "white")
            non_white = getattr(row, "other")
            white_pop[idx] = white
            minority_pop[idx] = non_white
            try:
                minority_prop[idx] = non_white / (white + non_white)
            except ZeroDivisionError: pass
        columns['white_pop'] = white_pop
        columns['minority_pop'] = minority_pop
        columns['minority_prop'] = minority_prop

        self.df = gpd.GeoDataFrame(data=columns)
        self.df.crs = vtd_map.crs
        self.df['total_pop'] = self.df['white_pop'] + self.df['minority_pop']

        keywords = ["VTD", "Voting District", "WD", "DISTRICT", "PRECINCT", "TWP", "PCT", "DIST", "TOWNSHIP", "CITY", "BORO", "BOROUGH", "TOWN", "CITY"]
        for i in range(1, 10): keywords.append(str(i)) # integers 1-9
        cities = []
        for row in self.df.itertuples():
            city = getattr(row, 'name')
            for k in keywords:
                city = city.split(' ' + k)[0]
            cities.append(city)
        self.df['city'] = cities
        super().__init__(resolution, decay)


        """
        Look into:
        BUFFALO/BUFFALO TWO
        CHERRY/CHERRY VALLEY
        LANGHORNE MANOR/LANGHORNE
        JOHNSTOWN *
        LOWER ALLEN ANNEX
        LONDONDERRY TWO
        UPPER PROVIDENC/UPPER PROVIDENCE
        Voting Districts not defined
        """
