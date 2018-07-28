# Elbridge Downloader
Precinct-level election data is spotty. Some states are more digitally conscious than others, and every state has its own conventions. Elbridge currently only supports three states (Maryland, Pennsylvania, and Wisconsin), but it relies on eight different data formats from five different sources. The scripts in this directory are intended to fully reproduce the various transformations necessary to wrangle all of this disparate data into a reasonably consistent format from scratch. There are thus many details in the code specifically related to the data sources used.

## Sources
Elbridge relies on data from [Harvard Election Data Archive](https://projects.iq.harvard.edu/eda/data), a subset of [Harvard Dataverse](https://dataverse.harvard.edu/); [OpenElections](http://openelections.net/); [Census.gov](https://www.census.gov/); [Maryland.gov](https://elections.maryland.gov/); and Wisconsin.gov's [Open Data Portal](https://data-ltsb.opendata.arcgis.com/). URLs for specific data sources are cleanly separated from the transformation code and reside in `data.yml`. The following descriptors are used:
- `vtd_map`: VTD-level election geodata from Harvard Dataverse (some voting data included). Not all `vtd_map` files are formatted identically.
- `demographics`: VTD-level demographic data from the U.S. Census Bureau in accordance with [Public Law 94-171](https://www.census.gov/rdo/about_the_program/public_law_94-171_requirements.html).
- `county_names`: small conversion tables from the U.S. Census Bureau to translate between county-level numerical FIPS codes and human-readable county names.
- `openelections`: VTD-level election data from OpenElections. Similarly to the Harvard data, not all OpenElections data is formatted the same way.
- `md_gov`: VTD-level election data from Maryland.gov.
- `ward_maps`: ward-level election data from Wisconsin.gov.

Note that "VTD" is an abbreviation for "voting district," which is essentially synonymous to "precinct."

## Downloading Data
Use `get.py` to download and parse data. It depends on being in this folder and expects two arguments: a two-letter state code and an empty directory to dump data to. Example usage:

`python get.py PA PA_data`

 Certain source websites dislike repeated calls and may ban IPs they perceive as abusive, so if it is necessary to reprocess data several times, call `get.py` with the `--no-download` flag. FTP-based Census data is particularly prone to timeouts.

## Output
`get.py` generates the following summary files (names configurable):
- `vtd_demographics.csv`: VTD-level/ward-level demographic data necessary to determine whether or not a district is majority-minority. The `white` column is the number of non-Hispanic white people in a district; the `other` column is the number of people who are non-white or Hispanic white in a district.
- `vtd_elections.csv`: VTD-level/ward-level vote counts (DV = Democratic votes, RV = Republican votes). Vote totals outside of the two major parties are omitted. Some VTDs may only have data available for years, as VTD maps can change.

Additionally, `get.py` retains the shapefiles that contain VTD/ward boundaries in either `vtd_map` or `ward_maps`. Raw data from the U.S. Census Bureau and OpenElections is deleted after processing, though this deletion can be disabled with the `--keep-census` and `--keep-openelections` flags, respectively.