#!/usr/bin/env python
""" Script for reproducibly creating the pa129 fixture. """
import os
import sys
import geopandas as gpd
from shutil import rmtree
from zipfile import ZipFile
from urllib.request import urlretrieve

PA_DATA_ZIP = "https://github.com/mggg-states/PA-shapefiles/raw/" \
              "a9fa163573c68338b42129ac4784ec179ef32a27/PA/PA_VTD.zip"
PA_FILENAME = "PA_VTD.zip"
PA_DIR = "PA_VTD"
PA_SHP = "PA_VTD.shp"
OUT_DIR = "pa129"
OUT_SHP = "pa129.shp"
PA_COUNTY = "129"

if __name__ == "__main__":
    if not os.path.isdir(PA_DIR):
        urlretrieve(PA_DATA_ZIP, PA_FILENAME)
    else:
        print("Dir '{}' already exists!".format(PA_DIR), file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)
    else:
        print("Dir '{}' already exists!".format(OUT_DIR), file=sys.stderr)
        sys.exit(1)
    ZipFile(PA_FILENAME).extractall(PA_DIR)
    pa = gpd.read_file(os.path.join(PA_DIR, PA_SHP))
    pa129 = pa[pa.COUNTYFP10 == PA_COUNTY].reset_index()
    pa129.to_file(os.path.join(OUT_DIR, OUT_SHP))
    rmtree(PA_DIR)
    os.remove(PA_FILENAME)
