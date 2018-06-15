"""
TODO: Models for reasonably approximating missing {DV,RV} vote totals in precinct $x$ for year $y$ based on all available features, namely:
- Performance of precinct $x$ in other elections
- Performance of $k$ nearest geographically neighboring precincts in year $y$ and other years
- Population density of precinct $x$ and its $k$ nearest neighbors
- 2010 population of           ""
- 2010 white population of     ""
- 2010 non-white population of ""

This model doesn't need to be perfect—it can't be!
Ideally we would produce two aggregated vote datasets: a non-interpolated version with a lot of 0s and an interpolated version with a disclaimer that some data is artificial and may be wrong.
However, interpolation that's slightly off is probably better than a bunch of 0s when ≤1% of data is missing—the goal is just to avoid having weird outliers. 
"""
# import xgboost (?—it's good enough for Kaggle)