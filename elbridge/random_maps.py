import states
import pickle
import click
import logging
import os
from json   import dumps
from copy   import deepcopy
from random import random
from math   import pi, exp
from statistics      import mean
from collections     import OrderedDict
from datetime        import datetime, timezone
from hashlib         import sha512
from shapely.ops     import unary_union
from google.cloud    import storage
from multiprocessing import Pool

N_WORKERS = 1             # number of maps to generate concurrently
ITERATIONS = 15000        # quit if this number of iterations is reached
RESET_ITERATIONS = 150    # reset the current district if no progress is made after this number of iterations
MAX_DELTA = 0.1           # in proportion of population (0.1 = go 10% of the state population away in one step)
MIN_YEAR = 2008           # for voting stats
MAX_YEAR = 2016           # for voting stats
COMPETITIVE_MARGIN = 0.05 # for voting stats (a district is said to be "competitive" 
                          # if the winning party's margin over 50% is less than 5%)
LAMBDA = 0.5              # for voting stats (exponential decay)
FORMAT = '%(asctime)-15s %(level)s %(message)s' # for logging

@click.command()
@click.argument('pkl')
@click.argument('density')
@click.argument('n_maps', type=int)
@click.argument('output_bucket')
def run(pkl, density, n_maps, output_bucket):
    logging.basicConfig(format=FORMAT, level=logging.WARNING)
    state = pickle.load(open(pkl, 'rb'))
    state.load_density_mapping(density)
    params = [(state, output_bucket) for _ in range(n_maps)]
    with Pool(N_WORKERS) as p:
        p.starmap_async(generate_map, params, error_callback=log_error)
        p.close()
        p.join()

def log_error(e):
    logging.error(e, exc_info=True)

def generate_map(state, bucket_name):
    state = deepcopy(state)
    state.reset()
    last_pop = 0
    iters_since_pop_change = 0
    for _ in range(ITERATIONS):
        if state.done: break
        if last_pop != state.district_pop_allocated:
            iters_since_pop_change = 0
        elif iters_since_pop_change > RESET_ITERATIONS:
            state.reset_district()
            last_pop = 0
            iters_since_pop_change = 0
        last_pop = state.district_pop_allocated
        iters_since_pop_change += 1

        delta = MAX_DELTA*random()
        theta = 2*pi*random()
        state.allocate(delta, theta)

    if state.done:
        stats   = {**generate_stats(state), 'created_at': datetime.now(timezone.utc).isoformat()}
        hash_id = sha512(dumps(stats).encode('utf-8')).hexdigest()
        os.makedirs(bucket_name, exist_ok=True)
        """
        bucket  = storage.Client().get_bucket(bucket_name)
        blob    = bucket.blob(hash_id[:12] + '.json')
        blob.upload_from_string(dumps({**stats, 'hash_id': hash_id}), content_type='application/json')
        print('uploaded', hash_id[:12] + '.json')
        """
        with open(os.path.join(bucket_name, '{}.json'.format(hash_id[:12])), 'w') as f:
            f.write(dumps({**stats, 'hash_id': hash_id}))

def generate_stats(state):
    districts = []
    for vtds in state.vtd_by_district[1:]:
        df = state.df.iloc[vtds]
        bounds = unary_union(df.geometry)
        area = bounds.area
        perimeter = bounds.length
        district = {
            'vtds':      vtds,
            'area':      area,
            'perimeter': perimeter,
            'white_pop':     int(df['white_pop'].sum()),
            'minority_pop':  int(df['minority_pop'].sum()),
            'minority_prop': df['minority_prop'].mean(),
            'total_pop':     int(state.total_pop[vtds].sum()),
            'dv': OrderedDict(),
            'rv': OrderedDict(),
            'dv_share': OrderedDict(),
            'rv_share': OrderedDict()
        }
        district['convex_hull']   = area / bounds.convex_hull.area
        district['polsby_popper'] = (4*pi*area) / (perimeter**2)
        for year in range(MIN_YEAR, MAX_YEAR + 2, 2):
            for v in ('dv', 'rv'):
                district[v][year] = int(state.df.iloc[vtds]['%s_%d' % (v, year)].sum())
            dv = int(district['dv'][year])
            rv = int(district['rv'][year])
            district['dv_share'][year] = dv / (dv + rv)
            district['rv_share'][year] = rv / (dv + rv)
        districts.append(district)

    overall = {
        'mean_convex_hull':   mean([d['convex_hull'] for d in districts]),
        'mean_polsby_popper': mean([d['polsby_popper'] for d in districts]),
        'n_dem': OrderedDict(),
        'n_rep': OrderedDict(),
        'n_competitive': OrderedDict()
    }
    for year in range(MIN_YEAR, MAX_YEAR + 2, 2):
        overall['n_dem'][year]    = sum([1 if d['rv_share'][year] < 0.5 else 0 for d in districts])
        overall['n_rep'][year]    = len(districts) - overall['n_dem'][year]
        overall['n_competitive'][year] = sum([1 if abs(d['rv_share'][year] - 0.5) < COMPETITIVE_MARGIN 
                                                else 0 for d in districts])
    for col in ('n_dem', 'n_rep', 'n_competitive'):
        overall['weighted_' + col] = weighted(overall[col].values(), LAMBDA)
    return {'districts': districts, 'overall': overall}

def weighted(arr, lamb):
    """ Apply exponential weighting to an array (last element has highest weighting). """
    exp_sum = 0
    arr_sum = 0
    for idx, val in enumerate(arr):
        exp_sum += exp(lamb*idx)
        arr_sum += exp(lamb*idx) * val
    return arr_sum / exp_sum

if __name__ == '__main__':
    run()