import os
import states
import pickle
import click
import logging
import numpy as np
from queue import Empty
from random import random
from hashlib import sha512
from math   import pi, exp
from copy   import deepcopy
from datetime import datetime
from multiprocessing import Manager, Pool

N_WORKERS = 1             # number of maps to generate concurrently
ITERATIONS = 15000        # quit if this number of iterations is reached
RESET_ITERATIONS = 150    # reset the current district if no progress is made after this number of iterations
MAX_DELTA = 0.1           # in proportion of population (0.1 = go 10% of the state population away in one step)
FORMAT = '%(asctime)-15s %(level)s %(message)s' # for logging
QUEUE_TIMEOUT = 120
P_RENDER = 0.1

@click.command()
@click.argument('pkl')
@click.argument('density')
@click.argument('geo')
@click.argument('n_maps', type=int)
@click.argument('out_dir')
def run(pkl, density, geo, n_maps, out_dir):
    logging.basicConfig(format=FORMAT, level=logging.WARNING)
    state = pickle.load(open(pkl, 'rb'))
    state.load_density_mapping(density)
    state.load_geo_mapping(geo)
    state.geo_weights = state.geo_weights.T
    # https://stackoverflow.com/a/9928191
    q = Manager().Queue()
    params = [(state, q) for _ in range(n_maps)]
    empty = False
    with Pool(N_WORKERS) as p:
        p.starmap_async(generate_map, params, error_callback=log_error)
        p.close()
        for idx in range(n_maps):
            try:
                frames = q.get(timeout=QUEUE_TIMEOUT)
                frames_hash = sha512(''.join(frames.keys()).encode('utf-8')).hexdigest()
                np.savez_compressed(os.path.join(out_dir, frames_hash[:12]), kwds=frames)
            except Empty:
                empty = True
                break
        if not empty:
            p.join()

def log_error(e):
    logging.error(e, exc_info=True)

def generate_map(state, queue):
    state = deepcopy(state)
    state.reset()
    last_pop = 0
    iters_since_pop_change = 0
    frames = {}
    for _ in range(ITERATIONS):
        if random() < P_RENDER:
            frame      = state.render()
            frame_hash = sha512(frame.tostring() +
                                datetime.now().isoformat().encode('utf-8')).hexdigest()
            frames[frame_hash] = frame.astype(np.float16) # reduced precision
        if state.done:
            queue.put(frames)
            return

        alloc = np.zeros(len(state.df))
        for district_idx, districts in enumerate(state.vtd_by_district):
            for d in districts:
                alloc[d] = district_idx

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


if __name__ == '__main__':
    run()