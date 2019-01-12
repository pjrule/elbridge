"""
This script executes rollouts for a state.* and
samples a specified number of frames, which are saved in *.npz
format. This is intended for generation of input data for autoencoder/PCA
training.
"""

import os
import click
import pickle
from multiprocessing import Manager, Queue

FORMAT = '%(asctime)-15s %(level)s %(message)s' # for logging
QUEUE_TIMEOUT = 120

@click.command()
@click.argument('state_pkl', help='The .pkl file containing the state object.')
@click.argument('out_dir', help='The output directory for the PNG frames.')
@click.argument('n_frames', help='The number of frames to sample.')
@click.option('--max-iterations', default=20000, help='The maximum number of '
              'iterations allowed to allocate a map.', type=int)
@click.option('--reset-iterations', default=150, help='The maximum number of '
              'iterations without allocation before resetting the working '
              'district.', type=int)
@click.option('--max-delta', default=0.01, help='The maximum distance (in percentage'
              'of state population) traveled by an agent in one iteration.', type=float)
@click.option('--max-map-frames', default=200, help='The maximum number of '
              'frames to sample from a map.', type=int)
@click.option('--sampling-prob', default=0.05, help='The probability of sampling'
              ' a frame during a single iteration of allocation.', type=float)
@click.option('--n-workers', default=2, help='The number of workers used for map'
              ' generation.', type=int)
# TODO: S3 support
@click.option('--output-bucket', help='If specified, frames are saved to the given '
             'Google Cloud Storage bucket.')
def sample(state_pkl, out_dir, n_frames, max_iterations, reset_iterations,
           max_delta, max_map_frames, sampling_prob, n_workers):
    state = pickle.load(open(state_pkl, 'rb'))
    state.reset()
    q = Manager().Queue()
    params = [(state, q) for _ in range(n_maps)]
    empty = False
    with Pool(n_workers) as p:
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
