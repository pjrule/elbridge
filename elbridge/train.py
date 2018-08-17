import ray
import torch
import click

"""
Elbridge's information flow:
       Maximum-resolution state information (raw precinct-level map data)
    -> Statically computed mappings from max-res state information to fixed-size state information (very quick to update!)
    -> an autoencoder mapping from max-res state information to a smaller 1D vector, trained dynamically as agents train (>1M paramters is OK)
    -> a set of agents with various reward functions (Type II GRU networks, 2 per agent; goal of ~100,000 parameters per agent total)
        (note that each agent consists of a bidder and an allocator)

The following state information must be maintained for each step in the allocation:
- For each agent:
    - Previous hidden state(s)
- Coins left for each agent 
- For each map:
    - Exact precinct allocations
    - Fixed-size state information
"""
#STATE_SIZE = ?

import click
from .states import Pennsylvania
from .objectives import *
from sys import exit

@click.command()
@click.argument('state')
@click.argument('vtd_map_file')
@click.argument('vtd_elections_file')
@click.argument('vtd_demographics_file')
def train(state, vtd_map_file, vtd_elections_file, vtd_demographics_file):
    if state == "PA":
        state_map = Pennsylvania(vtd_map_file, vtd_elections_file, vtd_demographics_file)
        optimizers = []
    else:
        print("State not recognized.")
        exit(1)

    
    