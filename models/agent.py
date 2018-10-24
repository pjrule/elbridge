import numpy as np


BIDDER_HIDDEN_STATE_SIZE = 128
ALLOCATOR_HIDDEN_STATE_SIZE = 256
MAP_SIZE = 256 # TODO
ALPHA = 0.05 # magnitude of Gaussian noise for GRU initialization

def _sigmoid(x): return 1 / (1 + np.exp(-x))

class Agent:
    """
    An agent has two components: a bidder network and an allocator network.
    For simplicity, all agents' networks are identical. (This could change in the future.)
    Ceteris paribus, fewer weights are better when evolving networks, so we will use one-layer Type II GRUs for now.
    See:
    - https://en.wikipedia.org/wiki/Gated_recurrent_unit
    - https://arxiv.org/abs/1701.05923

    For bidders: state representation + number of coins left -> Type II GRU -> linear mapping to one output (bid amount)
    For allocators: state representation -> Type II GRU -> linear mapping to two outputs (Δx, Δy)
    Note that the following normalizations will always apply:
    - There are 100 coins in any allocation process.
    - All maps are rectangular bounding boxes around the state/region being redistricted. 
      Coordinates are normalized such that the upper left corner is (0,0) and the lower right corner is (box width / box height, 1).

    These networks are small and not trivially parallelizable, so for now we'll just implement them in NumPy.
    """
    def __init__(self, initial_coin):
        """ Set up the neural networks. """
        self.bidder_weights = {
            # GRU
            'U_z': ALPHA * np.random.randn(BIDDER_HIDDEN_STATE_SIZE, BIDDER_HIDDEN_STATE_SIZE),
            'U_r': ALPHA * np.random.randn(BIDDER_HIDDEN_STATE_SIZE, BIDDER_HIDDEN_STATE_SIZE),
            'U_h': ALPHA * np.random.randn(BIDDER_HIDDEN_STATE_SIZE, BIDDER_HIDDEN_STATE_SIZE),
            'W_h': ALPHA * np.random.randn(BIDDER_HIDDEN_STATE_SIZE, MAP_SIZE),
            'b_h': np.zeros(BIDDER_HIDDEN_STATE_SIZE),

            # Linear model on output
            'W_out': ALPHA * np.random.randn(BIDDER_HIDDEN_STATE_SIZE),
            'b_out': 0
        }
        self.allocator_weights = {
            'U_z': ALPHA * np.random.randn(ALLOCATOR_HIDDEN_STATE_SIZE, ALLOCATOR_HIDDEN_STATE_SIZE),
            'U_r': ALPHA * np.random.randn(ALLOCATOR_HIDDEN_STATE_SIZE, ALLOCATOR_HIDDEN_STATE_SIZE),
            'U_h': ALPHA * np.random.randn(ALLOCATOR_HIDDEN_STATE_SIZE, ALLOCATOR_HIDDEN_STATE_SIZE),
            'W_h': ALPHA * np.random.randn(ALLOCATOR_HIDDEN_STATE_SIZE, MAP_SIZE),
            'b_h': np.zeros(ALLOCATOR_HIDDEN_STATE_SIZE),

            'W_out': ALPHA * np.random.randn(ALLOCATOR_HIDDEN_STATE_SIZE, 2),
            'b_out': np.zeros(2)
        }
        self.coin_left = initial_coin
        self.last_bid = 0
        self.last_bidder_h = np.zeros(BIDDER_HIDDEN_STATE_SIZE)
        self.last_allocator_h = np.zeros(ALLOCATOR_HIDDEN_STATE_SIZE)

    def _forward(self, map_state, h, w):
        z = _sigmoid(np.dot(w['U_z'], h))
        r = _sigmoid(np.dot(w['U_r'], h))
        h_new = (1-z)*h + z*np.tanh(np.dot(w['W_h'], map_state) + np.dot(w['U_h'], r*h) + w['b_h'])
        return (w['W_out'] * h_new) + w['b_out'], h_new

    def bid(self, map_state):
        self.last_bid, self.last_bidder_h = self._forward(map_state, self.last_bidder_h, self.bidder_weights)
        return self.last_bid

    def won_auction(self):
        self.coin_left -= self.last_bid # control implies winning, which requires payment

    def loc_delta(self, map_state):
        delta, self.last_allocator_h = self._forward(map_state, self.last_allocator_h, self.allocator_weights)
        return delta
    
