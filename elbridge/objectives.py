from agent import Agent
from map import Map

class Optimizer:
    def fitness(self, map):
        """
        *** This method must be implemented by child optimizers. ***
        Compute the fitness of a map on some absolute scale.
        The scale doesn't matter, as they will later be normalized.
        We're only concerned with the distribution of fitness--that is, which maps are more fit and which are less fit.
        """
        return 0 # child classes implement

class CompetitionOptimizer(Optimizer):
    def fitness(self, map):
        pass

class VRAOptimizer(Optimizer):
    def fitness(self, map):
        pass

class CompactnessOptimizer(Optimizer):
    def fitness(self, map):
        pass