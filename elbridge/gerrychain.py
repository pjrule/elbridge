""" Integrations between GerryChain and Elbridge. """
from typing import Dict
from elbridge import Graph as ElbridgeGraph
from gerrychain import Graph as GerryGraph


def to_gerrychain_graph(el_graph: ElbridgeGraph, cd_col: str) -> GerryGraph:
    base_graph = GerryGraph.from_geodataframe(el_graph.df)


def to_gerrychain_json(el_graph: ElbridgeGraph, cd_col: str) -> Dict:
    pass
