import networkx as nx
from pprint import pprint

def V(G: nx.Graph):
    return G.nodes()

def E(G: nx.Graph):
    return G.edges()

def dbg(v, **args):
    if len(args>0):
        for i in args:
            pprint(i)
    else:
        pprint(v)
