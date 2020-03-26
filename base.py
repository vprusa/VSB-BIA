import networkx as nx
from pprint import pprint
from pprint import pformat

def V(G: nx.Graph):
    return G.nodes()

def E(G: nx.Graph):
    return G.edges()

def dbg(v, *args):
    if len(args) > 0:
        # if len(args) == 1:
        #     pprint(v + ": " + pformat(args[0]))
        #     pass
        # else:
        print(v + ": ")
        for i in args:
            pprint(i,  width=200)
    else:
        pprint(v)


