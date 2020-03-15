from run import runner
from pprint import pprint
import networkx as nx
import matplotlib.pyplot as plt

r = runner()
from algorithms.mst.Kruskal import Kruskal as Kruskal

w = Kruskal()
r.G = nx.complete_graph(4)
# r.G = nx.random_regular_graph(d=3,n=4)
# r.G = nx.complete_graph(4)
# r.G = nx.MultiGraph()
# w.alg(r.G)

import random

# code creating G here
for (u, v, w) in r.G.edges(data=True):
    w['weight'] = random.randint(0, 10)

# import math
# posWeights = nx.get_edge_attributes(r.G,'weight')
# posWeights=nx.spring_layout(r.G, k=5/math.sqrt(r.G.order()))
posWeights = nx.spring_layout(r.G)
labels_edges = nx.get_edge_attributes(r.G, 'weight')

nx.draw_networkx_edge_labels(r.G, posWeights, edge_labels=labels_edges)

# from algorithms.mst.Kruskal import Kruskal
from base import *


class Kruskal(object):
    from disjoint_set import DisjointSet

    ds = DisjointSet()

    def MAKE_SET(self, v):
        pprint('MAKE_SET')
        pprint(v)
        self.ds.find(v)
        pass

    def FIND_SET(self, e):
        pprint("FIND_SET")
        pprint(e)
        if self.ds.__contains__(e):
            pprint("contains")
            res = self.ds.find(v)
            return res
        else:
            return None

    def UNION(self, F, e):
        pprint("UNION")
        pprint(self.ds.union(F, e))
        pass

    def alg(self, G):
        F = (V, ())
        pprint(V(G))
        for v in V(G):
            self.MAKE_SET(v)
            pprint(v)
        pprint("list")
        pprint(list(self.ds))

        pprint("sort edges according to w into a non-decreasing sequence")
        sortedPairs = sorted(G.edges().data(), key=lambda e: -e[2]['weight'])
        pprint(sortedPairs)
        for e in sortedPairs:
            pprint(e)
            if self.FIND_SET(e[0]) != self.FIND_SET(e[1]):
                self.union(F, e)
        pprint(F)
        pprint("list")
        pprint(list(self.ds))


#

k = Kruskal()
k.alg(r.G)

# nx.draw(r.G, with_labels = True, font_color='red')
nx.draw(r.G, with_labels=True)
# sorted(r.G.edges().data(), key=lambda e: e[2]['weight'])
plt.show()

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import matplotlib.animation

# Create Graph
np.random.seed(2)
G = nx.cubical_graph()
G = nx.relabel_nodes(G, {0: "O", 1: "X", 2: "XZ", 3: "Z", 4: "Y", 5: "YZ", 6: "XYZ", 7: "XY"})
pos = nx.spring_layout(G)

# Sequence of letters
sequence_of_letters = "".join(['X', 'Y', 'Z', 'Y', 'Y', 'Z'])
idx_colors = sns.cubehelix_palette(5, start=.5, rot=-.75)[::-1]
idx_weights = [3, 2, 1]

# Build plot
fig, ax = plt.subplots(figsize=(6, 4))


def update(num):
    ax.clear()
    i = num // 3
    j = num % 3 + 1
    triad = sequence_of_letters[i:i + 3]
    path = ["O"] + ["".join(sorted(set(triad[:k + 1]))) for k in range(j)]

    # Background nodes
    nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color="gray")
    null_nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=set(G.nodes()) - set(path), node_color="white", ax=ax)
    null_nodes.set_edgecolor("black")

    # Query nodes
    query_nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=path, node_color=idx_colors[:len(path)], ax=ax)
    query_nodes.set_edgecolor("white")
    nx.draw_networkx_labels(G, pos=pos, labels=dict(zip(path, path)), font_color="white", ax=ax)
    edgelist = [path[k:k + 2] for k in range(len(path) - 1)]
    nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, width=idx_weights[:len(path)], ax=ax)

    # Scale plot ax
    ax.set_title("Frame %d:    " % (num + 1) + " - ".join(path), fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


ani = matplotlib.animation.FuncAnimation(fig, update, frames=6, interval=1000, repeat=True)
plt.show()


