from run import runner
from pprint import pprint
import networkx as nx
import matplotlib.pyplot as plt

r = runner()
#V=()
from algorithms.mst.Kruskal import Kruskal as Kruskal
#source(findFile("algorithms/mst/","Kruskal.py"))
#krusla
w = Kruskal()
#r.G = nx.complete_graph(4)
#r.G = nx.random_regular_graph(d=3,n=4)
r.G = nx.tetrahedral_graph()
#w.alg(r.G)

import random
#code creating G here
for (u,v,w) in r.G.edges(data=True):
    w['weight'] = random.randint(0,10)

print(r.G)

#pos=nx.get_edge_attributes(r.G,'weight')
pos=nx.spring_layout(r.G)
labels = nx.get_edge_attributes(r.G,'weight')
nx.draw_networkx_edge_labels(r.G,pos, edge_labels=labels)
nx.draw(r.G)
plt.show() # display

from algorithms.mst.Kruskal import Kruskal

k=Kruskal()
k.alg(r.G)
#nx.draw(r.G)
#r.G.edges()

#

from run import runner
from pprint import pprint
import networkx as nx
import matplotlib.pyplot as plt

r = runner()
#V=()
from algorithms.mst.Kruskal import Kruskal as Kruskal
#source(findFile("algorithms/mst/","Kruskal.py"))
#krusla
w = Kruskal()
#r.G = nx.complete_graph(4)
#r.G = nx.random_regular_graph(d=3,n=4)
r.G = nx.tetrahedral_graph()
#w.alg(r.G)

import random
#code creating G here
for (u,v,w) in r.G.edges(data=True):
    w['weight'] = random.randint(0,10)

print(r.G)

#pos=nx.get_edge_attributes(r.G,'weight')
pos=nx.spring_layout(r.G)
labels = nx.get_edge_attributes(r.G,'weight')
nx.draw_networkx_edge_labels(r.G,pos, edge_labels=labels)
nx.draw(r.G)
#plt.show() # display

#from algorithms.mst.Kruskal import Kruskal
from base import *

def MAKE_SET(v):
    pass

def FIND_SET(e):
    pass

def UNION(F, e):
    pass

class Kruskal(object):
    def alg(self, G):
        F = ( V , ())
#        for v in V(G): MAKE_SET(v)
        "sort edges according to w into a non-decreasing sequence"
        sortedPairs = sorted(G.edges().data(), key=lambda e: e[2]['weight'])
#        for e in sortedPairs:
#            if FIND_SET( e.u ) != FIND_SET( e.v ):
#                F = F.union(e)
#                UNION(u, v)
#

k=Kruskal()
#k.alg(r.G)
#nx.draw(r.G)
#r.G.edges().data()[
sorted(r.G.edges().data(), key=lambda e: e[2]['weight'])
