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
        for v in V(G): MAKE_SET(v)
        "sort edges according to w into a non-decreasing sequence"
        sortedPairs = sorted(G.edges().data(), key=lambda e: e[2]['weight'])
        for e in sortedPairs:
            if FIND_SET( e.u ) != FIND_SET( e.v ):
                F = F.union(e)
                UNION(u, v)


#
