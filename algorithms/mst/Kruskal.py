from base import *

from disjoint_set import DisjointSet

class Kruskal(object):
    ds = DisjointSet()

    def MAKE_SET(self, v):
        dbg('MAKE_SET', v)
        self.ds.union(v, None)
        pass

    def FIND_SET(self, e):
        dbg("FIND_SET", e)
        res = self.ds.find(e)
        dbg(res)
        return res

    def UNION(self, F, e):
        dbg("UNION")
        dbg(self.ds.union(F, e))

    def alg(self, G):
        F = (V, ())
        for v in V(G):
            self.MAKE_SET(v)
        dbg("sort edges according to w into a non-decreasing sequence")
        sortedPairs = sorted(G.edges().data(), key=lambda e: -e[2]['weight'])
        for e in sortedPairs:
            if self.FIND_SET(e[0]) != self.FIND_SET(e[1]):
                self.union(F, e)
        dbg(list(self.ds))

#
