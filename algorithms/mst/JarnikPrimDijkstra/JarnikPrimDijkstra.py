from base import *
from VisualizationBase import *

from disjoint_set import DisjointSet
import random


class JarnikPrimDijkstra(VisualizationBase):
    ds = DisjointSet()

    def someV(self, of):
        random.choice(of)

    def lightestEdge(self, of, leaving):
        # TODO
        pass

    def alg(self, G):
        """
        choose some s ∈ V(G)
        T ← ({s}, ∅)
        for n − 1-times do
            e ← the lightest edge of G leaving T
            T ← T ∪ {e}
        """

        s = self.someV(of=V(G))
        dbg("some s", s)
        T = list(list(s))
        self.update(T)
        for i in range(len(self.G.edges()) - 1):
            dbg("i: ", i)
            e = self.lightestEdge(of=G, leaving=T)

#
