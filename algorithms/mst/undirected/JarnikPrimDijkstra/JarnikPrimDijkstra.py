from base import *
from VisualizationBase import *

import random


class JarnikPrimDijkstra(VisualizationBase):

    def someV(self, of):
        dbg("of", list(of))
        return random.choice(list(of))


    def lightestEdge(self, of, leaving):
        dbg("of", set(of.data()))
        dbg("leaving", leaving)
        # TODO
        pass

    def alg(self, G):
        """
        choose some s ∈ V(G)
        T ← ({s}, ∅)
        for n − 1-times do
            e ← the lightest edge of G leaving T
            T ← T ∪ {e}

        :param G:graph
        :return P:path
        """
        dbg("G", G.edges())

        s = self.someV(of=V(G))
        dbg("some s", s)
        # TODO refactor for general graphs, reflective edge is terrible idea
        T = list(list([[s,s]]))
        self.update(T)
        for i in range(len(self.G.edges()) - 1):
            dbg("i: ", i)
            dbg("T", T)
            e = self.lightestEdge(of=G, leaving=T)
            T.append(list(e))
        dbg("T: ", T)
        T
