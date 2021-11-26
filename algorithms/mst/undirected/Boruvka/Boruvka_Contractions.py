from VisualizationBase import *
from algorithms.mst.undirected.Boruvka.Boruvka import Boruvka


class Boruvka_Contractions(Boruvka):
    visualizeTrees = False
    visualizeComponents = True

    def alg(s, G):
        """
        while G has more than one vertex do
            foreach vertex v ∈ V ( G ) do
                e v ← the lightest edge incident to v
                add e v to the MST
            contract all edges e v
            eliminate parallel edges and loops

        :param G:graph
        :return P:path
        """

        s.update()
        F = dict(G.nodes())
        parents = dict(G.nodes())
        resultEdges = list(list())

        # TODO ...

        dbg("resultEdges", resultEdges)
        return resultEdges
