from VisualizationBase import *


class EdmondsBranching(VisualizationBase):
    """
    Chu-Liu (1965), Edmonds (1967)
    """
    visualizeTrees = False
    visualizeComponents = True

    def reducedWeight(self, D, w):
        """
        foreach e in E(D) do
            w'(e) = w(e) − w(e_v)

        :param D:
        :return:
        """
        return

    def isAnArborescence(self, D):
        # TODO verification
        # 1. not formal
        # 2. formal
        return True


    def alg(s, D, r, w):
        # TODO decide how to deal with storing weights w
        # - own structure
        # - or duplication of D
        # - or else depending on alg:
        # -- line 9: contractions
        """
        In: D, r, w

        F ← ∅
        foreach v ∈ V(D) \ {r} do
            e_v ← edge of minimum weight entering v
            add e_v to F
            foreach e = (u,v) ∈ E (D) do w' (e) ← w(e) − w(e_v)
        if (V,F) is an arborescence then
            return T = (V,F)
        else
            contract directed cycles in (V,F)
            let D' be the resulting graph
            T' ← EdmondsBranching(D',r,w')
            T ← T' with cycles expanded back (minus one edge)
            return T

        :param G:graph
        :return T:path
        """

        resultEdges = list(list())


        F = list(list())

        for v in (V(D) - {r}):
            pass

        return resultEdges
