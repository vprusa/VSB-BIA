from VisualizationBase import *


class EdmondsBranching(VisualizationBase):
    """
    Chu-Liu (1965), Edmonds (1967)
    """
    visualizeTrees = False
    visualizeComponents = True

    """
    Direction = order of vertices in edges 
    .virtenv/bin/ipython3 -i -c "%run ./run.py --l=mst.directed --a=EdmondsBranching --g cubical_graph --kr 10"
    .virtenv/bin/ipython3 -i -c "%run ./run.py --l=mst.directed --a=EdmondsBranching --g cubical_graph --kr 10 \
    --d \"[(0,1,{'weight': 2}),(1,2,{'weight': 6}),(0,3,{'weight': 5}),(3,1,{'weight': 1}),(1,4,{'weight': 3}),
           (4,3,{'weight': 4}),(4,5,{'weight': 7}),(1,5,{'weight': 8}),(2,5,{'weight': 9})]\""
    
    
    Edges: [(0,1,{2}),(1,2,{6}), (0,3,{5}),(3,1,{1}),(1,4,{3}),(4,3,{4}),(4,5,{7}), (1,5,{8}),(2,5,{9})]

    
    Some ASCII art graph 
    
    0 ---2--> 1 ---6--> 2
    |      7\ | \       |
    5    1/   3  \_8_   9
    |  _/     |      \  |
    v /       V      _V V
    3 <--4--- 4 ---7--> 5
    """


    def reducedWeight(s, D, w):
        """
        foreach e in E(D) do
            w'(e) = w(e) − w(e_v)

        :param D:
        :return:
        """
        return

    def isAnArborescence(s, D):
        # TODO verification
        # 1. not formal
        # 2. formal
        return True

    def alg(s,D):
        # s.algRec(D, r, w)
        pass

    def algRec(s, D, r, w):
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
