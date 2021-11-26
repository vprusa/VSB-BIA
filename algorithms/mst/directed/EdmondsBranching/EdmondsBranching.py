from VisualizationBase import *


class EdmondsBranching(VisualizationBase):
    """
    Chu-Liu (1965), Edmonds (1967)
    """
    visualizeTrees = False
    visualizeComponents = True
    isDirectedG = True

    """
    Direction = order of vertices in edges 
    .virtenv/bin/ipython3 -i -c "%run ./run.py --l=mst.directed --a=EdmondsBranching --g cubical_graph --kr 10"
    .virtenv/bin/ipython3 -i -c "%run ./run.py --l=mst.directed --a=EdmondsBranching --g cubical_graph --kr 10 \
    --d \"[(0,1,{'weight': 2}),(1,2,{'weight': 6}),(0,3,{'weight': 5}),(3,1,{'weight': 1}),(1,4,{'weight': 3}),
           (4,3,{'weight': 4}),(4,5,{'weight': 7}),(1,5,{'weight': 8}),(2,5,{'weight': 9})]\""
    
    
    Edges: [(0,1,{2}),(1,2,{6}), (0,3,{5}),(3,1,{1}),(1,4,{3}),(4,3,{4}),(4,5,{7}), (1,5,{8}),(2,5,{9})]
    
    Some ASCII art graph (0=r) 
    
    (0)----2--->(1)----6--->(2)
     |        7\ | \__       |
     5    _1_/   3    \_8_   9
     |  _/       |        \  |
     v /         V        _V V
    (3)<---4----(4)----7--->(5)
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

    def edgeOfMinWeight(s, D, entering):
        l = list(D.edges(entering, data=True))
        if len(l) == 0:
            return None
        e = min(l, key=lambda x: x[2]['weight'])
        return e

    def alg(s, D):
        resultEdges = list(list())
        s.update()
        T = s.algRec(D, r=D.nodes(0), w=D.edges(data=True))
        return T

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
        nx.minimum_spanning_arborescence()

        F = list(list())
        # TODO fix
        w_0 = list(list())

        for v in V(D):
            if v == r:
                dbg("skip root", r)
                continue
            e_v = s.edgeOfMinWeight(D, v)
            if e_v is None:
                # no other edges
                continue
            F.append(e_v)
            for e in E(D):
                # w'(e) = w(e) - w(e_v)
                # w_0 = w(e)
                w_0 = list(filter(lambda x: (x[0] == e[0] and x[1] == e[1]), list(w)))
                # w_0_w = w(e)
                w_0_w = w_0
                # w_0_e_v = w(e_v)
                w_0_e_v = list(filter(lambda x: (x[0] == e_v[0] and x[1] == e_v[1]), list(w)))
                # w_0 = w(e) - w(e_v)
                # w_0 = w_0_w[0][2]['weight'] - w_0_e_v[0][2]['weight']
                w_0_w[0][2]['weight'] = w_0_w[0][2]['weight'] - w_0_e_v[0][2]['weight']
                pprint(w)
                pass
            pass
        if s.isAnArborescence(F):
            T = (V, F)
            return T
            pass
        else:
            D_0 = s.contractCyclesIn(V, F)
            T_0 = s.alg(D_0, r, w_0)
            T = s.expandCycles(T_0)
            return T
            pass

        pass
        # s.contractCyclesIn(V,F)
        # return resultEdges

    def edgeOf(self, weight, entering):
        """
        :return e:'edge of minimum weight entering v'
        """
        e = weight.in_edges(entering)
        return e

    def contractCyclesIn(self, V, F):
        pass

    def expandCycles(self, T):
        # minus one edge
        pass
