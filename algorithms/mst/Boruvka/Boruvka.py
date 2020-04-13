from VisualizationBase import *

class Boruvka(VisualizationBase):

    def lightestEdgeExcept(s, v, ofG, notInC):
        l = list(filter(lambda x: x[1] not in notInC, list(ofG.edges(v, data=True))))
        if l == []:
            return None
        return min(l,
                   key=lambda x: x[2]['weight'])

    def lightestEdge(s, v, ofG):
        if v not in ofG:
            return None
        return min(list(ofG.edges(v, data=True)), key=lambda x: x[2]['weight'])

    def alg(s, G):
        """
        F ← ( V , ∅)                            // forest of one-vertex trees
        while F has more than one component do
            foreach component C of F do
                S ←∅                            // lightest edges leaving C
                foreach v ∈ V ( C ) do
                    e_v ← the lightest edge { v , w } s.t. w !∈ C
                    S ← S ∪ e_v
                F ← F ∪ (the lightest edge in S)

        :param G:graph
        :return P:path
        """

        # nx.maximum_spanning_tree(G, algorithm='boruvka')
        # I did not think of any better way how to store an information about parent of node
        # and so ended up using parent dict() , thank you nx.maximum_spanning_tree developers for this hint

        s.update()

        F = dict(G.nodes())
        parents = dict(G.nodes())

        resultEdges = list(list())

        # F = dict()
        for k in F.keys():
            F[k] = {k}
            parents[k] = k
        s.update(resultEdges)

        while len(F) > 1:
            for Ck in list(F):
                if Ck not in F:
                    # This may happen because of merging components
                    continue
                C = F[Ck]
                S = set()
                for v in C:
                    """e_v ← the lightest edge { v , w } s.t. w !∈ C"""
                    edgeOfV = s.lightestEdgeExcept(v, G, C)
                    if edgeOfV is None:
                        continue
                    # note: classical switching order od nodes in edge
                    if(edgeOfV[1] == v):
                        edgeOfV[0],edgeOfV[1],edgeOfV[2] = edgeOfV[1],edgeOfV[0],edgeOfV[2]
                    edgeOfV_noWeight = (edgeOfV[0], edgeOfV[1])
                    S.add(edgeOfV_noWeight)
                edgeOfVOfS = s.lightestEdge(v, G.edge_subgraph(edges=S))
                # if edgeOfVOfS is None:
                #     """ TODO remove? """
                #     continue
                """F ← F ∪ (the lightest edge in S)"""

                innerVOfE = edgeOfVOfS[0]
                outerVOfE = edgeOfVOfS[1]
                parentOfOuterVOfE = parents[outerVOfE]
                parentOfInnerVOfE = parents[innerVOfE]
                resultEdges.append(edgeOfVOfS)
                s.update(resultEdges)
                parentSet = F[parentOfOuterVOfE]
                F[parentOfInnerVOfE] = set(parentSet.union(F[parentOfInnerVOfE]))
                """ Now it is necessary to update information about parents """
                # TODO improve > O(n)
                for c in F[parentOfOuterVOfE]:
                    parents[c] = parentOfInnerVOfE
                F.pop(parentOfOuterVOfE)

        dbg("F", F)
