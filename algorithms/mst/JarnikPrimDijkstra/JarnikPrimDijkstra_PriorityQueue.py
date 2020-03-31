from queue import Queue

from algorithms.mst.JarnikPrimDijkstra.JarnikPrimDijkstra import JarnikPrimDijkstra
from base import *
from VisualizationBase import *
import heapq
import operator
import random
import sys

'''
Helpful links:
# https://www.geeksforgeeks.org/prims-algorithm-using-priority_queue-stl/
# https://books.google.cz/books?id=JBXDc83jRBwC&pg=PA56&lpg=PA56&dq=DECREASE-KEY(+Q,+v+,+w+(+u+,+v+)+)&source=bl&ots=IGZ2Lc2Dll&sig=ACfU3U2D-HOP9DAiZ6sK9XBURFu9hZ8tmw&hl=en&sa=X&ved=2ahUKEwjBk722icPoAhVITsAKHbLDAUYQ6AEwAHoECAUQAQ#v=onepage&q=DECREASE-KEY(%20Q%2C%20v%20%2C%20w%20(%20u%20%2C%20v%20)%20)&f=false
'''
class JarnikPrimDijkstra_PriorityQueue(JarnikPrimDijkstra):

    Q = []
    key = dict()
    pred = dict()

    def someV(s, of):
        return random.choice(list(of))

    def INSERT(s, Q, v):
        heapq.heappush(Q, (sys.maxsize, v))

    def EXTRACT_MIN(s, Q):
        return heapq.heappop(Q)

    def DECREASE_KEY(s, v, w):
        s.key[v] = w
        s.Q = [q for q in s.Q if q[1] != v]
        heapq.heappush(s.Q, (w, v))

    def alg(s, G):
        """
        INIT( Q )
        foreach v ∈ V ( G ) do
            key[v ] ← ∞ ; pred[v ] ← Nil
            INSERT( Q,v )
        key[s] ← 0
        while (not IS-EMPTY( Q ) ) do
            u ← EXTRACT-MIN( Q )
            foreach edge { u , v } s.t. v is in Q do
                if w ( u , v ) < key[v ] then
                    DECREASE-KEY( Q, v , w ( u , v ) )
                    pred[v] ← u

        :param G:graph
        :return P:path
        """

        s.key = dict()
        s.pred = dict()

        some = s.someV(V(G))
        s.update(list(list()))

        s.Q = []
        for v in V(G):
            if(v == some): continue
            s.key[v] = sys.maxsize
            # TODO s.update() input params (s.pred containing nodes instead edges)
            s.pred[v] = (v,v, {'weight': 0})
            s.INSERT(s.Q,v)

        s.pred[some] = (some, some, {'weight': 0})
        heapq.heappush(s.Q, (0, some))
        s.key[some] = 0
        s.update(list(s.pred.values()))

        while(len(s.Q) > 0):
            ut = s.EXTRACT_MIN(s.Q)
            u = ut[1]
            # foreach edge { u , v } s.t. v is in Q do
            Qv = list(i[1] for i in s.Q)
            Gd = G.edges().data()
            # TODO try use G.neighbours(u)
            Qe = list(filter(lambda e: (e[0] == u and e[1] in Qv) or (e[1] == u and e[0] in Qv), list(Gd)))
            for e in Qe:
                # switch because edge may not always be in lexicographical order
                if(e[1] == u):
                    e = (e[1],e[0],e[2])
                w = e[2]['weight']
                v = e[1]
                keyV = s.key[v]
                if w < keyV:
                    s.DECREASE_KEY(v,w)
                    s.pred[v] = e
                    s.update(list(s.pred.values()))
        dbg("s.pred", s.pred)
        dbg("Q", s.Q)

