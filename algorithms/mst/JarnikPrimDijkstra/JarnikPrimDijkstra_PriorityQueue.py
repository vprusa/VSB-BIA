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

    def some(self, of):
        dbg("of", list(of))
        return random.choice(list(of))

    def INSERT(self, Q, v):
        Q
        pass

    def alg(self, G):
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

        key = dict()
        pred = dict()
        s = self.some(V(G))
        self.update(list(list()))

        Q = []
        for v in V(G):
            dbg("v", v)
            key[v] = sys.maxsize
            pred[v] = 0
            #self.INSERT(Q,v)
            # Q.append(v)
            heapq.heappush(Q, (sys.maxsize, v))

        key[s] = 0
        dbg("s",s)
        dbg("key",key[s])
        dbg("Q", Q)

        while(len(Q) > 0):
            u = heapq.heappop(Q)

            dbg("Q",Q)
            dbg("min u",u)
            # foreach edge { u , v } s.t. v is in Q do
            # TODO fix performance, refactor
            Qv = list(i[1] for i in Q)
            for e in list(filter(lambda e: (e[1] in Qv and e[0] == u[1]), list(G.edges().data()))):
                w = e[2]['weight']
                dbg("e",e)
                dbg("w",w)
                dbg("Q",Q)
                dbg("w",w)
                dbg("e[1]",e[1])
                dbg("key[e[1]]", key[e[1]])
                dbg("key", key)
                dbg("pred", pred)

                u = e[0]
                v = e[1]
                if w < key[v]:
                    dbg("Decreasing")
                    #self.DECREASE_KEY(Q,v,w)
                    #heapq.heapreplace(Q,(w,v))
                    key[v] = w
                    # pq.push(make_pair(key[v], v));
                    heapq.heappush(Q, (w, v))
                    pred[v] = u
        dbg("key", key)
        dbg("pred", pred)
        dbg("Q", Q)
        # self.update()

        # dbg("G", G.edges())

    def EXTRACT_MIN(self, Q):
        #Q.remove(min(Q))
        pass

    def DECREASE_KEY(self, Q, v, w):
        Q[v]
        pass

