from pprint import pprint
import networkx as nx
from networkx import Graph as nxGraph
import matplotlib.pyplot as plt

class test(object):
    def p(self):
        pprint("test")
        pass


class runner(object):

    G = None

    def __init__(self):
        self.G = nx.Graph()
        pprint("__init__")
        pass

    def run(self):
        G=nx.Graph()
        G.add_node("a")
        G.add_nodes_from(["b","c"])

        G.add_edge(1,2)
        edge = ("d", "e")
        G.add_edge(*edge)
        edge = ("a", "b")
        G.add_edge(*edge)

        print("Nodes of graph: ")
        print(G.nodes())
        print("Edges of graph: ")
        print(G.edges())

        nx.draw(G)
        plt.show() # display
