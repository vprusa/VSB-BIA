import networkx as nx
from pprint import pprint
import random
import matplotlib.pyplot as plt
from base import *
import ast

try:
    import seaborn as sns
except UserWarning:
    pass
import matplotlib.animation

matplotlib.use("TkAgg")


class VisualizationBase(object):
    frameNo = 0

    frameTimeout = 1
    nxgraphType = "cubical_graph"
    nxgraphOptions = None
    graphData = None
    G = None
    plt = None
    pos = None

    def __init__(s, nxgraphType=None, nxgraphOptions=None, graphData=None, isDirected=False, frameTimeout=1):
        s.frameTimeout = frameTimeout
        s.nxgraphType = nxgraphType
        s.nxgraphOptions = nxgraphOptions
        s.graphData = graphData
        # nx.cubical_graph()
        if graphData is None:
            if nxgraphOptions is None:
                s.G = getattr(nx, s.nxgraphType)()
            else:
                s.G = getattr(nx, s.nxgraphType)(s.nxgraphOptions)
            if isDirected:
                s.G=nx.to_directed(s.G) # if already directed creates deep copy
            for (u, v, w) in s.G.edges(data=True):
                w['weight'] = random.randint(0, 40)
        else:
            if isDirected:
                s.G = nx.DiGraph()
                s.G.add_edges_from(ast.literal_eval(s.graphData))
            else:
                s.G = nx.from_edgelist(ast.literal_eval(s.graphData))

        # s.G = nx.from_edgelist(list(
        #     [(0, 1, {'weight': 15}), (0, 3, {'weight': 34}), (0, 4, {'weight': 25}), (1, 2, {'weight': 5}),
        #      (1, 7, {'weight': 23}), (2, 3, {'weight': 33}), (2, 6, {'weight': 29}), (3, 5, {'weight': 13}),
        #      (4, 5, {'weight': 5}), (4, 7, {'weight': 20}), (5, 6, {'weight': 38}), (6, 7, {'weight': 3})]
        #     ))

        s.plt = plt
        s.pos = nx.spring_layout(s.G)
        s.idx_weights = range(2, 30, 1)

        # Build plot
        s.fig, s.ax = s.plt.subplots(figsize=(6, 4))

    def alg(self, G):
        pass
        # alg(self, G) should have algorithm-required inputs as parameters

    def update(s, edges=None):
        if edges is None:
            edges = list(list())
        s.ax.clear()

        # Background nodes
        pprint(s.G.edges())
        nx.draw_networkx_edges(s.G, pos=s.pos, ax=s.ax, edge_color="gray",
                               # arrowstyle='->',
                               arrowstyle= '-|>',
                               arrowsize=10
                               )
        forestNodes = list([item for sublist in (([l[0], l[1]]) for l in edges) for item in sublist])

        # dbg("forestNodes", forestNodes)
        forestNodes = list(filter(None, forestNodes))
        # dbg("forestNodes -!None", forestNodes)
        # dbg(set(self.G.nodes()))
        null_nodes = nx.draw_networkx_nodes(s.G, pos=s.pos, nodelist=set(s.G.nodes()) - set(forestNodes),
                                            node_color="white", ax=s.ax)
        if (null_nodes is not None):
            null_nodes.set_edgecolor("black")
            nullNodesIds = set(s.G.nodes()) - set(forestNodes)
            # dbg("nullNodes", nullNodes)
            nx.draw_networkx_labels(s.G, pos=s.pos, labels=dict(zip(nullNodesIds, nullNodesIds)),
                                    font_color="black",
                                    ax=s.ax)

        # Query nodes
        s.idx_colors = sns.cubehelix_palette(len(forestNodes), start=.5, rot=-.75)[::-1]
        query_nodes = nx.draw_networkx_nodes(s.G, pos=s.pos, nodelist=forestNodes,
                                             node_color=s.idx_colors[:len(forestNodes)], ax=s.ax)
        if query_nodes is not None:
            query_nodes.set_edgecolor("white")
        nx.draw_networkx_labels(s.G, pos=s.pos, labels=dict(zip(forestNodes, forestNodes)), font_color="white", ax=s.ax)

        edges = list((l[0], l[1]) for l in edges)
        # TODO refactor edge-has-none check -> default behaviour if forest is single node
        # hasNone = edges is not None and (len(edges[0]) != 0 and len(edges[0][0]) != 0) and list(i if (i[0] is not None and i[1] is not None) else None for i in edges)[0] is not None
        # hasEdge = (len(edges) > 0 and len(edges[0]) > 0)
        # if hasEdge:
        # dbg("edgelist", edges)
        # dbg("self.G.edges()", self.G.edges())
        dbg("edges", edges)
        nx.draw_networkx_edges(s.G, pos=s.pos, edgelist=edges, width=s.idx_weights[:len(edges)]
                               # ,
                               # ax=s.ax, arrowstyle='->',
                               # arrowsize=10
                               )

        # draw weights
        labels = nx.get_edge_attributes(s.G, 'weight')
        nx.draw_networkx_edge_labels(s.G, s.pos, edge_labels=labels)

        # Scale plot ax
        s.ax.set_xticks([])
        s.ax.set_yticks([])
        s.ax.set_title("Step #{} ".format(VisualizationBase.frameNo))

        # self.plt.pause(5)
        s.plt.pause(s.frameTimeout)
        # self.plt.pause(3)
        VisualizationBase.frameNo += 1
