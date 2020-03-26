import networkx as nx
from pprint import pprint
import random
import matplotlib.pyplot as plt
from base import *

try:
    import seaborn as sns
except UserWarning:
    pass
import matplotlib.animation;

matplotlib.use("TkAgg")


class VisualizationBase(object):
    G = None
    plt = None
    pos = None

    def __init__(self):
        self.G = nx.cubical_graph()
        self.plt = plt
        # code creating G here
        for (u, v, w) in self.G.edges(data=True):
            w['weight'] = random.randint(0, 40)

        self.pos = nx.spring_layout(self.G)
        dbg("len(self.G.edges())", len(self.G.edges()))

        self.idx_weights = range(2, 30, 1)
        # Build plot
        self.fig, self.ax = self.plt.subplots(figsize=(6, 4))

    def update(self, forest = list(list()), step=0):
        self.ax.clear()
        dbg(forest)

        # Background nodes
        nx.draw_networkx_edges(self.G, pos=self.pos, ax=self.ax, edge_color="gray")
        forestNodes = list([item for sublist in (([l[0], l[1]]) for l in forest) for item in sublist])

        dbg("data", forestNodes)
        dbg(set(self.G.nodes()))
        null_nodes = nx.draw_networkx_nodes(self.G, pos=self.pos, nodelist=set(self.G.nodes()) - set(forestNodes),
                                            node_color="white", ax=self.ax)
        if (null_nodes is not None):
            null_nodes.set_edgecolor("black")
            nullNodesIds = set(self.G.nodes()) - set(forestNodes)
            # dbg("nullNodes", nullNodes)
            nx.draw_networkx_labels(self.G, pos=self.pos, labels=dict(zip(nullNodesIds, nullNodesIds)),
                                    font_color="black",
                                    ax=self.ax)

        # Query nodes
        self.idx_colors = sns.cubehelix_palette(len(forestNodes), start=.5, rot=-.75)[::-1]
        query_nodes = nx.draw_networkx_nodes(self.G, pos=self.pos, nodelist=forestNodes,
                                             node_color=self.idx_colors[:len(forestNodes)], ax=self.ax)
        if query_nodes is not None:
            query_nodes.set_edgecolor("white")
        nx.draw_networkx_labels(self.G, pos=self.pos, labels=dict(zip(forestNodes, forestNodes)), font_color="white",
                                ax=self.ax)
        forestNodes2 = list(forestNodes)
        edges = list((l[0], l[1]) for l in forest)
        dbg("edgelist", edges)
        dbg("self.G.edges()", self.G.edges())
        nx.draw_networkx_edges(self.G, pos=self.pos, edgelist=edges, width=self.idx_weights[:len(edges)], ax=self.ax)

        # draw weights
        labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=labels)

        # Scale plot ax
        # ax.set_title("Frame %d:    "%i(1) +  " - ".join(path), fontweight="bold")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("Step #{} ".format(step))

        self.plt.pause(1)
