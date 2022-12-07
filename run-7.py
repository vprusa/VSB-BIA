try:
    import seaborn as sns
    pass
except UserWarning:
    pass
import matplotlib.animation ; matplotlib.use("TkAgg")

import networkx as nx
from pprint import pprint
import random
import matplotlib.pyplot as plt
import numpy as np
from base import *
import ast

try:
    import seaborn as sns
except UserWarning:
    pass
import matplotlib.animation
import random

matplotlib.use("TkAgg")

class Vis2D(object):
    frameNo = 0
    min_individual_price = 0

    frameTimeout = 0.1
    # nxgraphType = "cubical_graph"
    nxgraphOptions = None
    graphData = None
    G = None
    D = None
    plt = None
    layout = None


    distances = (0,500)

    start_node = 0

    nxgraphType = "complete_graph"

    # NP = 6  # population cnt
    # DC = 6  # In TSP, it will be a number of cities
    # NP = 3  # population cnt
    # DC = 4  # In TSP, it will be a number of cities
    # GC = 5  # generation cnt
    # NP = 10  # population cnt
    # DC = 10  # In TSP, it will be a number of cities
    # GC = 20  # generation cnt
    GC = 20  # generation cnt
    NP = 20  # population cnt
    DC = 20  # In TSP, it will be a number of cities

    figsize = (10, 6)
    # NP = 20  # population cnt
    # GC = 40  # generation cnt
    # DC = 8  # In TSP, it will be a number of cities
    # figsize = (6, 4)

    pause_path = 0.01
    pause_iter_res = 3
    path_vis_str_mltp_const = 0.025

    population = None

    def __init__(s, nxgraphType=None, nxgraphOptions=None, graphData=None):
        if nxgraphType is not None:
            s.nxgraphType = nxgraphType
        s.nxgraphOptions = nxgraphOptions
        s.graphData = graphData
        if graphData is None:
            if nxgraphOptions is None:
                if s.nxgraphType == "complete_graph":
                    s.G = nx.complete_graph(s.DC)
                else:
                    s.G = getattr(nx, s.nxgraphType)()
            else:
                s.G = getattr(nx, s.nxgraphType)(s.nxgraphOptions)
        else:
            s.G = nx.from_edgelist(ast.literal_eval(s.graphData))
        # generates random weights to graph
        # poss = ((100,100), (400,100), (100,400), (400,400))
        # poss = ((100,100), (300,100), (100,200), (400,400))
        poss = None
        idx = 0
        for (u, v) in s.G.nodes(data=True):
            if poss is not None:
                v['pos'] = poss[idx]
            else:
                v['pos'] = (random.randint(s.distances[0], s.distances[1]), random.randint(s.distances[0], s.distances[1]))
            idx = idx + 1

        for (u, v, w) in s.G.edges(data=True):
            # w['weight'] = (random.randint(1, 40))
            u1 = s.G.nodes()[u]['pos']
            v1 = s.G.nodes()[v]['pos']
            real_dist = np.sqrt(np.power(u1[0]-v1[0], 2) + np.power(u1[1]-v1[1], 2))
            w['weight'] = int(real_dist)
            w['fer_n'] = 0
            w['fer_n_sum'] = s.feromon_prime_str_init
            w['fer_o'] = 0

        # or load graph with weights them directly...
        # s.G = nx.from_edgelist(list(
        #     [(0, 1, {'weight': 15}), (0, 3, {'weight': 34}), (0, 4, {'weight': 25}), (1, 2, {'weight': 5}),
        #      (1, 7, {'weight': 23}), (2, 3, {'weight': 33}), (2, 6, {'weight': 29}), (3, 5, {'weight': 13}),
        #      (4, 5, {'weight': 5}), (4, 7, {'weight': 20}), (5, 6, {'weight': 38}), (6, 7, {'weight': 3})]
        #     ))


        s.plt = plt
        s.fig, s.ax = plt.subplots()
        s.idx_weights = range(2, 30, 1)

        pos = {point: point for point in list(map(lambda x: x[1]['pos'], s.G.nodes(data=True)))}
        s.layout = list(pos)
        # s.fig.set_title("BIA - #7 - And Colony Optimization applied to Travelling Salesman Problem (TSP)")
        s.ax.set_xlim(s.distances[0], s.distances[1])
        s.ax.set_ylim(s.distances[0], s.distances[1])
        s.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        s.update()
        s.plt.pause(1)

    def show_sum_feromon(s):
        weights = list(map(lambda x: x[2]['fer_n_sum'] if x[2]['fer_n_sum'] != 0 else 0, list(s.G.edges(data=True))))
        nx.draw_networkx_edges(s.G, pos=s.layout, edgelist=list(s.G.edges()), width=weights, edge_color='green')
        pprint(list(s.G.edges(data=True)))

    def show_path(s, ga, ars='->', color='k', w=None, draw=True):
        price = 0
        ea = s.get_edges(ga)
        for e in list(ea):
            price = price + list(list(s.get_edges(ga))[0])[0][2]['weight']
        if draw:
            for e in ea:
                if w == None:
                    ew = w
                else:
                    # ew = s.idx_weights[:len(ga)]
                    ew = 1
                nx.draw_networkx_edges(s.G, pos=s.layout, edgelist=e, width=w, arrowstyle=ars, arrows=True, edge_color=color)
                s.show_axes()
        return ga, price

    vap_const = 0.5
    vap_sum_const = 0.3

    def vaporize_paths(s, price):
        # sum feromon intenzities of whole graph to single
        for e in list(s.G.edges(data=True)):
            e[2]['fer_sum_n'] = 0
            for p in range(0, len(s.np)):
                for seg_i in range(0,len(s.np[p])-1):
                    e[2]['fer_sum_n'] = e[2]['fer_sum_n'] + s.G.get_edge_data(s.np[p][seg_i], s.np[p][seg_i+1])['fer_n']

        # vaporize all paths
        for e in list(s.G.edges(data=True)):
            if 'fer_sum_o' not in e[2]:
                e[2]['fer_sum_o'] = 0
            e[2]['fer_sum_vap'] = (e[2]['fer_sum_o'] + e[2]['fer_sum_n']) * (s.vap_const/price)
            e[2]['fer_sum_o'] = e[2]['fer_sum_vap']

        # store new feromon to old feromon data
        for e in list(s.G.edges(data=True)):
            e[2]['fer_o'] = e[2]['fer_n']
            e[2]['fer_n'] = 0

        return 0

    def get_all_feromon_weight(s):
        return 0

    def get_best_paths_rel_fer(s):
        return s.get_some_paths_rel_fer()

    max_w = 10000000

    def get_some_paths_rel_fer(s, best=True):
        extreme = 0
        if not best:
            extreme = s.max_w  # TODO const inf...
        if (len(s.op) == 0):
            return -1, extreme
        min_idx = 0
        for i in range(0, len(s.op)-1):
            ap = s.op[i]
            sum = 0
            for pi in range(0, len(ap)-1):
                ni = ((pi+1) % len(ap))
                sum = sum + s.G.get_edge_data(ap[pi], ap[ni])['fer_o']

            if best and sum > extreme:
                extreme = sum
                min_idx = i
            elif not best and sum < extreme:
                extreme = sum
                min_idx = i
        return min_idx, extreme

    def get_worst_paths_rel_fer(s):
        return s.get_some_paths_rel_fer(False)

    old_best_price = None
    best_path = []

    feromon_const = 0.5
    feromon_prime_str_init = 0.1
    feromon_prime_str = 0.1

    def update_feromon(s, cur_node, next_node):
        besti, best = s.get_best_paths_rel_fer()
        worsti, worst = s.get_worst_paths_rel_fer()
        if best == 0 or worst == s.max_w:
            # return 0
            s.G.get_edge_data(cur_node, next_node)['fer_o'] = s.feromon_prime_str
        old_w = s.G.get_edge_data(cur_node, next_node)['fer_o']
        new_w = old_w + ((s.feromon_const * best) / worst)
        s.G.get_edge_data(cur_node, next_node)['fer_n'] = new_w
        s.G.get_edge_data(cur_node, next_node)['fer_n_sum'] = s.G.get_edge_data(cur_node, next_node)['fer_n_sum'] + new_w
        s.G.get_edge_data(cur_node, next_node)['fer_n_sum'] = s.G.get_edge_data(cur_node, next_node)['fer_n_sum'] * s.vap_sum_const
        return new_w

    alpha = 0.5
    beta = 0.5

    def tau(self, r, s):
        return 0

    def tau_eta(s, e):
        return np.power(e[2]['fer_n_sum'], s.alpha) * np.power(1/e[2]['weight'], s.beta)

    def next_node(s, ant, idx, path, cur):
        oes = list(filter(lambda x: x[1] not in path, list(s.G.edges(cur, data=True))))

        fer_dist_sum = 0
        psts = list()
        psts_all = list()
        for i in range(0, len(oes)):
            e = oes[i]
            fer_dist_sum = fer_dist_sum + s.tau_eta(e)
            psts.append(s.tau_eta(e))
        for i in range(0, len(oes)):
            if fer_dist_sum != 0:
                psts_all.append(psts[i] / fer_dist_sum)
            else:
                psts_all.append(0)

        pprint(psts_all)
        if sum(psts_all) == 0:
            next_e = random.choices(oes, k=1)
        else:
            next_e = random.choices(oes, weights=psts_all, k=1)

        return next_e[0][1]

    max_nn_search_cnt = 10

    def walk_path(s, ant):
        # at start position
        # pick edge depending on probability calculated with pheromone intensities
        path = list()
        cur_node = 0
        path.append(cur_node)
        for i in range(0, s.DC-1):
            next_node = s.next_node(ant, i, path, cur_node)
            nn_i = 0
            while next_node in path:
                next_node = s.next_node(ant, i, path, cur_node)
                nn_i = nn_i + 1
            s.update_feromon(cur_node, next_node)
            cur_node = next_node
            path.append(cur_node)
        s.update_feromon(next_node, s.start_node)

        return path

    def alg(s):
        """
            Genetic alg. for solving TSP
        """
        # generate ant colors
        s.colors = list()
        for i in range(0,s.NP):
            r = random.randint(0, 255) / 255.0
            g = random.randint(0, 255) / 255.0
            b = random.randint(0, 255) / 255.0
            rgb = [r, g, b]
            s.colors.append(rgb)
        # prepare ants generations data structures
        s.np = list()
        s.op = list()

        for i in range(0, s.GC):
            # for each ant find path
            for ai in range(0, s.NP):
                new_path = s.walk_path(ai)
                s.np.append(new_path)
                # display ant path
                width_a = np.power((s.NP - ai),s.path_vis_str_mltp_const)
                col = s.colors[ai]
                color = matplotlib.colors.to_rgba((col[0], col[1], col[2], 0.5), alpha=1.0)
                s.show_path(new_path, color=color, w=width_a)
                s.plt.pause(s.pause_path)
            besti, best = s.get_best_paths_rel_fer()
            # calc sum feromons and path price for later use and visualize it
            if besti != -1:
                s.update()
                s.show_sum_feromon()
                ga, price = s.show_path(s.op[besti], color='red', w=1)

                if Vis2D.old_best_price is None or price < Vis2D.old_best_price:
                    Vis2D.old_best_price = price
                    s.best_path = s.op[besti]
                Vis2D.min_individual_price = price
                s.vaporize_paths(price)
            else:
                s.vaporize_paths(1)

            # reset paths for each generation
            s.op = s.np
            s.np = list()
            # visualization base
            s.plt.pause(s.pause_iter_res)
            s.update()
            # s.update()

        s.show_sum_feromon()
        s.show_path(s.best_path, color='blue', w=1)

        s.plt.pause(10)
        pass

    def get_edges(s, nodes):
        edges = list()

        def get_edge(i1, i2):
            return list(filter(lambda i: i[1] == i2, s.G.edges(i1, data=True)))

        for i in range(0, len(nodes) - 1):
            edges.append(get_edge(nodes[i], nodes[i + 1]))
        edges.append(get_edge(nodes[len(nodes) - 1], nodes[0]))
        return edges

    #########################
    # additional methods
    #   - visualizaiton, etc.
    #########################

    def update(s, edges=None):
        """
        clear and update default network
        """
        if edges is None:
            edges = list(list())
        s.ax.clear()
        s.show_axes()

        # Background nodes
        pprint(s.G.edges())
        # nx.draw_networkx_edges(s.G, pos=s.layout, edge_color="gray", arrowstyle='-|>', arrowsize=10)
        forestNodes = list([item for sublist in (([l[0], l[1]]) for l in edges) for item in sublist])

        # dbg("forestNodes", forestNodes)
        forestNodes = list(filter(None, forestNodes))
        # dbg("forestNodes -!None", forestNodes)
        # dbg(set(self.G.nodes()))
        # null_nodes = nx.draw_networkx_nodes(s.G, pos=s.layout, nodelist=set(s.G.nodes()) - set(forestNodes),
        #                                     node_color="white", ax=s.ax)
        null_nodes = nx.draw_networkx_nodes(s.G, pos=s.layout, nodelist=set(s.G.nodes()), node_color="white", ax=s.ax)

        # start node highlight
        nx.draw_networkx_nodes(s.G, pos=s.layout, nodelist=set(filter(lambda i: i == s.start_node, s.G.nodes())), node_color="green", ax=s.ax)

        if (null_nodes is not None):
            null_nodes.set_edgecolor("gray")
            nullNodesIds = set(s.G.nodes()) - set(forestNodes)
            # dbg("nullNodes", nullNodes)
            nx.draw_networkx_labels(s.G, pos=s.layout, labels=dict(zip(nullNodesIds, nullNodesIds)),
                                    font_color="black",
                                    ax=s.ax)

        # Query nodes
        s.idx_colors = sns.cubehelix_palette(len(forestNodes), start=.5, rot=-.75)[::-1]
        color_map = []

        query_nodes = nx.draw_networkx_nodes(s.G, pos=s.layout, nodelist=forestNodes,
                                             node_color=s.idx_colors[:len(forestNodes)], ax=s.ax)

        if query_nodes is not None:
            query_nodes.set_edgecolor("white")
        # nx.draw_networkx_labels(s.G, pos=s.layout, labels=dict(zip(forestNodes[0], forestNodes[0])), font_color="red", ax=s.ax)
        nx.draw_networkx_labels(s.G, pos=s.layout, labels=dict(zip(forestNodes, forestNodes)), font_color="white", ax=s.ax)

        edges = list((l[0], l[1]) for l in edges)
        dbg("edges", edges)
        # nx.draw_networkx_edges(s.G, pos=s.layout, edgelist=edges, width=s.idx_weights[:len(edges)] ) # , ax=s.ax, arrowstyle='->', arrowsize=10

        # draw weights
        labels = nx.get_edge_attributes(s.G, 'weight')
        # nx.draw_networkx_edge_labels(s.G, s.layout, edge_labels=labels)

        # Scale plot ax
        # s.ax.set_xticks([])
        # s.ax.set_yticks([])

        # s.ax.set_title("Step #{}, Price: {}".format(Vis2D.frameNo, Vis2D.min_individual_price))
        s.ax.set_title("Step #{}, NP: {}, GC {}, DC: {}, Price: {}, Best Price: {}".format(Vis2D.frameNo,Vis2D.NP,Vis2D.GC,Vis2D.DC, Vis2D.min_individual_price, Vis2D.old_best_price))
        s.show_axes()

        # self.plt.pause(5)
        # s.plt.pause(s.frameTimeout)
        # self.plt.pause(3)
        Vis2D.frameNo += 1
    def show_axes(s):
        s.ax.set_xlim(s.distances[0]-(s.distances[1]*0.1), s.distances[1]+(s.distances[1]*0.1))
        s.ax.set_ylim(s.distances[0]-(s.distances[1]*0.1), s.distances[1]+(s.distances[1]*0.1))
        s.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)


# 3. Genetic algorithm used to solve Traveling Salesman Problem (TSP) (8 p)
class TSP(Vis2D):
    pass

r = TSP()
r.alg()


exit(0)
