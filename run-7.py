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

    g1 = list()  # generation
    g2 = list()  # generation
    # ng = list()  # next generation


    cg = list(list())
    ng = list(list())

    distances = (0,500)

    start_node = 0

    # NP = 20
    # G = 200
    # D = 20  # In TSP, it will be a number of cities
    nxgraphType = "complete_graph"

    # NP = 6  # population cnt
    # DC = 6  # In TSP, it will be a number of cities
    # NP = 3  # population cnt
    # DC = 4  # In TSP, it will be a number of cities
    # GC = 5  # generation cnt
    # NP = 10  # population cnt
    # DC = 10  # In TSP, it will be a number of cities
    GC = 15  # generation cnt
    # GC = 20  # generation cnt
    NP = 20  # population cnt
    DC = 20  # In TSP, it will be a number of cities

    figsize = (10, 6)
    # NP = 20  # population cnt
    # GC = 40  # generation cnt
    # DC = 8  # In TSP, it will be a number of cities
    # figsize = (6, 4)

    pause_path = 0.01
    pause_iter_res = 1
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

        # for (u, v) in s.G.nodes(data=True):
        #     v['pos'] = (random.randint(s.distances[0], s.distances[1]), random.randint(s.distances[0], s.distances[1]))
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
        # s.fig, s.ax = plt.fig("BIA - #3 - Genetic alg. on Traveling Salesman Problem (TSP) ", figsize=s.figsize)
        s.fig, s.ax = plt.subplots()
        # s.fig, s.ax = plt.subplots()
        s.idx_weights = range(2, 30, 1)

        # s.layout = nx.circular_layout(s.G)
        # list(map(lambda x: x[1]['pos'],s.G.nodes(data=True)))
        pos = {point: point for point in list(map(lambda x: x[1]['pos'], s.G.nodes(data=True)))}
        s.layout = list(pos)
        # s.fig = s.plt.figure("BIA - #3 - Genetic alg. on Traveling Salesman Problem (TSP) ", figsize=s.figsize)
        # s.fig.set_title("BIA - #3 - Genetic alg. on Traveling Salesman Problem (TSP) ")
        s.fig.set_title("BIA - #7 - ")
        # s.ax = s.plt.axes()
        # s.plt.axis("on")
        s.ax.set_xlim(s.distances[0], s.distances[1])
        s.ax.set_ylim(s.distances[0], s.distances[1])
        s.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        s.update()
        s.plt.pause(1)


    def f(s, i):
        """
            price function, idk why it is called 'f()' in this case,
            but for sake of balancing readeability and documentation
            I use internal function 'price()' (note: should I rename to 'cost()'?)
        """
        return s.price(i)
    #
    # def show_ant_path(s, i, color):
    #     # eval final population
    #     s.min_individual = s.population[0]
    #     Vis2D.min_individual_price = s.f(s.min_individual)
    #     for i in range(1, len(s.population)):
    #         price = s.f(s.population[i])
    #         if (Vis2D.min_individual_price > price):
    #             s.min_individual = s.population[i]
    #             Vis2D.min_individual_price = price
    #     s.update()
    #     s.show_path(s.min_individual, color=color, w=1)
    #     s.plt.pause(s.frameTimeout)


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

    def price(s, g):
        sum = 0
        for i in range(0, len(g) - 2):
            sum = sum + s.G[g[i]][g[i + 1]]['weight']
        return sum

    vap_const = 0.5
    vap_sum_const = 0.95

    def vaporize_paths(s):
        # sum feromon intenzities of whole graph to single

        # s.np_sum = s.np[0].copy()
        # for i in range(1, len(s.np)):
        #     for ii in s.np[i]:
        #         s.np_sum[ii] = s.np_sum[ii] + s.np[i][ii]
        # s.np_sum =w
        for e in list(s.G.edges(data=True)):
            e[2]['fer_sum_n'] = 0
            for p in range(0, len(s.np)):
                for seg_i in range(0,len(s.np[p])-1):
                    e[2]['fer_sum_n'] = e[2]['fer_sum_n'] + s.G.get_edge_data(s.np[p][seg_i], s.np[p][seg_i+1])['fer_n']

        # besti, best = s.get_best_paths_rel_fer()
        # vaporize all paths
        for e in list(s.G.edges(data=True)):
            if 'fer_sum_o' not in e[2]:
                e[2]['fer_sum_o'] = 0
            e[2]['fer_sum_vap'] = (e[2]['fer_sum_o'] + e[2]['fer_sum_n']) * s.vap_const
            e[2]['fer_sum_o'] = e[2]['fer_sum_vap']

        # store new feromon to old feromon data
        for e in list(s.G.edges(data=True)):
            e[2]['fer_o'] = e[2]['fer_n']
            e[2]['fer_n'] = 0
        # s.G.get_edge_data(cur_node, next_node)['fer_n_sum'] = s.G.get_edge_data(cur_node, next_node)['fer_n_sum'] * s.vap_sum_const

        # for i in range(0,len(list(s.G.edges()))-1):
        #     i2 = (i+1) % len(list(s.G.edges()))
        #     s.G.get_edge_data(i, i2)['fer_n_sum'] = s.G.get_edge_data(i, i2)['fer_n_sum'] * s.vap_sum_const

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
                # sum = sum + s.G.get_edge_data(ap[pi], ap[ni])['weight']
                sum = sum + s.G.get_edge_data(ap[pi], ap[ni])['fer_o']
            # sum = sum + s.G.get_edge_data(ap[0], len(ap)-1)['weight']

            if best and sum > extreme:
                extreme = sum
                min_idx = i
            elif not best and sum < extreme:
                extreme = sum
                min_idx = i
        return min_idx, extreme

    def get_worst_paths_rel_fer(s):
        return s.get_some_paths_rel_fer(False)

    def sel_next_node(s, i, cur):
        # return random.choice(s.G.nodes())
        return random.randint(0, s.NP-1)

    old_best_price = None
    best_path = []

    feromon_const = 0.5
    feromon_prime_str_init = 0.1
    feromon_prime_str = 0.1

    def update_feromon(s, cur_node, next_node):
    # def update_feromon(s, path):
        # for pi in range(0, len(path)):
        #     cur_node = path[pi]
        #     ni = (pi+1) % len(path)
        #     next_node = path[ni]
        # delta = s.get_all_feromon_weight()
        besti, best = s.get_best_paths_rel_fer()
        worsti, worst = s.get_worst_paths_rel_fer()
        if best == 0 or worst == s.max_w:
            # return 0
            s.G.get_edge_data(cur_node, next_node)['fer_o'] = s.feromon_prime_str
        old_w = s.G.get_edge_data(cur_node, next_node)['fer_o']
        new_w = old_w + ((s.feromon_const * best) / worst)
        s.G.get_edge_data(cur_node, next_node)['fer_n'] = new_w
        # if 'fer_n_sum' not in s.G.get_edge_data(cur_node, next_node):
        #     s.G.get_edge_data(cur_node, next_node)['fer_n_sum'] = 0
        s.G.get_edge_data(cur_node, next_node)['fer_n_sum'] = s.G.get_edge_data(cur_node, next_node)['fer_n_sum'] + new_w
        s.G.get_edge_data(cur_node, next_node)['fer_n_sum'] = s.G.get_edge_data(cur_node, next_node)['fer_n_sum'] * s.vap_sum_const
        return new_w

    alpha = 0.5
    beta = 0.5

    def tau(self, r, s):
        return 0

    # def next_node(s, ant, idx, path):
    #     oes = list(s.G.edges(0, data=True))
    #     max_i = None
    #     max_fer = 0
    #     for i in range(0, len(oes)-1):
    #         e = oes[i]
    #         if e[1] not in path:
    #             if max_fer <= e[2]['fer_n_sum']:
    #                 max_fer <= e[2]['fer_n_sum']
    #                 max_i = i
    #     return max_i

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
            # if e[1] not in path:
            #     if max_fer <= e[2]['fer_n_sum']:
            #         max_fer <= e[2]['fer_n_sum']
            #         max_i = i
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
        # cur_node = s.G.nodes()[0]
        cur_node = 0
        # path.append(cur_node)
        path.append(cur_node)
        prev_node = -1
        for i in range(0, s.DC-1):
            # next_node = s.sel_next_node(i, cur_node)
            # next_node = random.randint(0, s.DC - 1)
            next_node = s.next_node(ant, i, path, cur_node)
            nn_i = 0
            while next_node in path:
                # next_node = s.sel_next_node(i, cur_node)
                # next_node = random.randint(0, s.DC - 1)
                next_node = s.next_node(ant, i, path, cur_node)
                # if nn_i > s.max_nn_search_cnt:
                nn_i = nn_i + 1
            # s.update_feromon(next_node)
            s.update_feromon(cur_node, next_node)
            cur_node = next_node
            path.append(cur_node)
        s.update_feromon(next_node, s.start_node)

        return path

    def alg(s):
        """
            Genetic alg. for solving TSP
        """
        # population = Generate NP random individuals Evaluate individuals within population
        # s.population = list()
        # for i in range(0, s.NP):
        #     s.population.append(s.random_circle(list(s.G.nodes())))
        s.colors = list()
        for i in range(0,s.NP):
            r = random.randint(0, 255) / 255.0
            g = random.randint(0, 255) / 255.0
            b = random.randint(0, 255) / 255.0
            rgb = [r, g, b]
            s.colors.append(rgb)
        s.np = list()
        s.op = list()
        # init random feromons
        # for e in list(s.G.edges(data=True)):
        #     e[2]['fer_n_sum']

        for i in range(0, s.GC):
            # for each ant find path
            for ai in range(0, s.NP):
                new_path = s.walk_path(ai)
                s.np.append(new_path)
                # s.show_ant_path(i, color='orange')
                width_a = np.power((s.NP - ai),s.path_vis_str_mltp_const)
                # width_a = ((s.NP - ai)*1.5)
                col = s.colors[ai]
                color = matplotlib.colors.to_rgba((col[0], col[1], col[2], 0.5), alpha=1.0)
                # s.show_path(new_path, color=s.colors[ai], w=width)
                s.show_path(new_path, color=color, w=width_a)
                s.plt.pause(s.pause_path)

            s.vaporize_paths()
            color = matplotlib.colors.to_rgba((0, 0, 0, 0.6), alpha=0.3)
            s.op = s.np
            s.np = list()
            s.update()
            besti, best = s.get_best_paths_rel_fer()
            s.show_sum_feromon()
            ga, price = s.show_path(s.op[besti], color='red', w=1)

            if Vis2D.old_best_price is None or price < Vis2D.old_best_price:
                Vis2D.old_best_price = price
                s.best_path = s.op[besti]
            Vis2D.min_individual_price = price
            s.plt.pause(s.pause_iter_res)
            s.update()

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

    def random_circle(s, nodes):
        x = nodes.copy()
        random.shuffle(x)
        return x

    def mutate(s, nodes_, cnt = 1):
        # swaps 2 random elements in array 'cnt' times
        # nodes = nodes_.copy()
        # swapped = random.shuffle(nodes)
        swapped = nodes_.copy()
        for i in range(0, cnt):
            pos1 = random.randint(0, len(nodes_)-1)
            pos2 = random.randint(0, len(nodes_)-1)
            swapped[pos1], swapped[pos2] = swapped[pos2], swapped[pos1]
        return swapped

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
