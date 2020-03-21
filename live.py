import networkx as nx
import numpy as np
from pprint import pprint
from time import sleep
import random
from random import randrange
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    pass
except UserWarning:
    pass
import matplotlib.animation ; matplotlib.use("TkAgg")

# Create Graph
# np.random.seed(2)
# G = nx.cubical_graph()
# G = nx.relabel_nodes(G, {0:"O", 1:"X", 2:"XZ", 3:"Z", 4:"Y", 5:"YZ", 6: "XYZ", 7:"XY"})
# pos = nx.spring_layout(G)

from algorithms.mst.Kruskal import Kruskal as Kruskal
r = Kruskal()
#r.G = nx.complete_graph(4)
r.G = nx.cubical_graph()

# code creating G here
for (u, v, w) in r.G.edges(data=True):
    w['weight'] = random.randint(0, 40)

G = r.G
pos = nx.spring_layout(G)

idx_colors = sns.cubehelix_palette(8, start=.5, rot=-.75)[::-1]
#idx_weights = [3,2,1]
idx_weights = range(3,1)
# Build plot
fig, ax = plt.subplots(figsize=(6,4))

def update(num):
    ax.clear()
    # i = num // 3
    # j = num % 3 + 1
    # triad = sequence_of_letters[i:i+3]
    # path = ['0'] + ["".join(sorted(set(triad[:k + 1]))) for k in range(j)]
    #path = [item for item in range(0,random.randint(1, 6))]
    #path = [0,1,2,3,4,5,6,7]
    path = range(num+1)
    pprint(path)
    # Background nodes
    nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color="gray")
    null_nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=set(G.nodes()) - set(path), node_color="white",  ax=ax)
    if(null_nodes is not None):
        null_nodes.set_edgecolor("black")

    # Query nodes
    query_nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=path, node_color=idx_colors[:len(path)], ax=ax)
    if(query_nodes is not None):
        query_nodes.set_edgecolor("white")
    nx.draw_networkx_labels(G, pos=pos, labels=dict(zip(path,path)),  font_color="white", ax=ax)
    edgelist = [path[k:k+2] for k in range(len(path) - 1)]
    nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, width=idx_weights[:len(path)], ax=ax)

    # Scale plot ax
    # ax.set_title("Frame %d:    "%i(num+1) +  " - ".join(path), fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

#update()
# pprint("qwe")
# sleep(1000)
# pprint("qwe2")
# update(num = 8, path = [2,3,4])
# plt.show()

#ani = matplotlib.animation.FuncAnimation(fig, update, frames=8, interval=500, repeat=True)
#plt.show(block=False)
#plt.show()

for i in range(1,8):
    ax.cla()
    #ax.imshow(update(i))
    update(i)
    ax.set_title("frame {}".format(i))
    # Note that using time.sleep does *not* work here!
    plt.pause(0.5)

# sleep(10)
#plt.show()
