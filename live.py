import networkx as nx
from pprint import pprint
import random
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    pass
except UserWarning:
    pass
import matplotlib.animation ; matplotlib.use("TkAgg")

# Create Graph
from algorithms.mst.Kruskal.Kruskal import Kruskal as Kruskal
r = Kruskal()
r.alg(r.G)
r.plt.pause(10)
exit(0)
