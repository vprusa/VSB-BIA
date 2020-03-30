try:
    import seaborn as sns
    pass
except UserWarning:
    pass
import matplotlib.animation ; matplotlib.use("TkAgg")


from algorithms.mst.JarnikPrimDijkstra.JarnikPrimDijkstra_PriorityQueue import JarnikPrimDijkstra_PriorityQueue as JarnikPrimDijkstra_PriorityQueue
r = JarnikPrimDijkstra_PriorityQueue()
r.alg(r.G)
r.plt.pause(10)
exit(0)
#
# from algorithms.mst.JarnikPrimDijkstra.JarnikPrimDijkstra import JarnikPrimDijkstra as JarnikPrimDijkstra
# r = JarnikPrimDijkstra()
# r.alg(r.G)
# r.plt.pause(10)
# exit(0)

from algorithms.mst.Kruskal.Kruskal import Kruskal as Kruskal
r = Kruskal()
r.alg(r.G)
r.plt.pause(10)
exit(0)
