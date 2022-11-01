try:
    import seaborn as sns
    pass
except UserWarning:
    pass
import matplotlib.animation ; matplotlib.use("TkAgg")

from Vis2D import *

# 3. Genetic algorithm used to solve Traveling Salesman Problem (TSP) (8 p)
class TSP(Vis2D):
    pass

r = TSP()
r.alg()


exit(0)
