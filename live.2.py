import sys
from pprint import pprint
try:
    import seaborn as sns
    pass
except UserWarning:
    pass
import matplotlib.animation ; matplotlib.use("TkAgg")

from Vis3D import *

# from algorithms.ThreeD.Test.Test import Test as Test
# r = Test()
# r.alg(r.G)

class Rosenbrock(Vis3D):
    def alg(s, dx, dy):
        # Rosenbrock Function
        return (100 * np.power((dy - np.power(dx, 2)), 2) + np.power((dx - 1), 2))

class Ackley(Vis3D):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = 2

    def alg(s, xi, yi):
        part1 = - s.a * np.exp((-s.b * np.sqrt((1.0 / s.d) * (np.power(xi, 2) + np.power(yi, 2)))))
        part2 = - np.exp((1.0 / s.d) * (np.cos(s.c * xi) + np.cos(s.c * yi)))
        return part1 + part2 + s.a + np.exp(1)


while True:
    r = Ackley()
    r = Rosenbrock()
# r.alg()
# try:
#     r.alg(r.G)
# except:
#     exc_info = sys.exc_info()
#     pprint(exc_info)
# r.plt.pause(10)
exit(0)
