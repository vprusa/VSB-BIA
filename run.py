import sys
from pprint import pprint

import numpy as np

try:
    import seaborn as sns
    pass
except UserWarning:
    pass
import matplotlib.animation ; matplotlib.use("TkAgg")

from Vis3D import *

# Vaším úkolem je naimplementovat vizualizaci prohledávání globálního minima v prostoru.
# Vizualizaci proveďte nad funkcemi
# Sphere, Schwefel, Rosenbrock, Rastrigin, Griewangk, Levy, Michalewicz, Zakharov, Ackley
# https://www.sfu.ca/~ssurjano/optimization.html

class Test(Vis3D):
    def alg(s, dx, dy):
        return None

class Sphere(Vis3D):

    def alg(s, dx, dy):
        return (np.power(dx, 2) + (np.power(dy, 2)))

class Schwefel(Vis3D):
    """
    Problematic, neads tweaking of params for generating next generation
    """
    plane = [-500, 500, 500]

    c = 418.9829

    def alg(s, dx, dy):
        return (s.c * s.d - ( (dx * np.sin(np.sqrt(np.abs(dx))))) + (dy * np.sin(np.sqrt(np.abs(dy)))))

class Rosenbrock(Vis3D):
    def alg(s, dx, dy):
        return (100 * np.power((dy - np.power(dx, 2)), 2) + np.power((dx - 1), 2))

class Rastrigin(Vis3D):

    plane = [-5, 5, 30]

    c = 10

    def alg(s, dx, dy):
        def single(x):
            return (np.power(x, 2) - s.c * np.cos(2 * np.pi * x))
        return ((s.c * s.d ) + single(dx) + single(dy))

class Griewangk(Vis3D):

    def alg(s, dx, dy):
        return None

class Levy(Vis3D):

    def alg(s, dx, dy):
        return None

class Michalewicz(Vis3D):

    def alg(s, dx, dy):
        return None

class Zakharov(Vis3D):

    def alg(s, dx, dy):
        return None

class Ackley(Vis3D):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = 2

    def alg(s, xi, yi):
        part1 = - s.a * np.exp((-s.b * np.sqrt((1.0 / s.d) * (np.power(xi, 2) + np.power(yi, 2)))))
        part2 = - np.exp((1.0 / s.d) * (np.cos(s.c * xi) + np.cos(s.c * yi)))
        return part1 + part2 + s.a + np.exp(1)


# while True:
#
# Sphere, Schwefel, Rosenbrock, Rastrigin, Griewangk, Levy, Michalewicz, Zakharov\nAckley
# r = Ackley()
# r = Rosenbrock()

# r = Sphere()
# r = Schwefel()
# r = Rosenbrock()
r = Rastrigin()
# r = Griewangk()
# r = Levy()
# r = Michalewicz()
# r = Zakharov()
# r = Ackley()

exit(0)
