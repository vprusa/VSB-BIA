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
    # https://en.wikipedia.org/wiki/Griewank_function

    plane = [-100, 100, 80]

    def alg(s, dx, dy):
        def single2(x, i):
            return np.cos(x/np.sqrt(i))
        def single1(x, i):
            return np.power(x,2)
        return 1 + 1/4000 * (single1(dx, 1) + single1(dy, 2)) - (single2(dx, 1) * single2(dy, 2))

class Levy(Vis3D):

    plane = [-10, 10, 30]

    def alg(s, dx, dy):
        def wi(x):
            return 1 + (x -1)/4
        def single(x):
            return np.power((wi(x) -1),2) * (1+10*np.power(np.sin(np.pi * wi(x) + 1), 2))
        return np.power(np.sin(np.pi * wi(dx)),2) + single(dx) + single(dy) + np.power(wi(dy)-1, 2)*(1+(np.power(np.sin(2*np.pi*wi(dy)),2)))

class Michalewicz(Vis3D):

    m = 10
    plane = [0, 4, 30]

    def alg(s, dx, dy):
        def single(x, i):
            return (np.sin(x)*np.power(np.sin((i*np.power(x,2))/np.pi),2*s.m))
        return - (single(dx, 1) + single(dy,2))

class Zakharov(Vis3D):

    plane = [-10, 10, 30]

    def alg(s, dx, dy):
        def single1(x):
            return np.power(x,2)
        def single2(x, i):
            return 0.5 * i * x
        def single3(x, i):
            return 0.5 * i * x
        return single1(dx) + single1(dy) + np.power(single2(dx,1) + single2(dy,2),2) + np.power(single3(dx,1) + single3(dy,2),4)

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
# Sphere, Schwefel, Rosenbrock, Rastrigin, Griewangk, Levy, Michalewicz, Zakharov, Ackley

r = Sphere()
r = Schwefel()
r = Rosenbrock()
r = Rastrigin()
r = Griewangk()
r = Levy()
r = Michalewicz()
r = Zakharov()
r = Ackley()

# note: anything with spikes is problematic and playing parameters for
#   generating next generation of neighbours might help

exit(0)
