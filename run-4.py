# try:
#     import seaborn as sns
#     pass
# except UserWarning:
#     pass
# import matplotlib.animation ; matplotlib.use("TkAgg")
from random import random

# from Vis3D4 import *
#

# class Test(Vis3D4):
#     def alg(s, dx, dy):
#         return None
#
# class Ackley(Vis3D4):
#     a = 20
#     b = 0.2
#     c = 2 * np.pi
#     d = 2
#
#     def alg(s, xi, yi):
#         part1 = - s.a * np.exp((-s.b * np.sqrt((1.0 / s.d) * (np.power(xi, 2) + np.power(yi, 2)))))
#         part2 = - np.exp((1.0 / s.d) * (np.cos(s.c * xi) + np.cos(s.c * yi)))
#         return part1 + part2 + s.a + np.exp(1)
#
#
# # while True:
# # Sphere, Schwefel, Rosenbrock, Rastrigin, Griewangk, Levy, Michalewicz, Zakharov, Ackley
#
# r = Ackley()

# note: anything with spikes is problematic and playing parameters for
#   generating next generation of neighbours might help


from base import *
import numpy as np
import random

try:
    import seaborn as sns
except UserWarning:
    pass
import matplotlib.animation ; matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

def ackley(dx, dy, a, b, c, d):
    # part1 = - a * np.exp((-b * np.sqrt((1.0 / d) * (np.power(xi, 2) + np.power(yi, 2)))))
    # part2 = - np.exp((1.0 / d) * (np.cos(c * xi) + np.cos(c * yi)))
    # return part1 + part2 + a + np.exp(1)
    # return (100 * np.power((xi - np.power(yi, 2)), 2) + np.power((yi - 1), 2))
    # plane = [-10, 10, 30]

    # def alg(s, dx, dy):
    def single1(x):
        return np.power(x,2)
    def single2(x, i):
        return 0.5 * i * x
    def single3(x, i):
        return 0.5 * i * x
    return single1(dx) + single1(dy) + np.power(single2(dx,1) + single2(dy,2),2) + np.power(single3(dx,1) + single3(dy,2),4)



def f(xi, xi1):
    # Rosenbrock Function
    return (100 * np.power((xi1 - np.power(xi, 2)), 2) + np.power((xi - 1), 2))

class Vis3D4(object):
    frameNo = 0

    frameTimeout = 0.5
    nxgraphOptions = None
    G = None
    D = None
    plt = None
    layout = None
    fig = None

    dx = None
    dy = None
    dz = None

    dims = 2
    plane = [-32, 32, 60]
    points_cnt = 100
    max_iterations = 50

    a = 20
    b = 0.2
    c = 2 * np.pi
    d = 2

    g = 0
    g_maxim = 20

    F = 0.5
    cr = 0.7

    def __init__(s):
        Vis3D4.plt = plt
        Vis3D4.frameNo = 0

        s.d = len(s.plane) - 1

        if Vis3D4.fig is None:
            Vis3D4.fig = plt.figure("BIA - #4 - 4. Differential evolution")
            Vis3D4.plt.clf()

        ax = Vis3D4.plt.axes(projection='3d')
        s.ax = ax

        s.vis_base()
        Vis3D4.plt.pause(3)

        s.dx = np.random.choice(s.x, s.points_cnt)
        s.dy = np.random.choice(s.y, s.points_cnt)
        s.dz = ackley(s.dx, s.dy, s.a, s.b, s.c, s.d) + (Vis3D4.frameNo * 5)
        s.ax.scatter(s.dx, s.dy, s.dz, marker='o')

        # https://aicorespot.io/differential-evolution-from-the-ground-up-in-python/
        # reasonably documented code is here

        # pop = Generate NP random individuals(you can use the class Solution mentioned in Exercise 1)
        s.pop = [(s.dx[i], s.dy[i]) for i in range(0, len(s.dx))]

        while s.g < s.g_maxim:
            s.new_pop = s.pop.copy()   # new generation
        #     for each i, x in enumerate(pop):  # x is also denoted as a target vector
            i = 0
            for x in s.pop:
                r1, r2, r3 = random.sample(s.pop, 3)
                while s.peq(r1, r2) or s.peq(r2, r3) or s.peq(r3, x):
                    r1, r2, r3 = random.sample(s.pop, 3)
        #         v = (x_r1.params – x_r2.params) * F + x_r3.params  # mutation vector. TAKE CARE FOR BOUNDARIES!
                # wtf... terrible pseudocode, fortunately we have google...
                v = s.mutate([r1, r2, r3], s.F)
        #         u = np.zeros(dimension)  # trial vector
                vb = [np.clip(v[0], s.plane[0], s.plane[1]), np.clip(v[0], s.plane[0], s.plane[1])]
                u = s.crossover(vb, x, s.dims, s.cr)
        #         Note: This part is dealt with parameters
        #         j_rnd = np.random.randint(0, dimension)
        #         for j in range(dimension):
        #             if np.random.uniform() < CR or j == j_rnd:
        #                 u[j] = v[j]  # at least 1 parameter should be from a mutation vector v
        #             else:
        #                 u[j] = x_i.params[j]
        #
        #         f_u = Evaluate
                fx = s.alg(x[0], x[1])
        #         trial vector u
                fu = s.alg(u[0], u[1])
        #         if f_u is better or equals to f_x_i:  # We always accept a solution with the same fitness as a target vector
                if fx > fu:
        #             new_x = Solution(dimension, lower_bound, upper_bound)
                    s.pop[i] = u
        #             new_x.params = u
        #             new_x.f = f_u
        #         Note: I do recalc the function when displayed, it may be faster to cache it...
        #     pop = new_pop
                i = i + 1
            s.ax.clear()
            s.vis_base()
            # store x,y points and calc z
            s.dx = list(map(lambda x: x[0], s.pop))
            s.dy = list(map(lambda y: y[1], s.pop))
            s.dz = list(map(lambda z: s.alg(z[0], z[1]), s.pop))
            # display
            s.ax.scatter(s.dx, s.dy, s.dz, marker='o', zorder=10, color="red")

            s.update()
            s.g = s.g + 1
        Vis3D4.plt.pause(5)

        s.plt.clf()

    def peq(s, i1, i2):
        return i1[0] == i2[0] and i1[1] == i2[1]

    def crossover(s, m, t, dims, cr):
        # generate a uniform random value for every dimension
        p = np.random.rand(dims)
        # generate trial vector by binomial crossover
        trial = [m[i] if p[i] < cr else t[i] for i in range(dims)]
        return trial


    def mutate(s, rr, F):
        res = [0,0]
        # def mutate_i(i, ii):
        #     return (i[0][ii] – i[1][ii]) * F + i[2][ii]
        res[0] = (rr[0][0] - rr[1][0]) * F + rr[2][0]
        res[1] = (rr[0][0] - rr[1][0]) * F + rr[2][0]
        return res

    def alg(s, dx, dy):
        return ackley(dx, dy, s.a, s.b, s.c, s.d)
        # return (100 * np.power((dy - np.power(dx, 2)), 2) + np.power((dx - 1), 2))

    def cmp(s, oldx, oldy, newx, newy):
        newz = s.alg(newx, newy)
        oldz = s.alg(oldx, oldy)
        if oldz < newz:
            return oldx, oldy
        else:
            return newx, newy

    def vis_base(s):

        s.x = np.linspace(s.plane[0], s.plane[1], s.plane[2])
        s.y = np.linspace(s.plane[0], s.plane[1], s.plane[2])

        s.a = 20
        s.b = 0.2
        s.c = 2 * np.pi
        s.d = 2
        s.X, s.Y = np.meshgrid(s.x, s.y)
        s.Z = s.alg(s.X, s.Y)

        s.ax.plot_wireframe(s.X, s.Y, s.Z, cmap="coolwarm", zorder=1)
        s.ax.set_xlabel('x')
        s.ax.set_ylabel('y')
        s.ax.set_zlabel('z')
        pass

    def update(s):
        dbg("Iteration", Vis3D4.frameNo)
        # Scale plot ax
        s.ax.set_xticks([])
        s.ax.set_yticks([])
        s.ax.set_title(s.__class__.__name__ + ": Step #{} ".format(Vis3D4.frameNo))

        Vis3D4.plt.pause(s.frameTimeout)
        Vis3D4.frameNo += 1
        # if(s.max_iterations - 5 < VisualizationBase3D.frameNo):

r = Vis3D4()

exit(0)
