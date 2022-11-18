
from random import random

from base import *
import numpy as np
import random
import scipy.interpolate

try:
    import seaborn as sns
except UserWarning:
    pass
import matplotlib.animation ; matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

class Pt(object):
    x = 0
    y = 0
    z = 0
    vx = 0
    vy = 0
    vz = 0
    def __init__(s, x, y):
        s.x = x
        s.y = y



def ackley(xi, yi, a, b, c, d):
    part1 = - a * np.exp((-b * np.sqrt((1.0 / d) * (np.power(xi, 2) + np.power(yi, 2)))))
    part2 = - np.exp((1.0 / d) * (np.cos(c * xi) + np.cos(c * yi)))
    return part1 + part2 + a + np.exp(1)
    # return (100 * np.power((xi - np.power(yi, 2)), 2) + np.power((yi - 1), 2))
    # plane = [-10, 10, 30]

    # def alg(s, dx, dy):
    # def single1(x):
    #     return np.power(x,2)
    # def single2(x, i):
    #     return 0.5 * i * x
    # def single3(x, i):
    #     return 0.5 * i * x
    # return single1(dx) + single1(dy) + np.power(single2(dx,1) + single2(dy,2),2) + np.power(single3(dx,1) + single3(dy,2),4)


def ackley2(xi, a, b, c, d):
    # part1 = - a * np.exp((-b * np.sqrt((1.0 / d) * (np.power(xi, 2) + np.power(yi, 2)))))
    # part2 = - np.exp((1.0 / d) * (np.cos(c * xi) + np.cos(c * yi)))
    # return part1 + part2 + a + np.exp(1)
    # return (100 * np.power((xi - np.power(yi, 2)), 2) + np.power((yi - 1), 2))
    # plane = [-10, 10, 30]
    part1 = - a * np.exp((-b * np.sqrt((1.0 / d) * (np.power(xi, 2)))))
    part2 = - np.exp((1.0 / d) * (np.cos(c * xi)))
    return part1 + part2 + a + np.exp(1)

    # def alg(s, dx, dy):
    # def single1(x):
    #     return np.power(x,2)
    # def single2(x, i):
    #     return 0.5 * i * x
    # def single3(x, i):
    #     return 0.5 * i * x
    # return single1(dx) + single1(dy) + np.power(single2(dx,1) + single2(dy,2),2) + np.power(single3(dx,1) + single3(dy,2),4)


def f(xi, xi1):
    # Rosenbrock Function
    return (100 * np.power((xi1 - np.power(xi, 2)), 2) + np.power((xi - 1), 2))

class Vis2D5(object):
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
    points_cnt = 30
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
        Vis2D5.plt = plt
        Vis2D5.frameNo = 0

        s.d = len(s.plane) - 1

        if Vis2D5.fig is None:
            Vis2D5.fig = plt.figure("BIA - #4 - 4. Differential evolution")
            Vis2D5.plt.clf()

        # ax = Vis2D5.plt.axes(projection='2d')
        ax = Vis2D5.plt.axes()
        s.ax = ax

        s.vis_base()
        # Vis2D5.plt.pause(3)

        s.dx = np.random.choice(s.x, s.points_cnt)

        # swarm = Generate pop_size random individuals (you can use the class Solution mentioned in Exercise 1)
        s.swarm = s.gen_pop()
        # gBest = Select the best individual from the population
        s.gb = s.swarm[s.sel_best(s.swarm)]
        # For each particle, generate velocity vector v
        # s.gbv = s.gen_vel(s.swarm)
        s.gen_vel(s.swarm)
        s.gnb = s.gb.copy()
        # m = 0
        s.m = 0
        s.M_max = 10

        # while m < M_max :
        while s.m < s.M_max:
            # for each i, x in enumerate(swarm):
            idx = 0
            for i in s.swarm:
        #     Calculate a new velocity v for a particle x # Check boundaries of velocity (v_mini, v_maxi)
                s.nv = s.calc_vel(i)
        #     Calculate a new position for a particle x # Old position is always replaced by a new position. CHECK BOUNDARIES!
                s.nx, s.ny = s.calc_np(i)
        #     Compare a new position of a particle x to its pBest
        #     if new position of x is better than pBest:
                if s.pos_better(s.nx, s.ny, i, idx):
        #         pBest = new position of x
                    s.gnb[idx] = (s.xn, s.ny)
        #         if pBest is better than gBest:
                    if s.pbest_gbest(s.nx, s.gb, s.pb):
                        s.gb = s.pb
        #             gBest = pBest
                idx = idx + 1
        # m += 1/
            s.m = s.m + 1

        s.update()
        s.g = s.g + 1
        Vis2D5.plt.pause(5)

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
        # return ackley2(dx, s.a, s.b, s.c, s.d)
        return ackley(dx, dy, s.a, s.b, s.c, s.d)
        # return (100 * np.power((dy - np.power(dx, 2)), 2) + np.power((dx - 1), 2))

    def cmp(s, oldx, oldy, newx, newy):
        newz = s.alg(newx, newy)
        oldz = s.alg(oldx, oldy)
        if oldz < newz:
            return oldx, oldy
        else:
            return newx, newy

    # plane = [-10, 10, 30]
    def Zakharov(s, dx, dy):
        def single1(x):
            return np.power(x, 2)

        def single2(x, i):
            return 0.5 * i * x

        def single3(x, i):
            return 0.5 * i * x

        return single1(dx) + single1(dy) + np.power(single2(dx, 1) + single2(dy, 2), 2) + np.power(
            single3(dx, 1) + single3(dy, 2), 4)

    def vis_base(s):
        s.plane = [-10, 10, 60]

        s.x = np.linspace(s.plane[0], s.plane[1], s.plane[2])
        s.y = np.linspace(s.plane[0], s.plane[1], s.plane[2])
        s.z = np.linspace(s.plane[0], s.plane[1], s.plane[2])

        s.a = 20
        s.b = 0.2
        s.c = 2 * np.pi
        s.d = 2
        s.X, s.Y = np.meshgrid(s.x, s.y)

        # s.Z = s.alg(s.X, s.Y)
        # s.rbf = scipy.interpolate.Rbf(s.X, s.Y, s.Z, function='linear')
        # s.zi = s.rbf(s.x, s.y)
        #
        # # s.ax.plot_wireframe(s.X, s.Y, cmap="coolwarm", zorder=1)
        #
        # s.plt.imshow(s.zi, vmin=s.z.min(), vmax=s.z.max(), origin='lower', extent=[s.x.min(), s.x.max(), s.y.min(), s.y.max()])
        # s.plt.scatter(s.x, s.y, c=s.z)
        # s.plt.colorbar()
        # s.plt.show()

        # Generate data:
        # x, y, z = 100 * np.random.random((3,10))

        # Set up a regular grid of interpolation points
        # s.plane[2] = 100
        s.xi, s.yi = np.linspace(s.plane[0], s.plane[1], s.plane[2]), np.linspace(s.plane[0], s.plane[1], s.plane[2])
        s.xi, s.yi = np.meshgrid(s.xi, s.yi)

        # Interpolate
        s.zi = s.alg(s.xi, s.yi)
        # s.zi = s.Zakharov(s.xi, s.yi)

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html
        s.rbf = scipy.interpolate.Rbf(s.xi, s.yi, s.zi, function='linear')
        # s.zi = s.rbf(s.xi, s.yi)

        s.plt.imshow(s.zi, vmin=s.zi.min(), vmax=s.zi.max(), origin='lower', extent=[s.xi.min(), s.xi.max(), s.yi.min(), s.yi.max()])
        # s.plt.imshow(s.zi, vmin=s.zi.min(), vmax=s.zi.max(), origin='lower', extent=[s.x.min(), s.x.max(), s.y.min(), s.y.max()])
        # plt.scatter(x, y, c=z)
        # s.plt.colorbar()
        s.plt.show()

        s.ax.set_xlabel('x')
        s.ax.set_ylabel('y')
        s.ax.set_zlabel('z')
        pass

    def update(s):
        dbg("Iteration", Vis2D5.frameNo)
        # Scale plot ax
        s.ax.set_xticks([])
        s.ax.set_yticks([])
        s.ax.set_title(s.__class__.__name__ + ": Step #{} ".format(Vis2D5.frameNo))

        Vis2D5.plt.pause(s.frameTimeout)
        Vis2D5.frameNo += 1
        # if(s.max_iterations - 5 < VisualizationBase3D.frameNo):

    def calc_vel(s, i):
        pass

    def gen_pop(s):
        def rnd():
            return [random.uniform(s.plane[0], s.plane[1]) for _ in range(s.points_cnt)]
        return list(map(lambda x: Pt(x[0], x[1]), list(rnd(), rnd())))

    def pos_better(s, nx, ny, i, idx):
        pass

    def calc_np(s, i):
        pass

    b_lo = 0
    b_up = 1

    def gen_vel(s, swarm):
        def rnd():
            return random.uniform(s.b_lo, s.b_up)
        for p in swarm:
            p.vx = rnd()
            p.vy = rnd()
        pass

    def sel_best(s, swarm):
        bi = swarm[0]
        bv = s.alg(bi)
        bidx = 0
        idx = 0
        for p in swarm:
            nbv = s.alg(p.x, p.y)
            if nbv > bv:
                bv = nbv
                bidx = swarm[idx]
            idx = idx + 1
        return bidx


r = Vis2D5()

exit(0)
