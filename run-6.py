from random import random

from base import *
import numpy as np
import random

try:
    import seaborn as sns
except UserWarning:
    pass
import matplotlib.animation ; matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

class Pt(object):
    # x = 0
    # y = 0
    # z = 0
    # p = list()
    # vx = 0
    # vy = 0
    # vz = 0

    # n = list()

    # nx = 0
    # ny = 0
    # nz = 0

    # PTR vector

    def x(s, v=None):
        if v is not None:
            s.p[0] = v
        return s.p[0]

    def y(s, v=None):
        if v is not None:
            s.p[1] = v
        return s.p[1]

    def z(s, v=None):
        if v is not None:
            if len(s.p) <= 2:
                s.p.append(v)
            else:
                s.p[2] = v
        return s.p[2]

    def __init__(self, xx, yy):
        # s.x(xx)
        # s.y(yy)
        self.ptr_vec = [1, 1]

        self.n = list()
        self.p = list()
        self.p.append(xx)
        self.p.append(yy)

def ackley(xi, yi, a, b, c, d):
    part1 = - a * np.exp((-b * np.sqrt((1.0 / d) * (np.power(xi, 2) + np.power(yi, 2)))))
    part2 = - np.exp((1.0 / d) * (np.cos(c * xi) + np.cos(c * yi)))
    return part1 + part2 + a + np.exp(1)
    # return (100 * np.power((xi - np.power(yi, 2)), 2) + np.power((yi - 1), 2))

class Vis3D(object):
    frameNo = 0

    frameTimeout = 0.0005
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
    points_cnt = 6
    max_iterations = 7

    g = 0

    p_len_multiplier_abs = 1.5
    p_len_multiplier_rel = 1.5
    p_len = 1
    step = 0.2
    ptr = 1

    swarm = list()

    def __init__(s):
        Vis3D.plt = plt
        Vis3D.frameNo = 0

        s.d = len(s.plane) - 1
        s.m = 0

        if Vis3D.fig is None:
            Vis3D.fig = plt.figure("BIA - #5 - 5. Particle swarm optimization")
            Vis3D.plt.clf()

        ax = Vis3D.plt.axes(projection='3d')
        s.ax = ax
        s.ax.view_init(elev=270., azim=0)
        s.vis_base()
        s.update()

        # swarm = Generate pop_size random individuals (you can use the class Solution mentioned in Exercise 1)
        s.gen_pop()
        # gBest = Select the best individual from the population
        s.bg_idx = s.sel_best()
        s.gb = s.swarm[s.bg_idx]
        # For each particle, generate velocity vector v
        s.m = 0
        s.M_max = s.max_iterations
        Vis3D.plt.pause(1)
        s.updatep()
        # s.p_len = ( s.plane[1] - s.plane[0] ) * s.p_len_multiplier_abs

        # while m < M_max :
        while s.m < s.M_max:
            # for each i, x in enumerate(swarm):
            s.ax.scatter(s.gb.x(), s.gb.y(), s.gb.z(), marker='.', color="green")
            idx = 0
            for i in s.swarm:
                s.ax.scatter(i.x(), i.y(), i.z(), marker='o', color="blue")

                t = 0
                t_idx = 0
                i.n = list()
                ptt = Pt(i.x(), i.y())
                ptt.z(s.algp(ptt))
                i.n.append(ptt)
                i.nl = ptt
                while t <= s.p_len * s.p_len_multiplier_rel:
                    s.set_ptr(i)
                    # update new jump position
                    inx_1 = (i.n[t_idx].p[0] + (s.gb.p[0] - i.n[0].p[0]) * s.step * i.ptr_vec[0] * s.p_len_multiplier_abs)
                    iny_1 = (i.n[t_idx].p[1] + (s.gb.p[1] - i.n[0].p[1]) * s.step * i.ptr_vec[1] * s.p_len_multiplier_abs)

                    if s.is_out_range(inx_1) or s.is_out_range(iny_1):
                        break
                    npp = Pt(s.keep_in_range(inx_1), s.keep_in_range(iny_1))
                    npp.z(s.algp(npp))
                    i.n.append(npp)
                    # calc new jump function value
                    iz = s.algp(i.nl)
                    inz = s.algp(i.n[t_idx+1])
                    # TODO double check jumps
                    if inz < iz:
                        i.nl = i.n[t_idx+1]
                    s.updatev(i, t_idx+1)
                    t_idx = t_idx + 1
                    t = t + s.step
                    # TODO visualize here lines
                s.update()
                s.updatep()
                s.ax.scatter(s.gb.x(), s.gb.y(), s.gb.z(), marker='.', color="green")

                idx = idx + 1
            s.sleep()
            s.best_in_swarm()
            Vis3D.plt.pause(2)

            s.ax.clear()
            s.vis_base()
            s.update()
            s.m = s.m + 1

        s.update()
        s.g = s.g + 1
        Vis3D.plt.pause(2)
        s.plt.clf()

    def best_in_swarm(s):
        new_swarm = list()
        for i in s.swarm:
            new_swarm.append(i.nl)
            s.ax.scatter(i.nl.x(), i.nl.y(), i.nl.z(), marker='o', color="black")

            if i.nl.z() < s.gb.z():
                s.gb = i.nl
        s.swarm = new_swarm

    def set_ptr(s, i):
        i.ptr_vec[0] = 1
        i.ptr_vec[1] = 1
        # rnd_j = random.choice([0, 1])
        # for idx in range(0, len(i.ptr_vec)):
        #     if rnd_j < s.ptr:
        #         i.ptr_vec[idx] = rnd_j
        #     else:
        #         i.ptr_vec[idx] = 0

    def algp(s, p):
        return s.alg(p.x(), p.y())

    def alg(s, dx, dy):
        return ackley(dx, dy, s.a, s.b, s.c, s.d)

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

        s.ax.plot_wireframe(s.X, s.Y, s.Z, cmap="coolwarm", zorder=1, linewidth=1)
        # s.ax.plot_surface(s.X, s.Y, s.Z, cmap="coolwarm", zorder=1, linewidth=1)
        s.ax.set_xlabel('x')
        s.ax.set_ylabel('y')
        s.ax.set_zlabel('z')
        pass

    def update(s):
        dbg("Iteration", Vis3D.frameNo)
        # Scale plot ax
        s.ax.set_xticks([])
        s.ax.set_yticks([])
        s.ax.set_title(s.__class__.__name__ + ", iter: {}/{}, frame #{} ".format(s.m, Vis3D.max_iterations, Vis3D.frameNo))

        # if(s.max_iterations - 5 < VisualizationBase3D.frameNo):

    def sleep(s):
        Vis3D.plt.pause(s.frameTimeout)
        Vis3D.frameNo += 1

    def updatep(s):
        zoffset = 10
        swarm = list(filter(lambda i: i != s.gb,s.swarm.copy()))
        s.dx = list(map(lambda i: i.x(), swarm))
        s.dy = list(map(lambda i: i.y(), swarm))
        s.dz = list(map(lambda i: i.z() + zoffset, swarm))
        s.ax.scatter(s.dx, s.dy, s.dz, marker='.', color="red")
        s.ax.scatter(s.gb.x(), s.gb.y(), s.gb.z(), marker='.', color="green")

    def updatev(s, i, idx=0):
        oz = s.algp(i.n[idx-1])
        nz = s.algp(i.n[idx])
        s.ax.plot([i.n[idx].x(), i.n[idx-1].x()], [i.n[idx].y(), i.n[idx-1].y()], zs=[nz, oz], color="red", linewidth=1)
        s.sleep()

    wc = 0.1
    phi_p = 0.1
    phi_g = 0.1
    v_max = 5
    v_mult = 5

    def gen_pop(s):
        s.swarm = []
        for i in range(0, s.points_cnt):
            pt = Pt(random.uniform(s.plane[0], s.plane[1]), random.uniform(s.plane[0], s.plane[1]))
            pt.p.append(s.algp(pt))
            s.swarm.append(pt)

            # pt = Pt(random.uniform(s.plane[0], s.plane[1]), random.uniform(s.plane[0], s.plane[1]))
            # s.swarm.append()
            # s.swarm[i].p[2] = s.algp(s.swarm[i])
        return s.swarm

    def is_out_range(s, x):
        min = s.plane[0]
        max = s.plane[1]
        if x > max:
            return True
        if x < min:
            return True
        return False

    def keep_in_range(s, x):
        min = s.plane[0]
        max = s.plane[1]
        if x > max:
            return max
        if x < min:
            return min
        return x

    def sel_best(s):
        bi = s.swarm[0]
        bv = s.algp(bi)
        bidx = 0
        idx = 0
        for p in s.swarm:
            nbv = s.algp(p)
            if nbv <= bv:
                bv = nbv
                bidx = idx
            idx = idx + 1
        return bidx


#######################################
# Functions definitions
#######################################

class Sphere(Vis3D):

    max_iterations = 5

    def alg(s, dx, dy):
        return (np.power(dx, 2) + (np.power(dy, 2)))

class Schwefel(Vis3D):
    """
    Problematic, neads tweaking of params for generating next generation
    """
    plane = [-500, 500, 500]
    # max_iterations = 20
    v_max = 50
    v_mult = 30

    c = 418.9829

    def alg(s, dx, dy):
        return (s.c * s.d - ( (dx * np.sin(np.sqrt(np.abs(dx))))) + (dy * np.sin(np.sqrt(np.abs(dy)))))

class Rosenbrock(Vis3D):
    def alg(s, dx, dy):
        return (100 * np.power((dy - np.power(dx, 2)), 2) + np.power((dx - 1), 2))

class Rastrigin(Vis3D):

    plane = [-5, 5, 30]

    v_max = 10
    v_mult = 10

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

    v_max = 10
    v_mult = 10

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
        return - (single(dx, 1) + single(dy, 2))

class Zakharov(Vis3D):

    plane = [-10, 10, 30]

    # max_iterations = 20

    def alg(s, dx, dy):
        def f1(x):
            return np.power(x, 2)
        def f2(x, i):
            return 0.5 * i * x
        def f3(x, i):
            return 0.5 * i * x
        def pw(x, i):
            return np.power(x, i)
        return f1(dx) + f1(dy) + pw(f2(dx, 1) + f2(dy, 2), 2) + pw(f3(dx, 1) + f3(dy, 2), 4)

class Ackley(Vis3D):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = 2

    # max_iterations = 20

    def alg(s, xi, yi):
        part1 = - s.a * np.exp((-s.b * np.sqrt((1.0 / s.d) * (np.power(xi, 2) + np.power(yi, 2)))))
        part2 = - np.exp((1.0 / s.d) * (np.cos(s.c * xi) + np.cos(s.c * yi)))
        return part1 + part2 + s.a + np.exp(1)


# Sphere, Schwefel, Rosenbrock, Rastrigin, Griewangk, Levy, Michalewicz, Zakharov, Ackley

plt.pause(2)

r = Sphere()
r = Schwefel()
r = Rosenbrock()
r = Rastrigin()
r = Griewangk()
r = Levy()
r = Michalewicz()
r = Zakharov()
r = Ackley()
# r = Vis3D()

exit(0)
