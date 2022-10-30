from base import *
import numpy as np

try:
    import seaborn as sns
except UserWarning:
    pass
import matplotlib.animation ; matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

def ackley(xi, yi, a, b, c, d):
    part1 = - a * np.exp((-b * np.sqrt((1.0 / d) * (np.power(xi, 2) + np.power(yi, 2)))))
    part2 = - np.exp((1.0 / d) * (np.cos(c * xi) + np.cos(c * yi)))
    return part1 + part2 + a + np.exp(1)

def f(xi, xi1):
    # Rosenbrock Function
    return (100 * np.power((xi1 - np.power(xi, 2)), 2) + np.power((xi - 1), 2))

class Vis3D(object):
    frameNo = 0

    frameTimeout = 0.01
    nxgraphOptions = None
    graphData = None
    G = None
    D = None
    plt = None
    layout = None
    fig = None
    ax = None

    x = None
    y = None
    z = None

    X = None
    Y = None
    Z = None

    dx = None
    dy = None
    dz = None

    ndx = None
    ndy = None
    ndz = None

    plane = [-32, 32, 60]
    points_cnt = 100
    max_iterations = 100

    data = []
    data_old = []

    def __init__(s):
        Vis3D.plt = plt
        Vis3D.frameNo = 0

        s.d = len(s.plane) - 1

        if Vis3D.fig is None:
            Vis3D.fig = plt.figure("BIA - #1 - hill-climbing")
            Vis3D.plt.clf()
        # else:
        # VisualizationBase3D.fig.clear()
        # s.ax.clear()

        ax = Vis3D.plt.axes(projection='3d')
        s.ax = ax

        s.vis_base()
        # dz = ackley(dx, dy, a, b, c, d) + 1

        s.dx = np.random.choice(s.x, s.points_cnt)
        s.dy = np.random.choice(s.y, s.points_cnt)
        s.dz = ackley(s.dx, s.dy, s.a, s.b, s.c, s.d) + (Vis3D.frameNo * 5)
        s.ax.scatter(s.dx, s.dy, s.dz, marker='o')

        def next_rand(u):
            w = s.plane[1] - s.plane[0]
            return (((np.random.normal(0, 0.1, 1)[0] * (w)) + u) % (w/2))



        for i in range(1, s.max_iterations):
            s.ax.clear()
            s.vis_base()

            s.ndx = next_rand(s.dx)
            s.ndy = next_rand(s.dy)
            for xid in range(s.dx.shape[0]):  # for each x-axis
                tmpx, tmpy = s.cmp(s.dx[xid], s.dy[xid], s.ndx[xid], s.ndy[xid])
                s.dx[xid] = tmpx
                s.dy[xid] = tmpy
            # recalc for whole array, and add 3 because 'zorder' does not work
            s.dz = s.alg(s.dx, s.dy)

            s.ax.scatter(s.dx, s.dy, s.dz, marker='o', zorder=10, color="red")
            s.update()
        # s.ax.set_title("")
        # s.ax.clear()
        s.plt.clf()


    def alg(s, dx, dy):
        # return ackley(dx, dy, s.a, s.b, s.c, s.d)
        return (100 * np.power((dy - np.power(dx, 2)), 2) + np.power((dx - 1), 2))

    def cmp(s, oldx, oldy, newx, newy):
        oldz = s.alg(oldx, oldy)
        newz = s.alg(newx, newy)
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
        dbg("Iteration", Vis3D.frameNo)
        # Scale plot ax
        s.ax.set_xticks([])
        s.ax.set_yticks([])
        s.ax.set_title(s.__class__.__name__ + ": Step #{} ".format(Vis3D.frameNo))

        Vis3D.plt.pause(s.frameTimeout)
        Vis3D.frameNo += 1
        # if(s.max_iterations - 5 < VisualizationBase3D.frameNo):
