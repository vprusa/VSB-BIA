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

class VisualizationBase3D(object):
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

    data = []
    data_old = []

    def __init__(s):
        s.plt = plt

        s.fig = plt.figure()
        ax = s.plt.axes(projection='3d')
        s.ax = ax

        s.alg_base()
        # dz = ackley(dx, dy, a, b, c, d) + 1

        s.dx = np.random.choice(s.x, s.points_cnt)
        s.dy = np.random.choice(s.y, s.points_cnt)
        s.dz = ackley(s.dx, s.dy, s.a, s.b, s.c, s.d) + (VisualizationBase3D.frameNo * 5)
        s.ax.scatter(s.dx, s.dy, s.dz, marker='o')

        def next_rand(u):
            # return (np.random.normal(0, 0.1, 1)[0] * (s.plane[1]-s.plane[0]) * u) % (s.plane[1]-s.plane[0]) + s.plane[0]
            w = s.plane[1] - s.plane[0]
            # return ((np.random.normal(0, 1, 1)[0] * (w) * u) % (w)) - (w / 2)
            return (((np.random.normal(0, 0.1, 1)[0] * (w)) + u) % (w/2))

        def cmp(oldx, oldy, newx, newy):
            oldz = ackley(oldx, oldy, s.a, s.b, s.c, s.d)
            newz = ackley(newx, newy, s.a, s.b, s.c, s.d)

            if oldz < newz:
                return oldx, oldy
            else:
                return newx, newy

        for i in range(1, 500):
            # dz = ackley(dx, dy, a, b, c, d) + (5*i)
            s.ax.clear()
            s.alg_base()

            s.ndx = next_rand(s.dx)
            s.ndy = next_rand(s.dy)
            # s.ndy = np.random.choice(s.y, s.points_cnt)
            # s.ndz = ackley(s.dx, s.dy, s.a, s.b, s.c, s.d) + (VisualizationBase3D.frameNo * 5)
            # s.dz = ackley(s.dx, s.dy, s.a, s.b, s.c, s.d) + (VisualizationBase3D.frameNo * 5)
            # s.dx = s.ndx
            # s.dy = s.ndy
            for xid in range(s.dx.shape[0]):  # for each x-axis
                tmpx, tmpy = cmp(s.dx[xid], s.dy[xid], s.ndx[xid], s.ndy[xid])
                s.dx[xid] = tmpx
                s.dy[xid] = tmpy
            # s.ndx, s.ndy = cmp(s.dx, s.dy, s.ndx, s.ndy)
            s.dz = ackley(s.dx, s.dy, s.a, s.b, s.c, s.d) + 1
            dbg("dz", s.dz)
            dbg("ndz", s.ndz)

            s.ax.scatter(s.dx, s.dy, s.dz, marker='o', zorder=10, color="red")
            s.update()

        plt.show()

    def alg_base(s):

        s.x = np.linspace(s.plane[0], s.plane[1], s.plane[2])
        s.y = np.linspace(s.plane[0], s.plane[1], s.plane[2])

        s.a = 20
        s.b = 0.2
        s.c = 2 * np.pi
        s.d = 2
        s.X, s.Y = np.meshgrid(s.x, s.y)
        s.Z = ackley(s.X, s.Y, s.a, s.b, s.c, s.d)
        # s.Z = ackley(s.X, s.Y, a, b, c, d)

        # s.ax.plot_surface(s.X, s.Y, s.Z, cmap="coolwarm", zorder=1)
        s.ax.plot_wireframe(s.X, s.Y, s.Z, cmap="coolwarm", zorder=1)
        s.ax.set_xlabel('x')
        s.ax.set_ylabel('y')
        s.ax.set_zlabel('z')
        # s.plt.show()
        pass
        # alg(self, G) should have algorithm-required inputs as parameters


    def update(s):
        dbg("edges", "XXX")
        # Scale plot ax
        s.ax.set_xticks([])
        s.ax.set_yticks([])
        s.ax.set_title("Step #{} ".format(VisualizationBase3D.frameNo))

        # self.plt.pause(5)
        s.plt.pause(s.frameTimeout)
        # self.plt.pause(3)
        VisualizationBase3D.frameNo += 1


    def updateSingle(s, edges=None):
        s.ax.clear()

        # Scale plot ax
        s.ax.set_xticks([])
        s.ax.set_yticks([])
        s.ax.set_title("Step #{} ".format(VisualizationBase3D.frameNo))

        # self.plt.pause(5)
        s.plt.pause(s.frameTimeout)
        # self.plt.pause(3)
        VisualizationBase3D.frameNo += 1
