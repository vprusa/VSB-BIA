# import networkx as nx
from pprint import pprint
# import random
import matplotlib.pyplot as plt
from base import *
# import ast
import numpy as np

try:
    import seaborn as sns
except UserWarning:
    pass
import matplotlib.animation ; matplotlib.use("TkAgg")

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
# matplotlib.use("TkAgg")
# matplotlib.use("QtAgg")

class Point:
    x = 0
    y = 0
    z = 0

class VisualizationBase3D(object):
    frameNo = 0

    frameTimeout = 1
    nxgraphType = "cubical_graph"
    nxgraphOptions = None
    graphData = None
    G = None
    D = None
    plt = None
    layout = None
    fig = None
    ax = None

    plane = [-32, 32, 60]
    points_cnt = 50

    data: Point = []
    data_old: Point = []

    def __init__(s):
        s.plt = plt

        s.fig = plt.figure()
        ax = s.plt.axes(projection='3d')
        s.ax = ax

        def ackley(xi, yi, a, b, c, d):
            part1 = - a * np.exp((-b * np.sqrt((1.0/d) * (np.power(xi, 2) + np.power(yi, 2)))))
            part2 = - np.exp((1.0/d) * (np.cos(c*xi) + np.cos(c*yi)))
            return part1 + part2 + a + np.exp(1)

        def f(xi, xi1):
            # Rosenbrock Function
            return (100 * np.power((xi1 - np.power(xi, 2)),2) + np.power((xi - 1), 2))

        x = np.linspace(s.plane[0], s.plane[1], s.plane[2])
        y = np.linspace(s.plane[0], s.plane[1], s.plane[2])

        a = 20
        b = 0.2
        c = 2 * np.pi
        d = 2
        X,Y = np.meshgrid(x,y)
        Z2 = ackley(X, Y,a,b,c,d)

        ax.plot_surface(X, Y, Z2, cmap="coolwarm")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # xd = np.random.uniform(s.plane[0],s.plane[1],s.points_cnt)
        # yd = np.random.uniform(s.plane[0],s.plane[1],s.points_cnt)
        # def rand_point(x):
        #     # return x + (0.1 * np.random.randn(100))
        #     return x + (np.random.choice([-0.1, 0.1], 1) * 100)
        # dx = rand_point(xd)
        # dy = rand_point(yd)
        # # dy = Y # Y  (0.1 * np.random.randn(100))
        # dz = ackley(dx, dy, a, b, c, d)
        # s.ax.scatter(dx, dy, dz, marker='o')
        dx = np.random.choice(x, 32)
        dy = np.random.choice(y, 32)
        dz = ackley(dx, dy, a, b, c, d) + 1

        s.ax.scatter(dx, dy, dz, marker='o')

        plt.show()

    def alg(s):
        # xs = list(map(lambda p: p.x, s.data))
        # ys = list(map(lambda p: p.y, s.data))
        # zs = list(map(lambda p: p.z, s.data))
        # s.ax.scatter(xs, ys, zs, marker='o')

        pass
        # alg(self, G) should have algorithm-required inputs as parameters


    def update(s):
        s.ax.clear()

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
