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

    def __init__(s, nxgraphType=None, nxgraphOptions=None, graphData=None, isDirected=False, frameTimeout=1):
        s.plt = plt

        s.fig = plt.figure()
        # ax1 = plt.axes(projection='3d')
        ax = s.plt.axes(projection='3d')

        # Data for a three-dimensional line
        # zline = np.linspace(0, 15, 1000)
        # xline = np.sin(zline)
        # yline = np.cos(zline)
        # ax.plot3D(xline, yline, zline, 'gray')

        # def ackley(xi, yi, a, b, c, d):
        #     part1 = - a * np.exp((-b * np.sqrt((1.0/d) * np.power(xi, 2))))
        #     part2 = - np.exp((1.0/d) * np.cos(c*xi))
        #     return part1 + part2 + a + np.exp(1)

        def ackley(xi, yi, a, b, c, d):
            part1 = - a * np.exp((-b * np.sqrt((1.0/d) * (np.power(xi, 2) + np.power(yi, 2)))))
            part2 = - np.exp((1.0/d) * (np.cos(c*xi) + np.cos(c*yi)))
            return part1 + part2 + a + np.exp(1)

        def f(xi, xi1):
            # Rosenbrock Function
            return ( 100 * np.power((xi1 - np.power(xi, 2)),2) + np.power((xi - 1), 2))

        x = np.linspace(-32, 32, 60)
        y = np.linspace(-32, 32, 60)

        # X, Y = np.meshgrid(x, y)
        # Z = f(X, Y)
        # Z = f(X, Y)
        # Recommended variable values are: a = 20, b = 0.2 and c = 2π.
        # Z = ackley(X, 20, 0.2, 2 * np.pi, 3)
        # Z = ackley(X, 20, 0.2, 2 * np.pi, 2)
        # Y2 = ackley(Y, 20, 0.2, 2 * np.pi, 2)
        # ax.plot_surface(X, Y, Z)
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = 2
        X,Y = np.meshgrid(x,y)
        # pprint(X)
        # Y2 = ackley(X,a,b,c,d)
        Z2 = ackley(X, Y,a,b,c,d)

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.contour3D(X, Y, Z, 50, cmap='binary')
        ax.plot_surface(X, Y, Z2, cmap="coolwarm")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # ax2 = s.fig.add_subplot(projection='3d')

        def randrange(n, vmin, vmax):
            """
            Helper function to make an array of random numbers having shape (n, )
            with each number distributed Uniform(vmin, vmax).
            """
            return (vmax - vmin) * np.random.rand(n) + vmin
        n = 100
        for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
            xs = randrange(n, 23, 32)
            ys = randrange(n, 0, 100)
            zs = randrange(n, zlow, zhigh)
            ax.scatter(xs, ys, zs, marker=m)

        plt.show()

    def alg(self, G):
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
