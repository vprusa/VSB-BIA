from pprint import pprint

try:
    import seaborn as sns

    pass
except UserWarning:
    pass
import matplotlib.animation;

import sys
import argparse


def printHelp():
    print("printHelp")
    pass


def parseArgs():
    # Make parser object
    p = argparse.ArgumentParser(description="""Args parser""", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--algorithm", "--a", required=True, help="Use algorithm")
    p.add_argument("--lecture", "--l", required=True, help="Algorithm from lecture")
    p.add_argument("--keepRunning", "--kr", default=5, type=float, help="Keep visualization running")
    p.add_argument("--frameTimeout", "--t", default=1, type=float, help="Visualization frame timeout")
    p.add_argument("--directed", "--dir", action='store_true', help="Default Undirected graph")
    # p.add_argument("-h", "--help", help="Prints help")

    group1 = p.add_mutually_exclusive_group(required=True)
    group1.add_argument("--graphData", "--d", help="Use defined graph")
    group1.add_argument("--graph", "--g", help="Use networkx graph generation")
    group1.add_argument("--graphOptions", "--go", required=False, help="Networkx graph options")
    # group1.add_argument_group(nxgraphGroup)

    return (p.parse_args())


r = None
globR = None


def run(args):
    matplotlib.use("TkAgg")

    lecture = args.lecture.lower()
    algName = args.algorithm
    package = "algorithms." + lecture + "." + algName + "." + algName
    name = algName
    algClass = getattr(__import__(package, fromlist=[name]), name)
    r = algClass(nxgraphType=args.graph, nxgraphOptions=args.graphOptions, graphData=args.graphData,
                 isDirected=args.directed, frameTimeout=args.frameTimeout)
    try:
        print(r.G.edges(data=True))
        r.alg(r.G)
    except:
        exc_info = sys.exc_info()
        pprint(exc_info)
        # r.plt.pause(args.keepRunning)
    # exit(0)

    pass


"""
Exec:

.virtenv/bin/python3 ./run.py --l MST --a Boruvka --g cubical_graph --kr 10

.virtenv/bin/python3 ./run.py --l MST --a Boruvka --d \
"[(0, 1, {'weight': 15}), (0, 3, {'weight': 34}), (0, 4, {'weight': 25}), (1, 2, {'weight': 5}), \
(1, 7, {'weight': 23}), (2, 3, {'weight': 33}), (2, 6, {'weight': 29}), (3, 5, {'weight': 13}), \
(4, 5, {'weight': 5}), (4, 7, {'weight': 20}), (5, 6, {'weight': 38}), (6, 7, {'weight': 3})]"

"""
if __name__ == '__main__':
    if sys.version_info < (3, 0, 0):
        sys.stderr.write("You need python 3.0 or later to run this script\n")
        sys.exit(1)

    args = None
    try:
        args = parseArgs()
        print(args)
        print("source .virtenv/bin/activate && python3 ./run.py --l MST --a Boruvka --g cubical_graph --kr 10")
    except:
        exc_info = sys.exc_info()
        pprint(exc_info)
        printHelp()

    if args is not None:
        run(args)

    print()
