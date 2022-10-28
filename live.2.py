import sys
from pprint import pprint
try:
    import seaborn as sns
    pass
except UserWarning:
    pass
import matplotlib.animation ; matplotlib.use("TkAgg")



from algorithms.ThreeD.Test.Test import Test as Test
r = Test()
r.alg(r.G)
# try:
#     r.alg(r.G)
# except:
#     exc_info = sys.exc_info()
#     pprint(exc_info)
# r.plt.pause(10)
exit(0)
