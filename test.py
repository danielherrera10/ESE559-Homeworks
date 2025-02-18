import random
import math
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import time
import statistics



g = nx.DiGraph()
g.add_node((1,1))
g.add_node((4,4))
g.add_node((10,10))
g.add_node((20,20))
g.add_edge((1,1),(10,10))
g.add_edge((10,10),(20,20))
g.add_edge((1,1),(4,4))
preds = nx.dfs_predecessors(g,source=(1,1))
print(preds)
print(preds[(10,10)])


