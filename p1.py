# To run this code you can input your desired number of obstacles to be randomly generated in the obstacle_generator function. The number of RRT loops (n) can be specified
# at line 161. 
import random
import math
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import time
import statistics

#euclidean distance between two points
def distance(pos1,pos2):
    return math.sqrt((pos2[0]-pos1[0])**2+(pos2[1]-pos1[1])**2)

def sign(n):
    return (n > 0) - (n < 0)

#function for determining which grid cells are covered by line segment connecting two points.
#Inspired by Fast Voxel Traversal Algorithm by Woo
def traversed(A,B):
    (xA, yA) = A
    (xB, yB) = B
    dx = xB - xA
    dy = yB - yA
    (sx, sy) = (sign(dx), sign(dy)) #indicating whether X and Y are incremented or decremented as the edge crosses cell boundaries

    cellA = (math.floor(xA),math.floor(yA))
    cellB = (math.floor(xB),math.floor(yB))
    (x, y) = cellA #starting grid cell
    traversed = [cellA] #first cell added to traversed list is the cell containing point A

    # Compute tDeltaX and tDeltaY in units of t
    #indicates how far along the ray we must move (in units of t) for the horizontal component of such a movement to equal the width of a voxel
    tDeltaX = abs(1 / dx) if dx != 0 else float('inf')
    tDeltaY = abs(1 / dy) if dy != 0 else float('inf')

    #tMax is value of t at which the ray crosses the first vertical voxel boundary
    if dx != 0:
        tMaxX = (x + sx - xA) / dx
    else:
        tMaxX = float('inf')  # Never crosses a vertical boundary

    if dy != 0:
        tMaxY = (y + sy - yA) / dy
    else:
        tMaxY = float('inf')  # Never crosses a horizontal boundary

    while (x,y) != cellB:

        if tMaxX < tMaxY:
            tMaxX = tMaxX + tDeltaX
            x += sx
        else:
            tMaxY= tMaxY + tDeltaY
            y += sy

        # BOUNDARY CHECK: Ensure x and y stay in the grid
        if x < 0.0 or x > 9.5 or y < 0.0 or y > 9.5:
            break  # Stop if outside grid bounds

        traversed.append( (x,y) )

    return traversed

#nearest vertex
def nearest_vertex(graph,point):
    return min(graph.nodes, key=lambda node: distance(point,node))

#steer function finds point z that minimizes ||z-x_rand|| while remaining within epsilon from x_nearest
def steer(x_nearest,x_rand,epsilon,start_cell,c_free):
    min_d = float('inf')
    z = ()
    for p in c_free:
        if p != start_cell:
            d_epsilon = distance(p, x_nearest)
            d_p_to_pos2 = distance(p, x_rand)
            if d_p_to_pos2 < min_d:
                if d_epsilon < epsilon:
                    min_d = d_p_to_pos2
                    z = p
    return z

#function that checks whether or not the line segment connecting two points is obstacle free.
def obstacleFree(p1,p2,c_free):
    covered_cells = traversed(p1,p2)
    for cell in covered_cells:
        if cell not in c_free:
            return False
    return True
        

##############  RRT Algorithm ####################
def RRT(n, c_free, graph, epsilon, init_robot_cell, goal_pos, goal_rad):
    goal_found = False
    for i in range(n):
        x_rand_pt = random.choice(c_free)
        x_nearest = nearest_vertex(graph,x_rand_pt)
        x_new = steer(x_nearest, x_rand_pt, epsilon, init_robot_cell, c_free)
        if obstacleFree(x_nearest,x_new,c_free):
            graph.add_node(x_new)
            graph.add_edge(x_nearest,x_new)
        if distance(x_new,goal_pos)<=goal_rad:
            goal_found = True
            print("Goal Found!")
            return goal_found
    return goal_found


#get neighboring eight points since grid has resolution of 2
def getNeighbors(point):
    x,y = point
    return [(x+0.5,y),(x,y+0.5),(x-0.5,y),(x,y-0.5),(x+0.5,y+0.5),(x+0.5,y-0.5),(x-0.5,y-0.5),(x-0.5,y+0,5)]
# OBSTACLE FUNCTION
def obstacle_generator(start_pos,goal_points,num_obs, random_bool, grid):
    obstacle_positions = set() #immediate points surrounding center point of rectangle
    obstacle_centers = set()
    if random_bool == True:
        for i in range(num_obs):
            r = random.choice(grid)
            while r in goal_points or r == start_pos or r in obstacle_positions:
                r = random.choice(grid)
            obstacle_centers.add(r)
            for n in getNeighbors(r):
                if n in grid:
                    obstacle_positions.add(n)
        return obstacle_centers, obstacle_positions
    elif random_bool == False and num_obs == 1:
        p = (5,5)
        obstacle_centers.add((p))
        for n in getNeighbors(p):
            obstacle_positions.add(n)
        return obstacle_centers, obstacle_positions
    else:
        raise ValueError("obstacle generator only generates one object non randomly at position (5,5). Please enter num_obs = 1 for non random generation")

def grid(x_range, y_range, resolution):
    finer_grid = []
    for y in range(y_range):
        for x in range(x_range):
            # Divide each cell into smaller cells based on the resolution
            for dx in range(resolution):
                for dy in range(resolution):
                    # Generate the finer points (dx and dy are the sub-cell positions)
                    finer_x = x + dx / resolution
                    finer_y = y + dy / resolution
                    finer_grid.append((finer_x, finer_y))
    return finer_grid

def plot_path_to_goal(obs_centers, graph):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    plt.xlim(0,10)
    plt.ylim(0,10)
    # Plot obstacles
    for obs_p in obs_centers:
        obstacle_center = (obs_p[0]- 0.5, obs_p[1] - 0.5)
        obstacle = Rectangle(obstacle_center, 1, 1, color='red', alpha=0.6)
        ax.add_patch(obstacle)

    #plot goal
    goal = Circle((9,9),0.5,fill=False)
    ax.add_patch(goal)

    #plot nodes
    x_nodes, y_nodes = zip(*graph.nodes)
    plt.scatter(x_nodes, y_nodes, color='black', s = 20)

    #plot edges
    for edge in graph.edges:
        node1, node2 = edge
        x_vals = [node1[0], node2[0]]
        y_vals = [node1[1], node2[1]]
        plt.plot(x_vals, y_vals, 'k-',linewidth=2)


###### RRT PARAMETERS #########
# Create a finer grid by dividing each cell into 2x2 sub-cells
resolution = 2  # Dividing each cell into 2x2 smaller cells (4 points per cell)
finer_grid = grid(x_range=10, y_range=10, resolution=2)

#OBSTACLE
init_robot_pos = (0.5, 0.5)
goal_points = [(9.0,9.0),(9.0,9.5),(9.5,9.0),(9.5,9.5),(8.5,9.0),(8.5,9.5),(9.0,8.5),(8.5,8.5)]
obs_centers, obs_positions = obstacle_generator(init_robot_pos,goal_points,8,True,finer_grid)

#init c_free by removing obstacle from grid
# Initialize c_free by excluding obstacle positions from grid
c_free = [cell for cell in finer_grid if cell not in obs_positions and cell not in obs_centers]

n = 50
epsilon = 1.5
init_robot_cell = (0,0)
goal_pos = (9,9)
goal_rad = 0.5

######  RUNTIMES AND FIGURES
runtimes = []
for _ in range(20):
    start = time.time()
    graph = nx.Graph()
    graph.add_node(init_robot_pos)      
    goal_found = RRT(n, c_free, graph, epsilon, init_robot_cell, goal_pos, goal_rad)
    end = time.time()
    if goal_found == True:
        print("Goal found!")
        plot_path_to_goal(obs_centers,graph)
    runtime = end - start  # Calculate the runtime
    runtimes.append(runtime)  # Store the runtime



avg_runtime = statistics.mean(runtimes)
var_runtime = statistics.variance(runtimes)

print("avg_runtime = " + str(avg_runtime))
print("variance_runtime = " + str(var_runtime))


plt.show()