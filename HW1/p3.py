#To run this code input desired number of obstacles into obstacle_generator at line 271 and RRT star parameters at line 276
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

#randomly sample for c_free and avoids sampling start position
def rand_x(init_robot_pos, c_free):
    r = r = random.choice(c_free)
    while r == init_robot_pos:
        r = random.choice(c_free)
    return r

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
            if d_p_to_pos2 < min_d and d_epsilon < epsilon:
                min_d = d_p_to_pos2
                z = p
    return z

#function that checks whether or not the line segment connecting two points is obstacle free.
def CollisionFree(p1,p2,c_free):
    covered_cells = traversed(p1,p2)
    for cell in covered_cells:
        if cell not in c_free:
            return False
    return True

#given graph, returns vertices that are within distance r from p
def near(graph,p,r):
    vertices_near = []
    for node in graph.nodes:
        if distance(node,p)<r:
            vertices_near.append(node)
    return vertices_near

#returns parent of node
def parent_node(graph, node):
    parent = list(graph.predecessors(node)) #list of 1 tuple if node has parent
    if len(parent) > 0:
        return parent[0]
    return None

#calculate cost by finding distance along path from start node in graph to given node
def cost(graph,node,robot_init_pos):
    total_cost = 0
    while node != robot_init_pos:  # Traverse back to the start node
        parent = parent_node(graph, node)
        if parent == None:
            break
        total_cost += distance(node, parent)
        node = parent
    return total_cost

##############  RRT Star Algorithm ####################
def RRT_star(n, c_free, graph, epsilon, r_star, init_robot_cell, init_robot_pos, goal_pos, goal_rad):
    for i in range(n):
        goal_found = False
        path = []
        x_rand_pt = rand_x(init_robot_pos, c_free)
        x_nearest = nearest_vertex(graph,x_rand_pt)
        x_new = steer(x_nearest, x_rand_pt, epsilon, init_robot_cell, c_free)
        if CollisionFree(x_nearest,x_new, c_free):
            X_near = near(graph,x_new,min(r_star,epsilon)) #list of vertices that are within distance min(r_star, epsilon) of x_new
            graph.add_node(x_new)
            c_min = cost(graph,x_nearest, init_robot_pos) + distance(x_nearest,x_new)
            # Extend along minimum cost path
            x_min = x_nearest #by default set x_min to x_nearest before looking for lower cost options
            for x_near in X_near:
                if x_new != x_near and CollisionFree(x_near,x_new, c_free):
                    c_near_new = cost(graph, x_near, init_robot_pos) + distance(x_near, x_new)
                    if c_near_new < c_min:
                        x_min = x_near
                        c_min = c_near_new

            graph.add_edge(x_min, x_new)


            # Goal check: if x_new is close enough to the goal, break
            if distance(x_new, goal_pos) <= goal_rad:
                print("Goal reached!")
                goal_found = True
                path = get_path(graph, init_robot_pos, x_new)
                return (goal_found, path)

            #Rewire the tree
            for x_near in X_near:
                if CollisionFree(x_new,x_near,c_free):
                    if cost(graph,x_new,init_robot_pos) + distance(x_new, x_near) < cost(graph,x_near,init_robot_pos):
                        x_parent = parent_node(graph, x_near)
                        # Check if adding the edge would create a cycle
                        graph.add_edge(x_new, x_near)  # Temporarily add the edge
                        try:
                            # Try to find a cycle with the newly added edge
                            nx.find_cycle(graph, orientation="ignore")
                            # If a cycle is found, remove the edge and continue to the next x_near
                            graph.remove_edge(x_new, x_near)
                            continue
                        except nx.NetworkXNoCycle:
                            # No cycle found, safe to proceed with updating the graph
                            pass
                        
                        # If no cycle, remove old edge and add new one
                        if x_parent != x_new:
                            graph.remove_edge(x_parent, x_near)
                            graph.add_edge(x_new, x_near)
    return False, []

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

#get neighboring eight points since grid has resolution of 2
def getNeighbors(point):
    x,y = point
    return [(x+0.5,y),(x,y+0.5),(x-0.5,y),(x,y-0.5),(x+0.5,y+0.5),(x+0.5,y-0.5),(x-0.5,y-0.5),(x-0.5,y+0,5)]

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

#PLOTS OBSTACLES, GOAL, NODES AND EDGES
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

#FINDS PATH FROM START TO GOAL
def get_path(graph, start, goal):
    # Use DFS to find the path from start to goal
    predecessors = nx.dfs_predecessors(graph, start)
    
    # Check if goal is in the predecessors (i.e., if it was visited during DFS)
    if goal not in predecessors:
        print(f"No path found from {start} to {goal}")
        return []  # No path found
    
    path = []
    current_node = goal
    
    while current_node != start:
        path.append(current_node)
        current_node = predecessors[current_node]
    path.append(start)
    path.reverse()  # Reverse to get path from start to goal
    return path

###### RRT PARAMETERS #########
# Create a finer grid by dividing each cell into 2x2 sub-cells
resolution = 2  # Dividing each cell into 2x2 smaller cells (4 points per cell)
finer_grid = grid(x_range=10, y_range=10, resolution=2)

#OBSTACLE
init_robot_pos = (0.5, 0.5)
goal_points = [(9.0,9.0),(9.0,9.5),(9.5,9.0),(9.5,9.5),(8.5,9.0),(8.5,9.5),(9.0,8.5),(8.5,8.5)]
obs_centers, obs_positions = obstacle_generator(init_robot_pos,goal_points,8,True,finer_grid)

# Initialize c_free by excluding obstacle positions from grid
c_free = [cell for cell in finer_grid if cell not in obs_positions and cell not in obs_centers]

n = 20
epsilon = 1.5
r_star = 4.0
init_robot_cell = (0,0)
goal_pos = (9,9)
goal_rad = 0.5

######  RUNTIMES AND FIGURES
runtimes = []
path_lengths = []
for _ in range(20):
    start = time.time()
    graph = nx.DiGraph()
    graph.add_node(init_robot_pos)      
    goal_found, path = RRT_star(n, c_free, graph, epsilon, r_star, init_robot_cell, init_robot_pos, goal_pos, goal_rad)
    end = time.time()
    if goal_found == True:
        print("Goal found!")
        path_length = sum(distance(path[i], path[i+1]) for i in range(len(path)-1))
        path_lengths.append(path_length)
        #plot_path_to_goal(obs_centers,graph)
    runtime = end - start  # Calculate the runtime
    runtimes.append(runtime)  # Store the runtime


# Filter out paths that did not reach the goal
valid_path_lengths = [length for length in path_lengths if length > 0]
if len(valid_path_lengths)==0:
    print("No paths found to goal")
else:
    avg_path_length = statistics.mean(valid_path_lengths)
    print("Average Path Length: ", avg_path_length)

avg_runtime = statistics.mean(runtimes)
var_runtime = statistics.variance(runtimes)


print("avg_runtime = " , str(avg_runtime))
print("variance_runtime = " , str(var_runtime))

iterations = [30, 40, 50, 60 ,70, 80]
avg_path_lengths = [11.724635706548108, 13.25359690286135, 16.871630891611247, 12.990028897755842, 13.132276559301708, 14.548681024881303]
fig1 = plt.figure(figsize=(6,6))
plt.plot(iterations, avg_path_lengths)
plt.xlabel('# of iterations')
plt.ylabel('mean path length')


plt.show()

